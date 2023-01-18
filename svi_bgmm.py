"""
Stochastic Variational Inference for the Bayesian Gaussian mixture model
"""

import numpy as np
import scipy
from scipy.special import psi, gammaln, multigammaln, logsumexp
import math
import matplotlib.pyplot as plt
import timeit
import random
from sklearn import cluster


class SVI_bgmm:

    """
    Types of learning rate schedule:

        - "adaptive" (default) -- adaptive learning rate (Ranganath et al, 2013)
        Requires to specify an initial memory size *tau_0* for equation (2.58)

        - "decay_RM" -- Robbins-Monro decay learning rate
        Requires two parameters (equation 2.51):
            *delta* ∈ (0.5, 1] controls how quickly old information decays
            *omega* >= 0 down-weights early iterations
    """

    def __init__(
        self, data, num_components,
        init_method = "random", seed = random.seed(),
        mini_batch_size = 1024,
        learning_rate_schedule = "adaptive", tau_0 = 1000, delta = 0.9, omega = 1
    ):

        self.data = data
        self.N = data.shape[0]
        self.D = data.shape[1]
        self.K = num_components

        self.mini_batch_size = mini_batch_size
        self.learning_rate_schedule = learning_rate_schedule
        self.clock_start = timeit.default_timer()

        # Priors hyperparameters

        self.alpha0 = 1/self.K
        self.beta0 = 1
        self.m0 = np.zeros(self.D)
        self.W0 = np.eye(self.D)
        self.nu0 = self.D

        # Variational parameters initialization

        if init_method == "random":
        # responsibilities are initialized uniformly at random
            resp = np.random.rand(self.N,self.K)
            resp /= resp.sum(axis=1)[:,np.newaxis]
            self.log_resp = np.log(resp)

        elif init_method == "kmeans":
        # responsibilities are initialized using K-means
            epsilon = 1e-6
            resp = np.zeros((self.N,self.K)) + epsilon
            label = (cluster.KMeans(n_clusters=self.K, n_init=1, random_state=seed).fit(self.data).labels_)
            resp[np.arange(self.N), label] = 1-epsilon*(self.K-1)
            self.log_resp = np.log(resp)

        # compute the statistics on the full dataset (only for initialization)
        self._compute_resp_statistics(range(self.N))

        # other parameters are initialized on the basis of responsibilities
        self.alpha = np.zeros(self.K)
        self.beta = np.zeros(self.K)
        self.m = np.zeros((self.K,self.D))
        self.Winv = np.zeros((self.K,self.D,self.D))
        self.W_chol = np.zeros((self.K,self.D,self.D))
        self.nu = np.zeros(self.K)
        # in this case stochastic updates with learning rate equal to 1 are simply CAVI updates
        self._update_alpha(lr=1)
        self._update_beta(lr=1)
        self._update_m(lr=1)
        self._update_W(lr=1)
        self._update_nu(lr=1)

        # initialization of the learning rate parameters according to the schedule

        if self.learning_rate_schedule == "adaptive":
            self.tau = tau_0
            self._initmc_moving_avg()

        elif self.learning_rate_schedule == "decay_RM":
            self.delta = delta
            self.omega = omega


    def _compute_resp_statistics(self, indices):
        """
        evaluate the statistics of dataset in formulae (1.45),(1.46) and (1.47),
        which depend on the responsibilities values

        these quantities are rescaled appropriately to account for the number
        of minibatches used
        """

        self.N_k = self.N/len(indices) * (np.sum(np.exp(self.log_resp[indices]),0)) + np.finfo(
            np.exp(self.log_resp[indices]).dtype).eps
            # add a small quantity to ensure that the count of each cluster is non-zero

        self.xbar_k = self.N/len(indices) * np.dot(np.exp(self.log_resp[indices]).T, self.data[indices]) / self.N_k[:,np.newaxis]

        self.S_k = np.zeros((self.K,self.D,self.D))
        for k in range(self.K):
            diff = self.data[indices] - self.xbar_k[k]
            self.S_k[k] = self.N/len(indices) * np.dot(np.exp(self.log_resp[indices,k])*diff.T, diff) / self.N_k[k]


################################################################################
##  ADAPTIVE LEARNING RATE
################################################################################

    def _initmc_moving_avg(self):
        """
        initialize the moving averages through Monte Carlo estimates obtained by
        forming noisy gradients on several samples

        the number of Monte Carlo samples used is *tau_0* and
        each sample of length *size_mb* is drawn without replacement
        """

        size_mb = math.floor(self.N/self.tau)   # length sample

        self.gbar = np.zeros(self.K*(3+self.D+self.D**2))   # vectorized
        self.hbar = 0

        for mb in range(self.tau):

            indices = list(range(mb*size_mb, (mb+1)*size_mb))

            self._compute_resp_statistics(indices)
            gradient = self._compute_gradient()

            self.gbar += gradient
            self.hbar += np.dot(gradient.T, gradient)

        # Monte Carlo estimates
        self.gbar = self.gbar/self.tau
        self.hbar = self.hbar/self.tau

        self.rho = 0    # learning rate


    def _compute_gradient(self):
        """
        compute the vectorized natural gradient of the ELBO
        """

        gradient = np.concatenate([
            self.alpha0 + self.N_k - self.alpha,
            self.beta0 + self.N_k - self.beta,
            np.array(
                self.beta0 * self.m0 + self.N_k[:, np.newaxis] *
                self.xbar_k / self.beta[:, np.newaxis] - self.m
                ).flatten(),
            np.array([
                np.linalg.inv(self.W0) + self.N_k[k] * self.S_k[k] +
                self.beta0 * self.N_k[k] / self.beta[k] * np.outer(
                    self.xbar_k[k] - self.m0, self.xbar_k[k] - self.m0
                    ) - self.Winv[k] for k in range(self.K)
                ]).flatten(),
            self.nu0 + self.N_k - self.nu
        ])

        return gradient


    def _update_adaptive_lr(self):
        """
        update the learning rate and related quantities
        """

        gradient = self._compute_gradient()

        # (2.56)
        self.gbar = (1 - 1/self.tau)*self.gbar + 1/self.tau*gradient
        self.hbar = (1 - 1/self.tau)*self.hbar + 1/self.tau*np.dot(gradient.T, gradient)

        self.rho = np.dot(self.gbar.T, self.gbar) / self.hbar  # (2.57)
        self.tau = self.tau*(1-self.rho) + 1  # (2.58)


    def _update_alpha_adaptive(self):
        # (1.52)
        self.alpha = (1-self.rho)*self.alpha + self.rho*(self.alpha0 + self.N_k)

    def _update_beta_adaptive(self):
        # (1.54)
        self.beta = (1-self.rho)*self.beta + self.rho*(self.beta0 + self.N_k)

    def _update_m_adaptive(self):
        # (1.55)
        self.m = (1-self.rho)*self.m + self.rho*(self.beta0 * self.m0 + self.N_k[:, np.newaxis] * self.xbar_k) / self.beta[:, np.newaxis]

    def _update_W_adaptive(self):
        # (1.56)
        for k in range(self.K):
            diff = self.xbar_k[k] - self.m0
            self.Winv[k] = (1-self.rho)*self.Winv[k] + self.rho * (
                np.linalg.inv(self.W0) + self.N_k[k] * self.S_k[k]
                + self.beta0 * self.N_k[k] / self.beta[k] * np.outer(diff, diff)
            )
            cov_chol = scipy.linalg.cholesky(self.Winv[k], lower=True)
            # compute the Cholesky decomposition of precisions
            self.W_chol[k] = scipy.linalg.solve_triangular(
                cov_chol, np.eye(self.D), lower=True
            ).T

    def _update_nu_adaptive(self):
        # (1.57)
        self.nu = (1-self.rho)*self.nu + self.rho*(self.nu0 + self.N_k)


################################################################################
##  STANDARD STOCHASTIC GRADIENT ASCENT
################################################################################

    def _update_alpha(self, lr):
        # (1.52)
        self.alpha = (1-lr)*self.alpha + lr*(self.alpha0 + self.N_k)

    def _update_beta(self, lr):
        # (1.54)
        self.beta = (1-lr)*self.beta + lr*(self.beta0 + self.N_k)

    def _update_m(self, lr):
        # (1.55)
        self.m = (1-lr)*self.m + lr*(self.beta0 * self.m0 + self.N_k[:, np.newaxis] * self.xbar_k) / self.beta[:, np.newaxis]

    def _update_W(self, lr):
        # (1.56)
        for k in range(self.K):
            diff = self.xbar_k[k] - self.m0
            self.Winv[k] = (1-lr)*self.Winv[k] + lr * (
                np.linalg.inv(self.W0) + self.N_k[k] * self.S_k[k]
                + self.beta0 * self.N_k[k] / self.beta[k] * np.outer(diff, diff)
            )
            cov_chol = scipy.linalg.cholesky(self.Winv[k], lower=True)
            # compute the Cholesky decomposition of precisions
            self.W_chol[k] = scipy.linalg.solve_triangular(
                cov_chol, np.eye(self.D), lower=True
            ).T

    def _update_nu(self, lr):
        # (1.57)
        self.nu = (1-lr)*self.nu + lr*(self.nu0 + self.N_k)

################################################################################


    def _squared_mahalanobis_chol(self, nu, data, means, precisions_chol):
        """
        compute the squared Mahalanobis distance through the Cholesky decomposition
        and multiplies it by *nu*
        """
        qf = np.zeros((data.shape[0], means.shape[0]))
        for k in range(means.shape[0]):
            diff = np.dot(data, precisions_chol[k]) - np.dot(means[k], precisions_chol[k])
            qf[:,k] = nu[k] * np.sum(np.square(diff), axis=1)
        if qf.shape[0] == means.shape[0]: qf = np.diag(qf)
        return qf


    def _log_det_chol(self, precisions_chol):
        """
        compute the logarithmic determinant of the precision matrix through the
        Cholesky decomposition

        https://math.stackexchange.com/questions/3158303/using-cholesky-decomposition-to-compute-covariance-matrix-determinant
        """
        return [2 * np.sum(np.log(np.diag(precisions_chol[k]))) for k in range(np.shape(precisions_chol)[0])]


    def _log_pitilde(self):
        # (1.59), psi() is the digamma function (A.8)
        return psi(self.alpha) - psi(np.sum(self.alpha))

    def _log_Lambda(self):
        # (1.60)
        return (np.sum([psi(0.5*(self.nu-d)) for d in range(self.D)], axis=0
            ) + self.D*np.log(2.0) + self._log_det_chol(self.W_chol))


    def _update_log_responsibilities(self, indices):
        """
        update the logarithms of responsibilities, given the values of the other
        variational parameters

        we work with logarithms for numerical stability, since responsibilities
        are probabilities that may be very small in some cases
        """

        # (1.42)
        self.log_resp[indices] = (
            self._log_pitilde()
            + 0.5*self._log_Lambda()
            - 0.5*self.D*np.log(2*np.pi)
            - 0.5*(self.D/self.beta +
            self._squared_mahalanobis_chol(self.nu, self.data[indices], self.m, self.W_chol))
            )
        # normalization (1.44) using the log-sum-exp trick for numerical stability
        # https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        self.log_resp[indices] -= logsumexp(self.log_resp[indices], axis=1)[:, np.newaxis]



    def _log_dirichlet_constnorm(self):
        # (A.6)
        return gammaln(np.sum(self.alpha)) - np.sum(gammaln(self.alpha))

    def _log_wishart_constnorm(self):
        # (A.30)
        return -(0.5 * self.nu * self._log_det_chol(self.W_chol)
                + 0.5 * self.nu * self.D * np.log(2.0)
                + multigammaln(0.5*self.nu, self.D)
                )

    def _entropy_wishart(self):
        # (A.33)
        return -(self._log_wishart_constnorm()
            +0.5*(self.nu-self.D-1)*self._log_Lambda() - 0.5*self.nu*self.D
            )


    def compute_elbo(self, indices):
        """
        compute the lower bound (1.64) ignoring constants

        terms involving a sum over data points are appropriately rescaled
        to provide a stochastic version
        """

        return (self._elbo1(indices) + self._elbo2(indices) + self._elbo3() + self._elbo4()
                - self._elbo5(indices) - self._elbo6() - self._elbo7())

    def _elbo1(self, indices):
        # (1.65)
        return 0.5 * self.N/len(indices) * np.sum(
            np.exp(self.log_resp[indices])*(
            self._log_Lambda()
            - self.D/self.beta
            - self._squared_mahalanobis_chol(self.nu, self.data[indices], self.m, self.W_chol)
            - self.D * np.log(2*np.pi))
            )

    def _elbo2(self, indices):
        # (1.66)
        return self.N/len(indices) * np.sum(np.dot(np.exp(self.log_resp[indices]), self._log_pitilde()))

    def _elbo3(self):
        # (1.67)
        return (self.alpha0-1) * np.sum(self._log_pitilde())

    def _elbo4(self):
        # (1.68)
        return (0.5*np.sum(
            self._log_Lambda() - self.D*self.beta0/self.beta
            - self.beta0 * self._squared_mahalanobis_chol(self.nu, self.m, np.tile(self.m0,(self.K,1)), self.W_chol)
            + (self.nu0-self.D-1) * self._log_Lambda()
            - self.nu * [np.trace(np.dot(np.linalg.inv(self.W0),
                np.dot(self.W_chol[k], self.W_chol[k].T))) for k in range(self.K)
            ])
        )

    def _elbo5(self, indices):
        # (1.69)
        return self.N/len(indices) * np.sum(np.exp(self.log_resp[indices]) * self.log_resp[indices])

    def _elbo6(self):
        # (1.70)
        return np.sum((self.alpha-1) * self._log_pitilde()) + self._log_dirichlet_constnorm()

    def _elbo7(self):
        # (1.71)
        return np.sum(0.5*self._log_Lambda()
            + 0.5*self.D*np.log(self.beta)
            - self._entropy_wishart())


    def density_multivariate_t(self, x, mean, precision_chol, df):
        # compute the log density of the multivariate Student's t (A.27)
        logdet = self._log_det_chol(precision_chol)
        qf = self._squared_mahalanobis_chol(1/df,x,mean,precision_chol)

        logdensity = [
            gammaln(0.5*(df[k]+self.D)) - gammaln(0.5*df[k]) - 0.5*self.D*np.log(df[k]*np.pi)
            + 0.5*logdet[k] - 0.5*(df[k]+self.D)*np.log(1+qf[:,k])
            for k in range(self.K)
            ]
        return logdensity

    def avg_log_predictive(self, test_data):
        #  compute the average of the log predictive density (1.73) over all data points
        logdensity = self.density_multivariate_t(
            x=test_data,
            mean=self.m,
            precision_chol=[np.sqrt((self.nu[k]+1-self.D)*self.beta[k]/(1+self.beta[k])) *
                self.W_chol[k] for k in range(self.K)],
            df=self.nu+1-self.D
        )
        # logsumexp trick https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        ppd = (-np.log(np.sum(self.alpha)) +
            logsumexp(np.log(self.alpha)[:,np.newaxis] + logdensity, axis=0)
        )
        return np.mean(ppd)


    def _extract_mini_batches(self):
        """
        sample without replacement the indices defining all minibatches of an epoch
        """

        permutation = list(np.random.permutation(self.N))
        num_complete_minibatches = math.floor(self.N/self.mini_batch_size)
        mini_batch_indices = []

        for j in range(num_complete_minibatches):
            mini_batch_indices.append(
                permutation[j*self.mini_batch_size : (j+1)*self.mini_batch_size]
            )
        # handling the end case (last mini-batch < mini_batch_size)
        if self.N % self.mini_batch_size != 0:
            mini_batch_indices.append(
                permutation[int(self.N/self.mini_batch_size)*self.mini_batch_size : self.N]
            )

        return mini_batch_indices


    def fit(self, convergence_metric = "elbo", test_data = None, num_stop = 3, max_epochs = 100, tol = 1e-3):
        """
        fit the Bayesian Gaussian mixture model via SVI and, in case of
        convergence, return one dictionary with final results and one with history
        """

        results = {}
        history = {
            "resp": [], "cluster": [], "weights": [], "means": [], "covariances": [],
            "stepsize": [], "metric": [], "clock": [], "epoch_metric": [], "epoch_clock": [],
            "init_clock": 0, "init_metric": 0
        }

        epoch = 0
        stop = 0
        t = 1
        improvement = tol + 1

        if convergence_metric == "elbo":
            history["init_metric"] = self.compute_elbo(range(self.N))
        elif convergence_metric == "predictive density":
            history["init_metric"] = self.avg_log_predictive(test_data)
        history["init_clock"] = timeit.default_timer() - self.clock_start


        while True:

            mini_batch_indices = self._extract_mini_batches()

            for i in range(len(mini_batch_indices)):

                # Variational E-step
                self._update_log_responsibilities(mini_batch_indices[i])

                # Variational M-step
                self._compute_resp_statistics(mini_batch_indices[i])

                if self.learning_rate_schedule == "adaptive":

                    self._update_adaptive_lr()
                    self._update_alpha_adaptive()
                    self._update_beta_adaptive()
                    self._update_m_adaptive()
                    self._update_W_adaptive()
                    self._update_nu_adaptive()
                    history["stepsize"].append(self.rho)

                elif self.learning_rate_schedule == "decay_RM":

                    self._update_alpha(lr=(t+self.omega)**(-self.delta))
                    self._update_beta(lr=(t+self.omega)**(-self.delta))
                    self._update_m(lr=(t+self.omega)**(-self.delta))
                    self._update_W(lr=(t+self.omega)**(-self.delta))
                    self._update_nu(lr=(t+self.omega)**(-self.delta))
                    history["stepsize"].append((t+self.omega)**(-self.delta))


                if convergence_metric == "elbo":
                    history["metric"].append(self.compute_elbo(mini_batch_indices[i]))
                    #print(self.compute_elbo(mini_batch_indices[i]))
                elif convergence_metric == "predictive density":
                    history["metric"].append(self.avg_log_predictive(test_data))
                    #print(self.avg_log_predictive(test_data))

                history["clock"].append(timeit.default_timer() - self.clock_start)

                t += 1


            history["resp"].append(np.exp(self.log_resp))
            history["cluster"].append(self.cluster_assignments(np.exp(self.log_resp)))
            history["weights"].append(self.posterior_mixture_weights())
            history["means"].append(self.m)
            history["covariances"].append(self.S_k)
            history["epoch_metric"].append(np.mean(history["metric"][-len(mini_batch_indices):]))
            history["epoch_clock"].append(timeit.default_timer() - self.clock_start)
            print(np.mean(history["metric"][-len(mini_batch_indices):]))

            # convergence metric improvement
            if epoch > 1:
                improvement = history["epoch_metric"][epoch] - history["epoch_metric"][epoch-1]

            # if the convergence metric decreases for three consecutive iterations, we declare convergence
            if improvement < 0:
                stop += 1
            else:
                stop = 0

            # Convergence criterion
            if abs(improvement) < tol or stop >= num_stop:

                cond = (stop >= num_stop) * num_stop

                print("Converged at epoch {}".format(epoch - cond))
                print(convergence_metric, history["epoch_metric"][epoch - cond])
                plt.xlabel("Epochs")
                plt.ylabel(convergence_metric)
                plt.plot(history["epoch_metric"])
                plt.show()

                results["resp"] = history["resp"][epoch - cond]
                results["cluster"] = history["cluster"][epoch - cond]
                results["weights"] = history["weights"][epoch - cond]
                results["means"] = history["means"][epoch - cond]
                results["covariances"] = history["covariances"][epoch - cond]

                results["metric"] = history["epoch_metric"][epoch - cond]

                break

            epoch += 1
            print("epoch", epoch)

            if epoch > max_epochs:
                print("Maximum epoch reached, not converged")
                break

        return results, history


    def cluster_assignments(self,resp):
        # assign each observation to the mixture component that maximizes the
        # responsibility for the specific observation
        return np.argmax(resp, axis=1)

    def posterior_mixture_weights(self):
        # (1.62)
        return (self.alpha0 + self.N_k) / (self.K * self.alpha0 + self.N)
