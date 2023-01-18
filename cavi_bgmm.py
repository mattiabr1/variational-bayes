"""
Coordinate Ascent Variational Inference for the Bayesian Gaussian Mixture Model
"""

import numpy as np
import scipy
from scipy.special import psi, gammaln, multigammaln, logsumexp
import matplotlib.pyplot as plt
import timeit
import random
from sklearn import cluster


class CAVI_bgmm:

    def __init__(
        self, data, num_components,
        init_method = "random", seed = random.seed()
    ):

        self.data = data
        self.N = data.shape[0]
        self.D = data.shape[1]
        self.K = num_components

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

        self._compute_resp_statistics()

        # other parameters are initialized on the basis of responsibilities
        self.alpha = np.zeros(self.K)
        self.beta = np.zeros(self.K)
        self.m = np.zeros((self.K,self.D))
        self.W_chol = np.zeros((self.K,self.D,self.D))
        self.nu = np.zeros(self.K)
        self._update_alpha()
        self._update_beta()
        self._update_m()
        self._update_W()
        self._update_nu()


    def _compute_resp_statistics(self):
        """
        evaluate the statistics of dataset in formulae (1.45),(1.46) and (1.47),
        which depend on the responsibilities values
        """
        self.N_k = np.sum(np.exp(self.log_resp),0)

        self.xbar_k = np.dot(np.exp(self.log_resp).T, self.data) / self.N_k[:,np.newaxis]

        self.S_k = np.zeros((self.K,self.D,self.D))
        for k in range(self.K):
            diff = self.data - self.xbar_k[k]
            self.S_k[k] = np.dot(np.exp(self.log_resp[:,k])*diff.T, diff) / self.N_k[k]


    def _update_alpha(self):
        # (1.52)
        self.alpha = self.alpha0 + self.N_k

    def _update_beta(self):
        # (1.54)
        self.beta = self.beta0 + self.N_k

    def _update_m(self):
        # (1.55)
        self.m = (self.beta0 * self.m0 + self.N_k[:, np.newaxis] * self.xbar_k) / self.beta[:, np.newaxis]

    def _update_W(self):
        # (1.56)
        for k in range(self.K):
            diff = self.xbar_k[k] - self.m0
            cov_chol = scipy.linalg.cholesky(
                scipy.linalg.inv(self.W0) + self.N_k[k] * self.S_k[k] +
                self.beta0 * self.N_k[k] / self.beta[k] * np.outer(diff, diff),
                lower=True
            )
            # compute the Cholesky decomposition of precisions
            self.W_chol[k] = scipy.linalg.solve_triangular(
                cov_chol, np.eye(self.D), lower=True
            ).T


    def _update_nu(self):
        # (1.57)
        self.nu = self.nu0 + self.N_k


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


    def _update_log_responsibilities(self):
        """
        update the logarithms of responsibilities, given the values of the other
        variational parameters

        we work with logarithms for numerical stability, since responsibilities
        are probabilities that may be very small in some cases
        """
        # (1.42)
        self.log_resp = (
            self._log_pitilde()
            + 0.5*self._log_Lambda()
            - 0.5*self.D*np.log(2*np.pi)
            - 0.5*(self.D/self.beta +
            self._squared_mahalanobis_chol(self.nu, self.data, self.m, self.W_chol))
            )
        # normalization (1.44) using the log-sum-exp trick for numerical stability
        # https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        self.log_resp -= logsumexp(self.log_resp, axis=1)[:, np.newaxis]



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


    def compute_elbo(self):
        # compute the lower bound (1.64) ignoring constants
        return (self._elbo1() + self._elbo2() + self._elbo3() + self._elbo4()
                - self._elbo5() - self._elbo6() - self._elbo7())

    def _elbo1(self):
        # (1.65)
        return 0.5 * np.sum(self.N_k * (
            self._log_Lambda()
            - self.D/self.beta
            - self.nu * [np.trace(np.dot(self.S_k[k],
                np.dot(self.W_chol[k], self.W_chol[k].T))) for k in range(self.K)
            ]
            - self._squared_mahalanobis_chol(self.nu, self.xbar_k, self.m, self.W_chol)
            - self.D * np.log(2*np.pi))
            )

    def _elbo2(self):
        # (1.66)
        return np.sum(np.dot(np.exp(self.log_resp), self._log_pitilde()))

    def _elbo3(self):
        # (1.67)
        return (self.alpha0-1) * np.sum(self._log_pitilde())

    def _elbo4(self):
        # (1.68)
        return (0.5*np.sum(
            self._log_Lambda() - self.D*self.beta0/self.beta
            - self.beta0 * self._squared_mahalanobis_chol(self.nu, self.m, np.tile(self.m0,(self.K,1)), self.W_chol)
            + (self.nu0-self.D-1) * self._log_Lambda()
            - self.nu * [np.trace(np.dot(scipy.linalg.inv(self.W0),
            np.dot(self.W_chol[k], self.W_chol[k].T))) for k in range(self.K)
            ])
        )

    def _elbo5(self):
        # (1.69)
        return np.sum(np.exp(self.log_resp) * self.log_resp)

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
        # compute the average of the log predictive density (1.73) over all data points
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



    def fit(self, convergence_metric = "ELBO", test_data=None, num_stop = 3, max_iter = 500, tol = 1e-3):
        """
        fit the Bayesian Gaussian mixture model via CAVI and, in case of
        convergence, return one dictionary with final results and one with history
        """

        results = {}
        history = {
            "resp": [], "cluster": [], "weights": [], "means": [],
            "covariances": [], "metric": [], "clock": [],
            "init_clock": 0, "init_metric": 0
        }

        iter = 0
        stop = 0
        improvement = tol + 1

        if convergence_metric == "ELBO":
            history["init_metric"] = self.compute_elbo()
        elif convergence_metric == "predictive density":
            history["init_metric"] = self.avg_log_predictive(test_data)
        history["init_clock"] = timeit.default_timer() - self.clock_start


        while True:

            # Variational E-step
            self._update_log_responsibilities()

            # Variational M-step
            self._compute_resp_statistics()
            self._update_alpha()
            self._update_beta()
            self._update_m()
            self._update_W()
            self._update_nu()


            history["resp"].append(np.exp(self.log_resp))
            history["cluster"].append(self.cluster_assignments(np.exp(self.log_resp)))
            history["weights"].append(self.posterior_mixture_weights())
            history["means"].append(self.m)
            history["covariances"].append(self.S_k)

            if convergence_metric == "ELBO":
                history["metric"].append(self.compute_elbo())
                print(self.compute_elbo())
            elif convergence_metric == "predictive density":
                history["metric"].append(self.avg_log_predictive(test_data))
                print(self.avg_log_predictive(test_data))

            history["clock"].append(timeit.default_timer() - self.clock_start)

            # convergence metric improvement
            if iter > 1:
                improvement = history["metric"][iter] - history["metric"][iter-1]

            # if the convergence metric decreases for three consecutive iterations, we declare convergence
            if improvement < 0:
                stop += 1
            else:
                stop = 0

            # Convergence criterion
            if abs(improvement) < tol or stop >= num_stop:

                cond = (stop >= num_stop) * num_stop

                print("Converged at iteration {}".format(iter - cond))
                print(convergence_metric, history["metric"][iter - cond])
                plt.xlabel("Iterations")
                plt.ylabel(convergence_metric)
                plt.plot(history["metric"])
                plt.grid(True, linestyle='--')
                plt.show()

                results["resp"] = history["resp"][iter - cond]
                results["cluster"] = history["cluster"][iter - cond]
                results["weights"] = history["weights"][iter - cond]
                results["means"] = history["means"][iter - cond]
                results["covariances"] = history["covariances"][iter - cond]

                break

            iter += 1
            if iter % 10 == 0: print("iter", iter)

            if iter > max_iter:
                print("Maximum iteration reached, not converged")
                break

        return results, history


    def cluster_assignments(self,resp):
        # assign each observation to the mixture component that maximizes the
        # responsibility for the specific observation
        return np.argmax(resp, axis=1)

    def posterior_mixture_weights(self):
        # (1.62)
        return (self.alpha0 + self.N_k) / (self.K * self.alpha0 + self.N)
