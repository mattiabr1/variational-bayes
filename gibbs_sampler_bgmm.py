"""
Collapsed Gibbs sampler for the Bayesian Gaussian Mixture Model
"""

import numpy as np
import random
from scipy.special import logsumexp
from scipy.stats import multivariate_t
import matplotlib.pyplot as plt
import timeit
from sklearn import cluster


class Gibbs_bgmm:

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

        # Latent variables inizialization

        if init_method=="random":
        # observations are assigned randomly to one of the K components
            self.assignments = np.random.randint(0, self.K, self.N)

        elif init_method=="kmeans":
        # assignments are initialized using K-means
            self.assignments = (cluster.KMeans(n_clusters=self.K, n_init=1, random_state=seed).fit(self.data).labels_)

        # Parameters inizialization

        self.N_k = np.zeros(self.K)
        self.m_numerator = np.full((self.K, self.D), self.beta0*self.m0)
        self.Winv_tmp = np.full((self.K, self.D, self.D),
            np.linalg.inv(self.W0) + self.beta0*np.outer(self.m0, self.m0)
        )

        self.beta = np.full(self.K, self.beta0)
        self.m = np.zeros((self.K, self.D))
        self.Winv = np.zeros((self.K, self.D, self.D))
        self.nu = np.full(self.K, self.nu0)

        for k in range(self.K):
            for n in np.where(self.assignments == k)[0]:
                self._add_statistics_nk(n, k)


    def _add_statistics_nk(self, n, k):
        # add sufficient statistics to new cluster
        self.N_k[k] += 1
        self.m_numerator[k] += self.data[n]
        self.Winv_tmp[k] += np.outer(self.data[n], self.data[n])

        self.beta[k] += 1
        self.m[k] = self.m_numerator[k] / self.beta[k]
        self.Winv[k] = self.Winv_tmp[k] - self.beta[k] * np.outer(self.m[k], self.m[k])
        self.nu[k] += 1

        self.assignments[n] = k


    def _remove_statistics_n(self, n):
        # remove sufficient statistics from old cluster
        k = self.assignments[n]
        self.assignments[n] = -1

        self.N_k[k] -= 1
        self.m_numerator[k] -= self.data[n]
        self.Winv_tmp[k] -= np.outer(self.data[n], self.data[n])

        self.beta[k] -= 1
        self.m[k] = self.m_numerator[k] / self.beta[k]
        self.Winv[k] = self.Winv_tmp[k] - self.beta[k] * np.outer(self.m[k], self.m[k])
        self.nu[k] -= 1


    def _log_ppd_n(self, n):
        # compute the log posterior predictive density for the observation x_n
        # while leaving x_n out
        logdensity = [multivariate_t.logpdf(
                self.data[n],
                loc = self.m[k],
                shape = (1+self.beta[k]) / ((self.nu[k]+1-self.D)*self.beta[k]) * self.Winv[k],
                df = self.nu[k]+1-self.D
            )
            for k in range(self.K)
        ]

        return logdensity


    def avg_log_predictive(self, test_data):
        # compute the average of the log predictive density (1.73) over all data points
        logdensity = [multivariate_t.logpdf(
            test_data,
            loc=self.m[k],
            shape=(1+self.beta[k]) / ((self.nu[k]+1-self.D)*self.beta[k]) * self.Winv[k],
            df=self.nu[k]+1-self.D
            )
            for k in range(self.K)
        ]
        # logsumexp trick https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        ppd = (-np.log(np.sum(self.alpha0 + self.N_k)) +
            logsumexp(np.log(self.alpha0 + self.N_k)[:,np.newaxis] + logdensity, axis=0)
        )
        return np.mean(ppd)


    def sampler(self, test_data, num_iter=200, burnin=50):

        history = { "clock": [], "avg_log_ppd": [], "init_clock": 0, "init_pred": 0 }

        history["init_pred"] = self.avg_log_predictive(test_data)
        history["init_clock"] = timeit.default_timer() - self.clock_start

        for iter in range(num_iter):

            obs_idx = list(range(self.N))
            random.shuffle(obs_idx)

            for n in obs_idx:

                self._remove_statistics_n(n)

                log_prob_zn = np.log(self.N_k + self.alpha0) + self._log_ppd_n(n)
                prob_zn = np.exp(log_prob_zn - logsumexp(log_prob_zn))  # normalization

                k = np.random.choice(range(self.K), size=1, p=prob_zn) # sample

                self._add_statistics_nk(n,k)

            history["avg_log_ppd"].append(self.avg_log_predictive(test_data))
            history["clock"].append(timeit.default_timer() - self.clock_start)

            print("iter", iter)


        return history
