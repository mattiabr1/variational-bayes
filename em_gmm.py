"""
Maximum likelihood Expectation Maximization algorithm for the Gaussian Mixture Model

Not efficient, it's only to do a (small) comparison with the CAVI Bayesian mixture
"""

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


class EM_gmm:

    def __init__(self, data, num_components):

        self.data = data
        self.N = data.shape[0]
        self.D = data.shape[1]
        self.K = num_components

        # parameters are randomly initialized
        self.mu_k = np.random.randn(self.K, self.D)
        A = np.random.rand(self.K, self.D, self.D)
        self.sigma_k = [np.dot(A[k], A[k].T) for k in range(self.K)]
        self.mixing_coef_k = np.repeat(1/self.K, self.K)

        self.N_k = np.zeros(self.K)
        self.resp = np.zeros((self.N, self.K))


    def _e_step(self):
        # Formula (1.24)
        for n in range(self.N):
            for k in range(self.K):
                self.resp[n,k] = (
                    self.mixing_coef_k[k] *
                    multivariate_normal.pdf(self.data[n], self.mu_k[k], self.sigma_k[k])
                )
        # normalization
        self.resp /= self.resp.sum(axis = 1)[:,np.newaxis]


    def _m_step(self):
        # (1.28)
        self.N_k = np.sum(self.resp,0)
        # (1.27)
        self.mu_k = np.dot(self.resp.T, self.data) / self.N_k[:,np.newaxis]
        # (1.29)
        for k in range(self.K):
            diff = self.data - self.mu_k[k]
            self.sigma_k[k] = np.dot(self.resp[:,k]*diff.T, diff) / self.N_k[k]
        # (1.32)
        self.mixing_coef_k = self.N_k / self.N


    def log_likelihood(self):
        # (1.21)
        logL = 0
        for n in range(self.N):
            logL += np.log(np.sum(
                [self.mixing_coef_k[k] *
                multivariate_normal.pdf(self.data[n], self.mu_k[k], self.sigma_k[k])
                for k in range(self.K)]
                ))
        return logL


    def fit(self, max_iter = 500, tol = 1e-3):
        """
        fit the Gaussian mixture model by means of the maximum likelihood EM
        algorithm and, in case of convergence, return a dictionary with final results

        in this case convergence is assessed by evaluating the log likelihood,
        also monitoring the parameters values works though
        """

        results = {}
        logL = []
        iter = 0

        while True:

            self._e_step()
            self._m_step()

            logL.append(self.log_likelihood())

            # log likelihood improvement
            improvement = logL[iter] - logL[iter-1] if iter > 0 else logL[iter]

            # Convergence criterion
            if iter > 0 and improvement < tol:

                print('Converged at iteration {}'.format(iter))
                print("log-likelihood", logL[-1])
                plt.plot(logL)
                plt.ylabel("log-likelihood")
                plt.xlabel("Iterations")
                plt.show()

                results["resp"] = self.resp
                results["cluster"] = self.cluster_assignments(self.resp)
                results["weights"] = self.mixing_coef_k
                results["means"] = self.mu_k
                results["covariances"] = self.sigma_k

                break

            iter += 1
            if iter % 10 == 0: print("iter", iter)

            if iter > max_iter:
                print('Maximum iteration reached, not converged')
                break

        return results


    def cluster_assignments(self,resp):
        # assign each observation to the mixture component that maximizes the
        # responsibility for the specific observation
        return np.argmax(resp, axis=1)
