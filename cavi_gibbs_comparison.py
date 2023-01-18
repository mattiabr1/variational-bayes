"""
Comparison between CAVI and collapsed Gibbs sampling in fitting a
Bayesian GMM to a toy dataset
"""

import numpy as np
import matplotlib.pyplot as plt

from gibbs_sampler_bgmm import Gibbs_bgmm
from cavi_bgmm import CAVI_bgmm


def main():

    n_samples = 500  # number of samples per component

    # Simulate 2-dimensional random Gaussians, k=3 components

    # Covariances matrices
    C1 = np.array([[0.9, 0.1], [-0.1, 0.7]])
    C2 = np.array([[1.0, 0.0], [0.0, 1.0]])
    C3 = np.array([[0.7, -0.2], [-0.1, 0.8]])

    # Means
    m1 = np.array([-2, -2])
    m2 = np.array([2, -3])
    m3 = np.array([0, 1])

    # Generate synthetic data
    np.random.seed(34)
    X = np.r_[
        np.dot(np.random.randn(n_samples, 2), C1) + m1,
        np.dot(np.random.randn(n_samples, 2), C2) + m2,
        np.dot(np.random.randn(n_samples, 2), C3) + m3,
        ]
    # True cluster assignments
    Z = np.repeat([1,2,0], n_samples)

    X_test = np.r_[
        np.dot(np.random.randn(n_samples, 2), C1) + m1,
        np.dot(np.random.randn(n_samples, 2), C2) + m2,
        np.dot(np.random.randn(n_samples, 2), C3) + m3,
        ]

    # GIBBS SAMPLING

    history_gibbs = Gibbs_bgmm(X, 3).sampler(test_data = X_test, num_iter=50)

    # CAVI

    history_cavi = CAVI_bgmm(X, 3).fit(convergence_metric = "predictive density", test_data=X_test, tol=1e-5)[1]


    clock_cavi = history_cavi["clock"]
    metric_cavi = history_cavi["metric"]
    clock_gibbs = history_gibbs["clock"]
    metric_gibbs = history_gibbs["avg_log_ppd"]

    clock_cavi.insert(0, history_cavi["init_clock"])
    metric_cavi.insert(0, history_cavi["init_metric"])
    clock_gibbs.insert(0, history_gibbs["init_clock"])
    metric_gibbs.insert(0, history_gibbs["init_pred"])


    fig, ax = plt.subplots(ncols=2, sharey="row")

    yticks = [-4.1,-4.0,-3.9,-3.8,-3.7,-3.6,-3.53]

    ax[0].plot(clock_cavi, metric_cavi, label= "CAVI")
    ax[0].set(xlabel="Seconds", ylabel="Log predictive density")
    ax[0].set_yticks(yticks)
    ax[0].legend()
    ax[0].grid(True, linestyle='--')

    ax[1].plot(clock_gibbs, metric_gibbs, label= "Gibbs", color="#d62728")
    ax[1].set(xlabel="Seconds")
    ax[1].legend()
    ax[1].set_yticks(yticks)
    ax[1].grid(True, linestyle='--')

    for a in fig.axes:
        a.tick_params(axis='y', labelleft=True)

    fig.subplots_adjust(wspace=0.25)

    plt.show()


if __name__ == "__main__":
    main()
