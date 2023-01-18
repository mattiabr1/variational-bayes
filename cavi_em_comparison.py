"""
Simulation study 2

Plot the confidence ellipsoids of a mixture of Gaussians obtained with
Expectation Maximization (*EM_gmm* class)
and compares it with the corresponding Bayesian model obtained via
Coordinate Ascent Variational Inference (*CAVI_bgmm* class).

Adapted from a scikit-learn library example
"""

import itertools
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from em_gmm import EM_gmm
from cavi_bgmm import CAVI_bgmm


color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])

def plot_results(data, cluster_assignments, means, covariances, nr_std, index, title):
    """
    cluster_assignments -- predicted clusters
    nr_std -- number of standard deviations for the density contours
    """
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * nr_std * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # mixture components to which no observation is assigned are not plotted
        if not np.any(cluster_assignments == i):
            continue
        plt.scatter(data[cluster_assignments==i, 0], data[cluster_assignments==i, 1], 0.8, color=color)

        # plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-9.0, 5.0)
    plt.ylim(-3.0, 6.0)
    if index==0: plt.xticks(())
    plt.title(title)


def main():

    # Number of samples per component
    n_samples = 500

    # Generate random sample, two components
    np.random.seed(0)
    C = np.array([[0.0, -0.2], [1.2, 0.4]])
    X = np.r_[
        np.dot(np.random.randn(n_samples, 2), C),
        0.7 * np.random.randn(n_samples, 2) + np.array([-6, 3]),
    ]

    # Fit a Gaussian mixture with EM using four components

    gmm = EM_gmm(X, 4).fit()

    # Fit a Bayesian Gaussian mixture with Dirichlet prior using four components

    dpgmm = CAVI_bgmm(X, 4).fit()[0]

    plot_results(X, gmm["cluster"], gmm["means"], gmm["covariances"], 2, 0,
    "EM Gaussian Mixture")

    plot_results(X, dpgmm["cluster"], dpgmm["means"], dpgmm["covariances"], 2, 1,
    "Variational Bayesian Gaussian Mixture")

    plt.show()

    print("posterior expected mixture weights", dpgmm["weights"])


if __name__ == "__main__":
    main()
