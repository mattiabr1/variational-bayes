"""
Simulation study 1

Fit the variational Bayesian mixture to a synthetic dataset

Illustrate the evolution of the variational densities contours for the
mixture components as the CAVI algorithm progresses
"""

import itertools
import numpy as np
from scipy import linalg

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter

from cavi_bgmm import CAVI_bgmm


color_iter = itertools.cycle(["darkorange", "cornflowerblue", "c"])

def plot_results(data, true_labels, predicted_labels, means, covariances, nr_std, index, title):
    """
    true_labels -- actual clusters
    predicted_labels -- predicted clusters
    nr_std -- number of standard deviations for the density contours
    """
    splot = plt.subplot(2, 2, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * nr_std * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # mixture components to which no observation is assigned are not plotted
        if not np.any(predicted_labels == i):
            continue
        # plot the data, each point is colored according to its true cluster
        plt.scatter(data[true_labels==i, 0], data[true_labels==i, 1], 2, color=color)

        # plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%g'))
    plt.xticks([-5, -2.5, 0, 2.5, 5])
    plt.ylim(-6.0, 4.0)
    if index==0 or index==1:
        plt.xticks(())
    plt.title(title)


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
    Z = np.repeat([2,0,1], n_samples)

    history = CAVI_bgmm(X, 3).fit()[1]

    plot_results(X, Z, history["cluster"][0], history["means"][0], history["covariances"][0], 2, 0, "Initialization")
    plot_results(X, Z, history["cluster"][10], history["means"][10], history["covariances"][10], 2, 1, "Iteration 10")
    plot_results(X, Z, history["cluster"][22], history["means"][22], history["covariances"][22], 2, 2, "Iteration 22")
    plot_results(X, Z, history["cluster"][37], history["means"][37], history["covariances"][37], 2, 3, "Iteration 37 (converged)")

    plt.show()


if __name__ == "__main__":
    main()
