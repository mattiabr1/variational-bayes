# Variational Bayes

Python implementation of the algorithms used in my Master's thesis. Most of them concern inferential methods for the Gaussian Mixture Model, however some demos on (Variational) Autoencoders are also included.

## Content

### Main files

* `cavi_bgmm.py`: Coordinate Ascent Variational Inference for the Bayesian Gaussian Mixture Model.

* `em_gmm.py`: maximum likelihood Expectation Maximization algorithm for the Gaussian Mixture Model.

* `gibbs_sampler_bgmm.py`: collapsed Gibbs sampler for the Bayesian Gaussian Mixture Model.

* `svi_bgmm.py`: Stochastic Variational Inference for the Bayesian Gaussian Mixture Model.

### Analyses

* `cavi_simulation.py`: fit a BGMM to synthetic data via CAVI.

* `cavi_em_comparison.py`: compare the variational Bayesian estimation (CAVI) with the maximum likelihood EM algorithm in fitting a GMM to a toy dataset.

* `cavi_gibbs_comparison.py`: compare CAVI and collapsed Gibbs sampling in fitting a
Bayesian GMM to synthetic data.

* `stl10_cavi_color_histograms.py`: fit a BGMM to STL-10 data histograms via CAVI for clustering images according to their color profiles.

### Autoencoder demos

* `mnist_standard_ae.py`: demo of a simple standard Autoencoder applied to MNIST data.

* `mnist_vae.py`: demo of a simple Variational Autoencoder applied to MNIST data.

* `celeba_vae_ae_comparison.ipynb`:






