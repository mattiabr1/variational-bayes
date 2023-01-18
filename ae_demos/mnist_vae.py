"""
Demo of a simple variational autoencoder applied to MNIST data

Copyright: adapted from Arun Pandey, KU Leuven
Based on work done for the course Data Mining and Neural Networks
by Johan Suykens, KU Leuven
"""

# Import necessary libraries for this course ----------------------
import torch
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# Set hyper-parameters
mb_size = 100 # mini-batch size
Z_dim = 2  # latent-space dimension
h_dim = 400
c = 0
lr = 1e-3  # learning rate
max_epochs = 20


def mnist_dataloader(path_to_data='mnist'):
    """MNIST dataloader with (28, 28) images."""
    all_transforms = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(path_to_data, train=True, download=True, transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=mb_size, shuffle=True)
    return train_loader


def xavier_init(size):
    """Xavier inizialization"""
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)

# Load Data
mnist = mnist_dataloader()
_, channels, x, y = next(iter(mnist))[0].size()
X_dim = channels * x * y

# Q(z|X)

Wxh = xavier_init(size=[X_dim, h_dim])
bxh = Variable(torch.zeros(h_dim), requires_grad=True)

Whz_mu = xavier_init(size=[h_dim, Z_dim])
bhz_mu = Variable(torch.zeros(Z_dim), requires_grad=True)

Whz_var = xavier_init(size=[h_dim, Z_dim])
bhz_var = Variable(torch.zeros(Z_dim), requires_grad=True)


def Q(X):
    h = nn.relu(X @ Wxh + bxh.repeat(X.size(0), 1))
    z_mu = h @ Whz_mu + bhz_mu.repeat(h.size(0), 1)
    z_var = h @ Whz_var + bhz_var.repeat(h.size(0), 1)
    return z_mu, z_var


def sample_z(mu, log_var):
    eps = Variable(torch.randn(mb_size, Z_dim))  # sample from unit gaussian
    return mu + torch.exp(log_var / 2) * eps  # re-parameterization trick


# P(X|z)

Wzh = xavier_init(size=[Z_dim, h_dim])
bzh = Variable(torch.zeros(h_dim), requires_grad=True)

Whx = xavier_init(size=[h_dim, X_dim])
bhx = Variable(torch.zeros(X_dim), requires_grad=True)

def P(z):
    h = nn.relu(z @ Wzh + bzh.repeat(z.size(0), 1))
    X = torch.sigmoid(h @ Whx + bhx.repeat(h.size(0), 1))
    return X


# TRAINING
params = [Wxh, bxh, Whz_mu, bhz_mu, Whz_var, bhz_var,
          Wzh, bzh, Whx, bhx]

solver = optim.Adam(params, lr=lr)  # Adam optimizer

for it in range(max_epochs):  # Epochs
    avg_loss = 0
    for _, (X, _) in enumerate(tqdm(mnist, desc="Iter-{}".format(it))):

        X = X.view(mb_size, -1)

        # Forward
        z_mu, z_var = Q(X)
        z = sample_z(z_mu, z_var)

        # Sampling from random z
        X_sample = P(z)

        # Loss
        # E[log P(X|z)]
        recon_loss = nn.binary_cross_entropy(X_sample, X, reduction='sum') / mb_size
        # https://github.com/y0ast/VAE-TensorFlow/issues/3

        # D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1. - z_var, 1))
        loss = recon_loss + kl_loss

        # Backward
        loss.backward()

        # Update
        solver.step()

        # Housekeeping
        for p in params:
            if p.grad is not None:
                data = p.grad.data
                p.grad = Variable(data.new().resize_as_(data).zero_())
        avg_loss += loss / len(mnist)

    print('Loss: {:.4}'.format(avg_loss.item()))

    # plot sometimes
    if it % 2 == 0:
        samples = P(z).data.numpy()[:64]

        plt.close()
        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(8, 8)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        if not os.path.exists('out/'):
            os.makedirs('out/')
        plt.savefig('out/{}_{}.png'.format(it, str(c).zfill(3)), bbox_inches='tight')
        c += 1

    # plot the manifold

    if Z_dim == 2 and it == max_epochs-1:

        nx = ny = 15
        x_values = np.linspace(.05, .95, nx)
        y_values = np.linspace(.05, .95, ny)

        canvas = np.empty((28*ny, 28*nx))
        for i, yi in enumerate(x_values):
            for j, xi in enumerate(y_values):
                z_mu = np.array([[norm.ppf(xi), norm.ppf(yi)]]).astype('float32')
                x_mean = P(torch.from_numpy(z_mu))
                canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean.data.numpy()[0].reshape(28, 28)

        plt.close()
        plt.figure(figsize=(8, 10))
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(canvas, origin="upper", cmap="gray")
        plt.savefig('out/manifold.png', bbox_inches='tight')

print("Done")
