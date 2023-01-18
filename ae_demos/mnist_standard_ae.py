"""
Demo of a simple standard autoencoder applied to MNIST data

Copyright: adapted from the repository of Udacity's Deep Learning v7 Nanodegree program
"""

import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# Set hyper-parameters ----------------------
mb_size = 16 # mini-batch size
h_dim = 50
lr = 1e-3  # learning rate
max_epochs = 20

path_to_data = 'mnist'


# Load Data ---------------------------------------------------------------

all_transforms = transforms.Compose([transforms.ToTensor()])

train_data = datasets.MNIST(path_to_data, train=True, download=True, transform=all_transforms)
test_data = datasets.MNIST(path_to_data, train=False, download=True, transform=all_transforms)

train_loader = DataLoader(train_data, batch_size=mb_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=mb_size, shuffle=True)


# Define the autoencoder architecture ======================================

class Autoencoder(nn.Module):

    def __init__(self, encoding_dim):

        super(Autoencoder, self).__init__()
        ## encoder ##
        # linear layer (784 -> encoding_dim)
        self.fc1 = nn.Linear(28 * 28, h_dim)

        ## decoder ##
        # linear layer (encoding_dim -> input size)
        self.fc2 = nn.Linear(h_dim, 28*28)

    def forward(self, x):
        # add layer, with relu activation function
        x = fn.relu(self.fc1(x))
        # output layer (sigmoid for scaling from 0 to 1)
        x = torch.sigmoid(self.fc2(x))
        return x

# Initialize the NN
model = Autoencoder(h_dim)

# =============================== TRAINING ====================================

criterion = nn.BCELoss()

solver = optim.Adam(model.parameters(), lr=lr)  # Adam optimizer

for it in range(max_epochs):  # Epochs

    train_loss = 0.0

    for data in train_loader:
        # _ stands in for labels, here
        images, _ = data
        # flatten images
        images = images.view(images.size(0), -1)
        # clear the gradients of all optimized variables
        solver.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(images)
        # calculate the loss
        loss = criterion(outputs, images)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        solver.step()
        # update running training loss
        train_loss += loss.item()*images.size(0)

    # print avg training statistics
    train_loss = train_loss/len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        it,
        train_loss
        ))

# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()

images_flatten = images.view(images.size(0), -1)
# get sample outputs
output = model(images_flatten)
# prep images for display
images = images.numpy()

# output is resized into a batch of images
output = output.view(mb_size, 1, 28, 28)
# use detach when it's an output that requires_grad
output = output.detach().numpy()

# plot the original images

fig = plt.figure(figsize=(4, 4))
gs = gridspec.GridSpec(4, 4)
gs.update(wspace=0.05, hspace=0.05)

for i, img in enumerate(images):
    ax = plt.subplot(gs[i])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(np.squeeze(img), cmap='Greys_r')

if not os.path.exists('out/'):
    os.makedirs('out/')
plt.savefig('out/images.png', bbox_inches='tight')
plt.close()

# plot the reconstruction images

fig = plt.figure(figsize=(4, 4))
gs = gridspec.GridSpec(4, 4)
gs.update(wspace=0.05, hspace=0.05)

for j, out in enumerate(output):
    ax = plt.subplot(gs[j])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(np.squeeze(out), cmap='Greys_r')

plt.savefig('out/reconstruction.png', bbox_inches='tight')

print("Done")
