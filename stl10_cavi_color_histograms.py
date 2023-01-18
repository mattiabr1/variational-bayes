"""
Fit a Bayesian Gaussian mixture model to STL-10 data via CAVI
"""

from __future__ import print_function

import sys
import os, sys, tarfile, errno
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
from tqdm import tqdm

if sys.version_info >= (3, 0, 0):
    import urllib.request as urllib
else:
    import urllib
try:
    from imageio import imsave
except:
    from scipy.misc import imsave

from cavi_bgmm import CAVI_bgmm


# image shape
HEIGHT = 96
WIDTH = 96
DEPTH = 3

# size of a single image in bytes
SIZE = HEIGHT * WIDTH * DEPTH

# path to the directory with the data
DATA_DIR = "./stl10data"

# url of the binary data
DATA_URL = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"

# path to the binary train file with unlabeled image data
UNLAB_DATA_PATH = DATA_DIR + "/stl10_binary/unlabeled_X.bin"

def download_and_extract():
    """
    Download and extract the STL-10 dataset
    :return: None
    """
    dest_directory = DATA_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split("/")[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write("\rDownloading %s %.2f%%" % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        print("Downloaded", filename)
        tarfile.open(filepath, "r:gz").extractall(dest_directory)

def read_single_image(image_file):
    """
    This method uses a file as input instead of the path - so the
    position of the reader will be remembered outside of context of this method.
    :param image_file: the open file containing the images
    :return: a single image
    """
    # read a single image, count determines the number of uint8"s to read
    image = np.fromfile(image_file, dtype=np.uint8, count=SIZE)
    # force into image matrix
    image = np.reshape(image, (3, 96, 96))
    # transpose to standard format
    # You might want to comment this line or reverse the shuffle
    # if you will use a learning algorithm like CNN, since they like
    # their channels separated.
    image = np.transpose(image, (2, 1, 0))
    return image

def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, "rb") as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        return images

def save_image(image, name):
    imsave("%s.jpg" % name, image, format="jpg")

def save_images(images):
    i = 0
    for image in tqdm(images, position=0):
        directory = DATA_DIR + "/" + "unlabeled_img" + "/"
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                pass
        filename = directory + str(i)
        save_image(image, filename)
        i = i+1

def plot_image(image):
    """
    :param image: the image to be plotted in a 3-D matrix format
    :return: None
    """
    plt.imshow(image)
    plt.xticks(())
    plt.yticks(())
    plt.show()

def plot_image_color_histograms(image, bins, max_freq):
    """
    plot the RGB color histograms of a image
    :image: input image
    :bins: number of bins into which to divide the pixel intensities (the same for each color channel)
    :max_freq: maximum pixel counts (to adjust the plots)
    """

    colors = ("red", "green", "blue")
    channel_ids = (0, 1, 2)

    for channel_id, color in zip(channel_ids, colors):
        histogram, bin_edges = np.histogram(
            image[:, :, channel_id], bins=bins, range=(0, 256)
        )
        plt.subplot(3, 1, 1 + channel_id)
        plt.xlim([0, 256])
        plt.ylim([0, max_freq])
        plt.plot(bin_edges[0:-1], histogram, color=color)
        plt.xlabel("Pixel intensity")
        plt.ylabel("Pixel counts")

    plt.show()


def histogram_values(images, bins):
    """
    concatenate the pixel counts of the three RGB histograms, providing a
    3*bins-dimensional representation of each image

    :images: image dataset
    :bins: number of bins into which to divide the pixel intensities (the same for each color channel)
    """

    hist_values = np.zeros((len(images), 3*bins))
    channel_ids = (0, 1, 2)

    for i in range(len(images)):
        for ch in channel_ids:
            hist_values[i, range(ch*bins, (ch+1)*bins)], _ = np.histogram(
                images[i, :, :, ch], bins=bins, range=(0, 256)
            )

    return hist_values


def main():

    # download data if needed
    download_and_extract()

    images = read_all_images(UNLAB_DATA_PATH)
    # test to check if the whole dataset is read correctly
    # print(images.shape)

    # save images to disk
    # save_images(images)

    M = np.shape(images)[0]
    N = 10000
    bins = 64
    num_cluster = 30

    random.seed(770)

    # Randomly select N training images
    idx = random.sample(range(M), 2*N)
    train_images = images[idx[:N]]
    # test_images = images[idx[N:]]

    # Plot a sample image and its color histograms
    # plot_image(train_images[71])
    # plot_image_color_histograms(train_images[71], bins, 1000)

    ###
    # Obtain the data histograms to which to fit the Bayesian Gaussian mixture

    train_hist_values = histogram_values(train_images, bins)
    # test_hist_values = histogram_values(test_images, bins)

    bgmm = CAVI_bgmm(train_hist_values, num_cluster, init_method="kmeans").fit()[0]

    # Plot the nine most representative images from each of the mixture clusters

    max_resp = np.amax(bgmm["resp"], axis=1)
    df_estimates = pd.DataFrame({"Idx": range(N), "Maxresp": max_resp, "Cluster": bgmm["cluster"]})

    sorted_df = df_estimates.groupby(["Cluster"]).apply(lambda x: x.sort_values(["Maxresp"], ascending = False)).reset_index(drop=True)
    images_to_plot = sorted_df.groupby(["Cluster"]).head(9)

    for k in range(num_cluster):
        idx_cluster = images_to_plot.loc[images_to_plot["Cluster"] == k, "Idx"]
        for i in range(idx_cluster.shape[0]):
            splot = plt.subplot(3, 3, 1+i)
            plt.imshow(train_images[idx_cluster.iloc[i]])
            plt.xticks(())
            plt.yticks(())
        plt.show()



if __name__ == "__main__":
    main()
