from tensorflow.keras.datasets import cifar10
from sklearn.cluster import KMeans
import click
from deeplearn import io
import numpy as np
import cv2


class VisualBagOfWords:

    def __init__(self):
        self.train_data = None
        self.test_data = None

    def build_bag(self):

        self.load_data()
        self.train_data = self.train_data[0:100]
        self.test_data = self.test_data[0:10]
        descriptor_list_training = []
        descriptor_list_test = []
        descriptor_list_training_stack = None

        click.echo("Extracting SIFT descriptors...")
        # For each training image extract desriptors using SIFT and store in descriptor_list
        i = 0
        for img in self.train_data:
            _, descriptors = cv2.xfeatures2d.SIFT_create().detectAndCompute(img, None)
            if descriptors is not None:
                descriptor_list_training.append(descriptors)
                if descriptor_list_training_stack is None:
                    descriptor_list_training_stack = descriptors
                else:
                    descriptor_list_training_stack = np.vstack((descriptor_list_training_stack, descriptors))

        # For each training image extract desriptors using SIFT and store in descriptor_list
        for img in self.test_data:
            _, descriptors = cv2.xfeatures2d.SIFT_create().detectAndCompute(img, None)
            if descriptors is not None:
                descriptor_list_test.append(descriptors)
        click.echo("SIFT descriptors extracted")

        # For every
        def build_histogram(descriptor_list, cluster_alg):
            histogram = np.zeros(len(cluster_alg.cluster_centers_))
            cluster_result = cluster_alg.predict(descriptor_list)
            for i in cluster_result:
                histogram[i] += 1.0
            return histogram

        # Cluster the descriptors together, every cluster will correspond to a visual word
        click.echo("KMeans clustering of descriptors...")
        kmeans = KMeans(n_clusters=20)
        kmeans.fit(descriptor_list_training_stack)
        click.echo("Clustering done")

        # For every image in both training and testing, build a histogram from the descriptors and which cluster they
        # belong to the values of the histogram is the number of occurences of the visual words
        click.echo("Constructing image histograms...")
        training_histograms = []
        for dsc in descriptor_list_training:
            histogram = build_histogram(dsc, kmeans)
            training_histograms.append(histogram)

        test_histograms = []
        for dsc in descriptor_list_test:
            histogram = build_histogram(dsc, kmeans)
            test_histograms.append(histogram)
        click.echo("[Done] Visual bag of words histograms created")

        io.export_bov(training_histograms, test_histograms)

    def load_data(self):
        click.echo('Reading data...')
        self.train_data = io.import_numpy('data/CIFAR-10_train_data.npy')
        self.test_data = io.import_numpy('data/CIFAR-10_test_data.npy')
        click.echo('[DONE] Reading data.')