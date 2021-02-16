import click
import cv2
import numpy as np
from keras import datasets
from os import path
from deeplearn import io
from sklearn.cluster import KMeans


class Model:

    def __init__(self, data):
        self.data = data
        self.train_data = None
        self.test_data = None
        self.test_labels = None
        self.train_labels = None

    def load_data(self):
        click.echo('Preparing {}...'.format(self.data))
        if self.data == 'CIFAR-10':
            if path.exists('data/cifar10_train_data.npy'):
                click.echo('{} already exists.'.format(self.data))
            else:
                click.echo('Fetching {}...'.format(self.data))
                (train_data, train_labels), (test_data, test_labels) = datasets.cifar10.load_data()
                click.echo('[DONE] Fetching {}.'.format(self.data))

                click.echo('Exporting dataset files for {}...'.format(self.data))
                io.export_data(self.data, train_data, train_labels, test_data, test_labels)
                click.echo('[DONE] Exporting dataset files for {}.'.format(self.data))
        elif self.data == 'FashionMNIST':
            if path.exists('data/fashion_mnist_train_data'):
                click.echo('{} already exists.'.format(self.data))
            else:
                click.echo('Fetching {}...'.format(self.data))
                (train_data, train_labels), (test_data, test_labels) = datasets.cifar10.load_data()
                click.echo('[DONE] Fetching {}.'.format(self.data))

                click.echo('Exporting dataset files for {}...'.format(self.data))
                io.export_data(self.data, train_data, train_labels, test_data, test_labels)
                click.echo('[DONE] Exporting dataset files for {}.'.format(self.data))
        click.echo('[DONE] Preparing {}.'.format(self.data))

    def feature_rep(self):
        pass

    def feature_extraction(self):
        pass

    def vbow(self):
        self.load_training_data()
        y_train = self.train_labels
        y_test = self.test_labels
        # self.train_data = self.train_data[0:100]
        # self.test_data = self.test_data[0:10]
        descriptor_list_training = []
        descriptor_list_test = []
        length = 0
        click.echo("Extracting SIFT descriptors...")
        # For each training image extract desriptors using SIFT and store in descriptor_list
        index = 0
        for img in self.train_data:
            _, descriptors = cv2.xfeatures2d.SIFT_create().detectAndCompute(img, None)
            if descriptors is not None:
                descriptor_list_training.append(descriptors)
                length += np.shape(descriptors)[0]
            else:
                y_train[index] = 255
            index += 1
        y_train = y_train[np.where(y_train != 255)]

        # For each training image extract desriptors using SIFT and store in descriptor_list
        index = 0
        for img in self.test_data:
            _, descriptors = cv2.xfeatures2d.SIFT_create().detectAndCompute(img, None)
            if descriptors is not None:
                descriptor_list_test.append(descriptors)
            else:
                y_test[index] = 255
            index += 1
        y_train = y_train[np.where(y_train != 255)]
        click.echo("SIFT descriptors extracted")

        # For every
        def build_histogram(descriptor_list, cluster_alg):
            histogram = np.zeros(len(cluster_alg.cluster_centers_))
            cluster_result = cluster_alg.predict(descriptor_list)
            for i in cluster_result:
                histogram[i] += 1.0
            return histogram

        def format(l):
            stack = np.zeros((length, np.shape(l[0])[1]), dtype=np.float32)
            index = 0
            for remaining in l[0:]:
                for row in remaining:
                    stack[index, :] = row
                    index += 1
            return stack

        descriptor_list_training_stack = format(descriptor_list_training)

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
        training_histograms = np.asarray(training_histograms)

        test_histograms = []
        for dsc in descriptor_list_test:
            histogram = build_histogram(dsc, kmeans)
            test_histograms.append(histogram)
        test_histograms = np.asarray(test_histograms)

        click.echo("[Done] Visual bag of words histograms created")
        io.export_bov(training_histograms, test_histograms, y_train, y_test)

    def load_training_data(self):
        click.echo('Reading data...')
        self.train_data = io.import_numpy('data/CIFAR-10_train_data.npy')
        self.test_data = io.import_numpy('data/CIFAR-10_test_data.npy')
        self.train_labels = io.import_numpy('data/CIFAR-10_train_labels.npy')
        self.test_labels= io.import_numpy('data/CIFAR-10_test_labels.npy')
        click.echo('[DONE] Reading data.')
