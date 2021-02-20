import click
import cv2
import numpy as np
from keras import datasets
from os import path
from deeplearn import io
from sklearn.cluster import KMeans
from PIL import Image


class Model:

    def __init__(self, data):
        self.data = data
        self.train_data = None
        self.test_data = None
        self.test_labels = None
        self.train_labels = None

    def fetch_dataset(self):
        if path.exists('data/{}_raw_train_data.npy'.format(self.data)):
            (self.train_data, self.train_labels), (self.test_data, self.test_labels) = io.import_data(self.data, 'raw')
            click.echo('Raw data already exists.')
        else:
            click.echo('[START] Fetching {}...'.format(self.data))
            if self.data == 'CIFAR-10':
                (self.train_data, self.train_labels), (self.test_data, self.test_labels) = datasets.cifar10.load_data()
            elif self.data == 'Fashion-MNIST':
                (self.train_data, self.train_labels), (self.test_data, self.test_labels) = datasets.fashion_mnist.load_data()
            io.export_data(self.data, 'raw', self.train_data, self.train_labels, self.test_data, self.test_labels)
            click.echo('[DONE] Fetched {}.'.format(self.data))

    def feature_representation(self):
        if path.exists('data/{}_rep_train_data.npy'.format(self.data)):
            click.echo('Feature representation data already exists.')
            return

        feature_data_train = []
        feature_data_test = []

        click.echo("[START] Creating feature representation for training data...")

        for fileName in self.train_data:
            fileImage = Image.fromarray(fileName)

            # ensure that all images are RGB
            fileImage = fileImage.convert('RGB')

            # extract feature to 1D array
            features = fileImage.histogram()

            feature_data_train.append(features)

        click.echo("[DONE] Created feature representation for training data.")
        click.echo("[START] Creating feature representation for test data...")

        for fileName in self.test_data:
            fileImage = Image.fromarray(fileName)

            # ensure that all images are RGB
            fileImage = fileImage.convert('RGB')

            # extract feature to 1D array
            features = fileImage.histogram()

            feature_data_test.append(features)

        click.echo("[DONE] Created feature representation for test data.")

        train_data = np.array(feature_data_train)
        test_data = np.array(feature_data_test)
        io.export_data(data=self.data, prefix='rep', train_data=train_data, test_data=test_data)

    def visual_bag_of_words(self):
        if path.exists('data/{}_vbow_train_data.npy'.format(self.data)):
            click.echo('Visual Bag of Words data already exists.')
            return

        click.echo('[START] Creating Visual Bag of Words...')
        y_train = self.train_labels.copy()
        y_test = self.test_labels.copy()
        descriptor_list_training = []
        descriptor_list_test = []
        length = 0
        click.echo("[START] Extracting SIFT descriptors...")
        # For each training image extract descriptors using SIFT and store in descriptor_list
        index = 0
        for img in self.train_data:
            _, descriptors = cv2.SIFT_create().detectAndCompute(img, None)
            if descriptors is not None:
                descriptor_list_training.append(descriptors)
                length += np.shape(descriptors)[0]
            else:
                y_train[index] = 255
            index += 1
        y_train = y_train[np.where(y_train != 255)]

        # For each training image extract descriptors using SIFT and store in descriptor_list
        index = 0
        for img in self.test_data:
            _, descriptors = cv2.xfeatures2d.SIFT_create().detectAndCompute(img, None)
            if descriptors is not None:
                descriptor_list_test.append(descriptors)
            else:
                y_test[index] = 255
            index += 1
        y_test = y_test[np.where(y_test != 255)]
        click.echo("[DONE] Extracted SIFT descriptors.")

        # For every
        def build_histogram(descriptor_list, cluster_alg):
            hist = np.zeros(len(cluster_alg.cluster_centers_))
            cluster_result = cluster_alg.predict(descriptor_list)
            for i in cluster_result:
                hist[i] += 1.0
            return hist

        def format_stack(descriptor):
            stack = np.zeros((length, np.shape(descriptor[0])[1]), dtype=np.float32)
            i = 0
            for remaining in descriptor[0:]:
                for row in remaining:
                    stack[i, :] = row
                    i += 1
            return stack

        descriptor_list_training_stack = format_stack(descriptor_list_training)

        # Cluster the descriptors together, every cluster will correspond to a visual word
        click.echo("[START] KMeans clustering of descriptors...")
        kmeans = KMeans(n_clusters=20)
        kmeans.fit(descriptor_list_training_stack)
        click.echo("[DONE] KMeans clustering.")

        # For every image in both training and testing, build a histogram from the descriptors and which cluster they
        # belong to the values of the histogram is the number of occurrences of the visual words
        click.echo("[START] Constructing image histograms...")
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

        click.echo("[DONE] Constructed histograms.")
        io.export_data(self.data, 'vbow', training_histograms, y_train, test_histograms, y_test)
        click.echo('[DONE] Created Visual Bag of Words.')
