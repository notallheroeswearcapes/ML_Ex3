import click
from keras import datasets
from os import path
from deeplearn import io


class Model:

    def __init__(self, data):
        self.data = data

    def load_data(self):
        click.echo('Preparing {}...'.format(self.data))
        if self.data == 'CIFAR-10':
            if path.exists('data/cifar10'):
                click.echo('{} already exists.'.format(self.data))
            else:
                click.echo('Fetching {}...'.format(self.data))
                (train_data, train_labels), (test_data, test_values) = datasets.cifar10.load_data()
                click.echo('\t[DONE] Fetching {}.'.format(self.data))
                click.echo('\tExporting dataset files for {}...'.format(self.data))
                io.export_numpy('data/cifar10/cifar10_train_data', train_data)
                io.export_numpy('data/cifar10/cifar10_test_data', test_data)
                io.export_numpy('data/cifar10/cifar10_train_labels', train_labels)
                io.export_numpy('data/cifar10/cifar10_test_labels', test_values)
                click.echo('\t[DONE] Exporting dataset files for {}.'.format(self.data))
        elif self.data == 'FashionMNIST':
            if path.exists('data/fashion_mnist'):
                click.echo('{} already exists.'.format(self.data))
            else:
                click.echo('Fetching {}...'.format(self.data))
                (train_data, train_labels), (test_data, test_values) = datasets.cifar10.load_data()
                click.echo('\t[DONE] Fetching {}.'.format(self.data))
                click.echo('\tExporting dataset files for {}...'.format(self.data))
                io.export_numpy('data/fashion_mnist/fashion_mnist_train_data', train_data)
                io.export_numpy('data/fashion_mnist/fashion_mnist_test_data', test_data)
                io.export_numpy('data/fashion_mnist/fashion_mnist_train_labels', train_labels)
                io.export_numpy('data/fashion_mnist/fashion_mnist_test_labels', test_values)
                click.echo('\t[DONE] Exporting dataset files for {}.'.format(self.data))
        click.echo('[DONE] Preparing {}.'.format(self.data))

    def feature_rep(self):
        pass

    def feature_extraction(self):
        pass

    def vbow(self):
        pass
