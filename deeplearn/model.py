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
        pass
