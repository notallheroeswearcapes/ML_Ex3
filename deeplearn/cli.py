import click
from deeplearn import model, classifier


@click.group()
def cli():
    """INSTRUCTIONS"""


@cli.command()
@click.option("-d", "--data", type=str, required=True,
              help="Dataset to build the model on. Either CIFAR-10 or FashionMNIST.")
def train(data):
    """Loads the specified dataset and prepares it for classification."""

    if data != ("CIFAR-10" or "FashionMNIST"):
        click.echo(
            "Wrong input. Please specify the \'-d\' or \'--data\' option either as \'CIFAR-10\' or \'FashionMNIST\'.")
        return
    x = model.Model(data)
    x.load_data()
    x.feature_rep()
    x.feature_extraction()
    x.vbow()


@cli.command()
@click.option("-a", "--algorithm", type=str, required=True, help="The classifier algorithm to run on the model.")
def classify(algorithm):
    """Runs a simple classification."""

    if algorithm != ("knn" or "mlp" or "randomforest"):
        click.echo(
            "Wrong input. Please specify the \'-a\' or \'--algorithm\' option as either \'knn\', \'mlp\' or "
            "\'randomforest\'.")
        return
    clf = classifier.Classifier(algorithm)
    clf.classify()


@cli.command()
@click.option("-a", "--architecture", type=str, required=True,
              help="The architecture of the neural network to run on the trained model.")
def cnn(architecture):
    """Runs a Convolutional Neural Network."""

    click.echo(architecture)


@cli.command()
def evaluate():
    """Evaluates and compares the classification results."""
    pass
