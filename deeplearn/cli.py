import click
from deeplearn import model, classifier


class Context:

    def __init__(self, data):
        self.data = data


@click.group(invoke_without_command=True)
@click.option("-d", "--data", type=str, required=True,
              help="Dataset to build the model on. Either CIFAR-10 or FashionMNIST.")
@click.pass_context
def cli(ctx, data):
    """under construction """

    if data != ("CIFAR-10" or "FashionMNIST"):
        click.echo(
            "\nWrong input. Please specify the \'-d\' or \'--data\' option either as \'CIFAR-10\' or \'FashionMNIST\'.")
        return

    ctx.obj = Context(data)

    click.echo('\n[START] Training the model for {}...'.format(data))
    trained_model = model.Model(data)
    trained_model.fetch_dataset()
    trained_model.feature_representation()
    trained_model.visual_bag_of_words()
    click.echo('[DONE] Trained the model for {}.'.format(data))


@cli.command()
@click.pass_context
@click.option("-a", "--algorithm", type=str, required=True, help="The classifier algorithm to run on the model.")
def classify(ctx, algorithm):
    """Runs a simple classification."""

    if algorithm != ("knn" or "mlp" or "randomforest"):
        click.echo(
            "\nWrong input. Please specify the \'-a\' or \'--algorithm\' option as either \'knn\', \'mlp\' or "
            "\'randomforest\'.")
        return

    clf = classifier.Classifier(algorithm, ctx.obj.data)
    clf.classify()


@cli.command()
@click.pass_context
@click.option("-a", "--architecture", type=str, required=True,
              help="The architecture of the neural network to run on the trained model.")
def cnn(architecture):
    """Runs a Convolutional Neural Network."""

    click.echo(architecture)


@cli.command()
@click.pass_context
def evaluate():
    """Evaluates and compares the classification results."""
    pass
