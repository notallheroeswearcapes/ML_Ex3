import click
from deeplearn import model, classifier, run_cnn


class Context:

    def __init__(self, data):
        self.data = data


@click.group(invoke_without_command=True)
@click.option("-d", "--data", type=str, required=True,
              help="Dataset to build the model on. Either \'CIFAR-10\' or \'FashionMNIST\'.")
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
@click.option("-a", "--algorithm", type=str, required=True,
              help="The classification algorithm to run on the model. Either \'k-NN\', \'MLP\' or "
                   "\'RandomForest\'.")
def classify(ctx, algorithm):
    """Runs a simple classification."""

    if algorithm != ("k-NN" or "MLP" or "RandomForest"):
        click.echo(
            "\nWrong input. Please specify the \'-a\' or \'--algorithm\' option as either \'k-NN\', \'MLP\' or "
            "\'RandomForest\'.")
        return

    clf = classifier.Classifier(algorithm, ctx.obj.data)
    clf.classify()


@cli.command()
@click.pass_context
@click.option("-a", "--architecture", type=str, required=True,
              help="The architecture of the neural network to run on the trained model.")
def cnn(ctx, architecture):
    """Runs a Convolutional Neural Network."""
    if architecture != ("Resnet-50"):
        click.echo(
            "\nWrong input. Please specify the \'-a\' or \'--algorithm\' option as either \'Resnet-50\'.")
        return
    arc = run_cnn.Cnn(architecture, ctx.obj.data)
    arc.run_classification()


@cli.command()
@click.pass_context
def evaluate():
    """Evaluates and compares classification results."""
    pass
