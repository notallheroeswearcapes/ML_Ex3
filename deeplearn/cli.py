import click
from os import path
from deeplearn import model, classifier, neuralnetwork, evaluator


class Context:

    def __init__(self, data):
        self.data = data


ACCEPTED_DATA = ['CIFAR-10', 'Fashion-MNIST']
ACCEPTED_CLASSIFIERS = ['k-NN', 'MLP', 'RandomForest']
ACCEPTED_CNN = ['Resnet-50', 'CNN']


@click.group(invoke_without_command=True)
@click.option("-d", "--data", type=str, required=True,
              help="Dataset to build the model on. Either \'CIFAR-10\' or \'Fashion-MNIST\'.")
@click.pass_context
def cli(ctx, data):
    """under construction """

    if data not in ACCEPTED_DATA:
        click.echo(
            "\nWrong input. Please specify the \'-d\' or \'--data\' option either as one of: {}".format(ACCEPTED_DATA))
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

    if algorithm not in ACCEPTED_CLASSIFIERS:
        click.echo(
            "\nWrong input. Please specify the \'-a\' or \'--algorithm\' option as one of: {}.".format(
                ACCEPTED_CLASSIFIERS))
        return

    clf = classifier.Classifier(algorithm, ctx.obj.data)
    clf.run_classification()


@cli.command()
@click.pass_context
@click.option("-a", "--architecture", type=str, required=True,
              help="The architecture of the neural network to run on the trained model.")
def cnn(ctx, architecture):
    """Runs a Convolutional Neural Network."""

    if architecture not in ACCEPTED_CNN:
        click.echo(
            "\nWrong input. Please specify the \'-a\' or \'--architecture\' option as one of: {}.".format(ACCEPTED_CNN))
        return

    arc = neuralnetwork.NeuralNetwork(architecture, ctx.obj.data)
    arc.run_classification()


@cli.command()
@click.pass_context
@click.option("-c", "--classifiers", type=str, multiple=True, required=True,
              help="The classifiers to evaluate. Multiple classifiers are allowed. Specify at least one.")
def evaluate(ctx, classifiers):
    """Evaluates and compares classification results."""

    for clf in classifiers:
        if clf not in ACCEPTED_CLASSIFIERS and clf not in ACCEPTED_CNN:
            click.echo(
                "\nWrong input. Please specify the \'-c\' or \'--classifiers\' option as {} or {}.".format(
                    ACCEPTED_CLASSIFIERS, ACCEPTED_CNN))
            return
        if clf in ACCEPTED_CLASSIFIERS:
            if not (path.exists('data/{}_{}_rep_results.json'.format(ctx.obj.data, clf)) and path.exists(
                    'data/{}_{}_vbow_results.json'.format(ctx.obj.data, clf))):
                click.echo('\nClassification results for {} with {} not available.'.format(ctx.obj.data, clf))
                if click.confirm('Do you want to run this classification?'):
                    clf = classifier.Classifier(clf, ctx.obj.data)
                    clf.run_classification()
                else:
                    click.echo('Stopping deeplearn.')
                    return
        if clf in ACCEPTED_CNN:
            if not path.exists('data/{}_{}_cnn_results.json'.format(ctx.obj.data, clf)):
                click.echo('\nCNN classification results for {} with {} not available.'.format(ctx.obj.data, clf))
                if click.confirm('Do you want to run this classification?'):
                    arc = neuralnetwork.NeuralNetwork(clf, ctx.obj.data)
                    arc.run_classification()
                else:
                    click.echo('Stopping deeplearn.')
                    return

    evaluator.evaluate(ctx.obj.data, classifiers)
