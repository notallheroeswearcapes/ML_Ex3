import click
from deeplearn.model import model


class Context:
    def __init__(self, data):
        self.data = data
        self.model = model.Model(data)


@click.group()
@click.pass_context
def cli(ctx):
    """INSTRUCTIONS"""
    pass


@cli.command()
@click.option("-d", "--data", type=str, required=True,
              help="Dataset to build the model on. Either CIFAR-10 or FashionMNIST.")
@click.pass_context
def train(ctx, data):
    if data != ("CIFAR-10" or "FashionMNIST"):
        click.echo(
            "Wrong input. Please specify the \'-d\' or \'--data\' option either as \'CIFAR-10\' or \'FashionMNIST\'.")
        return
    ctx.obj = Context(data)
    ctx.obj.model.test_prog_bar()


@cli.command()
@click.option("-a", "--algorithm", type=str, required=True, help="The classifier to run on the trained model.")
@click.pass_context
def classify(ctx, classifier):
    click.echo(classifier)


@cli.command()
@click.option("-a", "--architecture", type=str, required=True,
              help="The architecture of the neural network to run on the trained model.")
@click.pass_context
def cnn(ctx, architecture):
    click.echo(architecture)


@cli.command()
@click.pass_context
def evaluate(ctx):
    pass
