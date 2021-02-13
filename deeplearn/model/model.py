import click


class Model:

    def __init__(self, data):
        self.data = data

    def test_prog_bar(self):
        with click.progressbar(range(100000),
                               label='Test progress bar',
                               length=100000) as bar:
            for x in bar:
                if x == 99999:
                    click.echo("finished")

    def load_data(self):
        pass

    def classify(self, classifier):
        click.echo(classifier)
