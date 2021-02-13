import click


class Model:

    def __init__(self, data):
        self.data = data

    def load_data(self):
        click.echo(self.data)

    def feature_rep(self):
        pass

    def feature_extraction(self):
        pass

    def vbow(self):
        pass
