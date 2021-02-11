"""
@author Matthias Eder
@since 11.02.2021
"""
import click


@click.command()
@click.option('--dataset', help='Number of greetings.')
def cli():
    print('Hello World!')
