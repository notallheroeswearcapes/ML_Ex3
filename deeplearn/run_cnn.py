from sklearn.metrics import accuracy_score

from deeplearn import io
from tensorflow import keras
import numpy as np
from pathlib import Path

import click


class Cnn:
    def __init__(self, architecture, data):
        self.data = data
        self.architecture = architecture
        self.test_data = None
        self.test_labels = None

    def run_classification(self):
        click.echo("Running classification on pretrained model...")

        self.load_data()
        reconstructed_model = self.load_model()
        if self.architecture == "Resnet-50":
            if self.data == "CIFAR-10":
                test_data = keras.layers.UpSampling2D(size=(4, 4))(self.test_data)

        predictions = reconstructed_model.predict(test_data)
        predictions = np.argmax(predictions, axis=1)
        click.echo('[DONE] Classification with {}.'.format(self.architecture))
        results_cnn = {
            'data': self.data,
            'algorithm': self.architecture,
            'input_data': 'cnn',
             #'accuracy': score,
            'prediction': predictions.tolist()
        }
        io.export_result(results_cnn, self.data, self.architecture, 'cnn')

    def load_data(self):
        click.echo('Reading data...')

        _, (self.test_data, self.test_labels) = io.import_data('CIFAR-10', 'raw')
        click.echo('[DONE] Reading data.')

    def load_model(self):
        path = Path(__file__).parent.parent
        reconstructed_model = keras.models.load_model(path/"trained_models/{}".format(self.architecture), compile=True)
        return reconstructed_model
