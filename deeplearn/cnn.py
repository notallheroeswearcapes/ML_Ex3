from deeplearn import io
from tensorflow import keras
import numpy as np

import click

from deeplearn.io import load_model


class Cnn:
    def __init__(self, architecture, data):
        self.data = data
        self.architecture = architecture
        self.test_data = None
        self.test_labels = None

    def run_classification(self):
        click.echo("[START] Running classification on pretrained model...")

        _, (self.test_data, self.test_labels) = io.import_data(self.data, 'raw')
        reconstructed_model = load_model(self.architecture)
        test_data = []
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
        io.export_results(results_cnn, self.data, self.architecture, 'cnn')
