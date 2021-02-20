from sklearn.metrics import accuracy_score

from deeplearn import io
from tensorflow import keras
import numpy as np
import click
from deeplearn import io, evaluator


class NeuralNetwork:

    def __init__(self, architecture, data):
        self.data = data
        self.architecture = architecture
        self.test_data = None
        self.test_labels = None

    def run_classification(self):
        click.echo("[START] Running classification on pre-trained model...")

        _, (self.test_data, self.test_labels) = io.import_data(self.data, 'raw')
        reconstructed_model = io.load_model(self.architecture, self.data)
        test_data = []
        if self.architecture == "Resnet-50":
            test_data = keras.layers.UpSampling2D(size=(4, 4))(self.test_data)
        if self.architecture == "CNN":
            test_data = self.test_data.astype('float32') / 255

        predictions = reconstructed_model.predict(test_data)
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(predictions, self.test_labels)
        results_cnn = {
            'data': self.data,
            'algorithm': self.architecture,
            'input_data': 'cnn',
            'accuracy': accuracy,
            'prediction': predictions.tolist()
        }
        io.export_results(results_cnn, self.data, self.architecture, 'cnn')
        evaluator.create_confusion_matrix(predictions, self.test_labels, self.data, self.architecture)
        #evaluator.evaluate_cnn(results_cnn) will be usable once accuracy and runtime are provided in the results
        click.echo('[DONE] Classification with {}.'.format(self.architecture))
