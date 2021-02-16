import click
from sklearn import neighbors, metrics, neural_network, ensemble
from deeplearn import io
from timeit import default_timer
from prettytable import PrettyTable


class Classifier:

    KNN_PARAMS = {
        'n_neighbors': 3,
        'weights': 'distance'
    }
    MLP_PARAMS = {
        'activation': 'relu',
        'solver': 'adam',
        'learning_rate_init': 0.001,
        'max_iter': 100,
        'random_state': 1
    }
    RANDOMFOREST_PARAMS = {
        'n_estimators': 100,
        'criterion': 'gini',
        'random_state': 1,
        'ccp_alpha': 0.0
    }

    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        self.results = None

    def classify(self):
        click.echo('Parameters for classifier \'{}\': {}'.format(self.algorithm, self.get_algorithm_params()))
        self.load_data()

        click.echo('Starting classification...')
        st = default_timer()
        clf = None
        if self.algorithm == "knn":
            clf = neighbors.KNeighborsClassifier(
                n_neighbors=self.KNN_PARAMS['n_neighbors'],
                weights=self.KNN_PARAMS['weights']
            )
        elif self.algorithm == "mlp":
            clf = neural_network.MLPClassifier(
                activation=self.MLP_PARAMS['activation'],
                solver=self.MLP_PARAMS['solver'],
                learning_rate_init=self.MLP_PARAMS['learning_rate_init'],
                max_iter=self.MLP_PARAMS['max_iter'],
                random_state=self.MLP_PARAMS['random_state']
            )
        elif self.algorithm == "randomforest":
            clf = ensemble.RandomForestClassifier(
                n_estimators=self.RANDOMFOREST_PARAMS['n_estimators'],
                criterion=self.RANDOMFOREST_PARAMS['criterion'],
                random_state=self.RANDOMFOREST_PARAMS['random_state'],
                ccp_alpha=self.RANDOMFOREST_PARAMS['ccp_alpha']
            )

        clf.fit(self.train_data, self.train_labels)
        prediction = clf.predict(self.test_data)
        score = metrics.accuracy_score(self.test_labels, prediction)

        runtime = default_timer() - st
        self.results = {
            'algorithm': self.algorithm,
            'runtime': runtime,
            'accuracy': score,
            'prediction': prediction
        }
        io.export_result(self.results, 'cifar10', self.algorithm)

        click.echo('[DONE] Classification with {}.'.format(self.algorithm))

    def load_data(self):
        click.echo('Reading data...')
        self.train_data = io.import_data('data/cifar10_train_data.npy')
        self.train_labels = io.import_data('data/cifar10_train_labels.npy')
        self.test_data = io.import_data('data/cifar10_test_data.npy')
        self.test_labels = io.import_data('data/cifar10_test_labels.npy')
        click.echo('[DONE] Reading data.')

    def evaluate(self):
        if self.results is None:
            click.echo('Error: No results available.')
            return
        table = PrettyTable()
        table.field_names = (['Runtime', 'Accuracy'])
        table.add_row(['{}s'.format(round(self.results['runtime'], 2)), '{}'.format(self.results['accuracy'])])
        click.echo(table)

    def get_algorithm_params(self):
        if self.algorithm == "knn":
            return self.KNN_PARAMS
        elif self.algorithm == "mlp":
            return self.MLP_PARAMS
        elif self.algorithm == "randomforest":
            return self.RANDOMFOREST_PARAMS
