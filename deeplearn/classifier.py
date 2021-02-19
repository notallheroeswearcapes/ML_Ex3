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

    def __init__(self, algorithm, data):
        self.algorithm = algorithm
        self.data = data
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        self.results_rep = None
        self.results_vbow = None

    def classify(self):
        click.echo('\nParameters for classifier {}: {}'.format(self.algorithm, self.get_algorithm_params()))

        clf = None
        if self.algorithm == "k-NN":
            clf = neighbors.KNeighborsClassifier(
                n_neighbors=self.KNN_PARAMS['n_neighbors'],
                weights=self.KNN_PARAMS['weights']
            )
        elif self.algorithm == "MLP":
            clf = neural_network.MLPClassifier(
                activation=self.MLP_PARAMS['activation'],
                solver=self.MLP_PARAMS['solver'],
                learning_rate_init=self.MLP_PARAMS['learning_rate_init'],
                max_iter=self.MLP_PARAMS['max_iter'],
                random_state=self.MLP_PARAMS['random_state']
            )
        elif self.algorithm == "RandomForest":
            clf = ensemble.RandomForestClassifier(
                n_estimators=self.RANDOMFOREST_PARAMS['n_estimators'],
                criterion=self.RANDOMFOREST_PARAMS['criterion'],
                random_state=self.RANDOMFOREST_PARAMS['random_state'],
                ccp_alpha=self.RANDOMFOREST_PARAMS['ccp_alpha']
            )

        click.echo('[START] Running classification on feature representation data...')
        (self.train_data, self.train_labels), (self.test_data, self.test_labels) = io.import_data(self.data, 'rep')
        click.echo('Shape of training data: {}'.format(self.train_data.shape))
        click.echo('Shape of training labels: {}'.format(self.train_labels.shape))
        click.echo('Shape of test data: {}'.format(self.test_data.shape))
        click.echo('Shape of test labels: {}'.format(self.test_labels.shape))
        st = default_timer()
        clf.fit(self.train_data, self.train_labels.ravel())
        prediction = clf.predict(self.test_data)
        score = metrics.accuracy_score(self.test_labels, prediction)
        runtime = default_timer() - st
        results_rep = {
            'data': self.data,
            'algorithm': self.algorithm,
            'input_data': 'rep',
            'runtime': runtime,
            'accuracy': score,
            'prediction': prediction.tolist()
        }
        io.export_result(results_rep, self.data, self.algorithm, 'rep')
        click.echo('[DONE] Classification on feature representation data.')

        click.echo('[START] Running classification on visual bag of words data...')
        (self.train_data, self.train_labels), (self.test_data, self.test_labels) = io.import_data(self.data, 'vbow')
        click.echo('Shape of training data: {}'.format(self.train_data.shape))
        click.echo('Shape of training labels: {}'.format(self.train_labels.shape))
        click.echo('Shape of test data: {}'.format(self.test_data.shape))
        click.echo('Shape of test labels: {}'.format(self.test_labels.shape))
        st = default_timer()
        clf.fit(self.train_data, self.train_labels)
        prediction = clf.predict(self.test_data)
        score = metrics.accuracy_score(self.test_labels, prediction)
        runtime = default_timer() - st
        results_vbow = {
            'data': self.data,
            'algorithm': self.algorithm,
            'input_data': 'vbow',
            'runtime': runtime,
            'accuracy': score,
            'prediction': prediction.tolist()
        }
        io.export_result(results_vbow, self.data, self.algorithm, 'vbow')
        click.echo('[DONE] Classification on visual bag of words data.')

        self.evaluate(results_rep, results_vbow)

        click.echo('\n[DONE] Classification with {}.'.format(self.algorithm))

    def evaluate(self, results_rep, results_vbow):
        click.echo('\nResults for classification of {} with {}:\n'.format(self.data, self.algorithm))
        table = PrettyTable(['', 'Runtime', 'Accuracy'])
        table.add_row(['feature representation',
                       '{}s'.format(round(results_rep['runtime'], 2)),
                       '{}'.format(results_rep['accuracy'])])
        table.add_row(['visual bag of words',
                       '{}s'.format(round(results_vbow['runtime'], 2)),
                       '{}'.format(results_vbow['accuracy'])])
        mean_acc = (results_rep['accuracy'] + results_vbow['accuracy']) / 2
        mean_runtime = (results_rep['runtime'] + results_vbow['runtime']) / 2
        table.add_row(['mean',
                       '{}s'.format(round(mean_runtime, 2)),
                       '{}'.format(mean_acc)])
        print(table)

    def get_algorithm_params(self):
        if self.algorithm == "k-NN":
            return self.KNN_PARAMS
        elif self.algorithm == "MLP":
            return self.MLP_PARAMS
        elif self.algorithm == "RandomForest":
            return self.RANDOMFOREST_PARAMS
