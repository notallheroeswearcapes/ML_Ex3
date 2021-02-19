import click
from sklearn import neighbors, metrics, neural_network, ensemble
from deeplearn import io, evaluator
from timeit import default_timer


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

    def run_classification(self):
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
        results_rep = self.classify(clf, 'rep')
        evaluator.create_confusion_matrix(results_rep['prediction'], self.test_labels, self.data, self.algorithm,
                                          input_data='Feature representation')
        click.echo('[DONE] Classification on feature representation data.')

        click.echo('[START] Running classification on visual bag of words data...')
        (self.train_data, self.train_labels), (self.test_data, self.test_labels) = io.import_data(self.data, 'vbow')
        results_vbow = self.classify(clf, 'vbow')
        evaluator.create_confusion_matrix(results_vbow['prediction'], self.test_labels, self.data, self.algorithm,
                                          input_data='Visual Bag of Words')
        click.echo('[DONE] Classification on visual bag of words data.')

        evaluator.evaluate_classification(results_rep, results_vbow)
        click.echo('\n[DONE] Classification with {}.'.format(self.algorithm))

    def classify(self, clf, input_data):
        st = default_timer()
        clf.fit(self.train_data, self.train_labels.ravel())
        prediction = clf.predict(self.test_data)
        score = metrics.accuracy_score(self.test_labels, prediction)
        runtime = default_timer() - st
        results = {
            'data': self.data,
            'algorithm': self.algorithm,
            'input_data': input_data,
            'runtime': runtime,
            'accuracy': score,
            'prediction': prediction.tolist()
        }
        io.export_results(results, self.data, self.algorithm, input_data)
        return results

    def get_algorithm_params(self):
        if self.algorithm == "k-NN":
            return self.KNN_PARAMS
        elif self.algorithm == "MLP":
            return self.MLP_PARAMS
        elif self.algorithm == "RandomForest":
            return self.RANDOMFOREST_PARAMS
