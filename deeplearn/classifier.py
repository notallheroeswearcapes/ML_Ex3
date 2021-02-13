import click
from deeplearn import io
from sklearn import neighbors


class Classifier:

    KNN_PARAMS = {'n_neighbors': 3, 'weights': 'distance'}
    MLP_PARAMS = {}
    RANDOMFOREST_PARAMS = {}
    
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.train_data = None

    def classify(self):
        click.echo('Parameters for classifier \'{}\': {}'.format(self.algorithm, self.get_algorithm_params()))
        helper = io.IO()
        self.train_data = helper.read_json()
        if self.algorithm == "knn":
            self.knn_classify()
        elif self.algorithm == "mlp":
            self.mlp_classify()
        elif self.algorithm == "randomforest":
            self.randomforest_classify()
        click.echo('[DONE] classification with {}'.format(self.algorithm))

    def knn_classify(self):
        knn = neighbors.KNeighborsClassifier()

    def mlp_classify(self):
        pass

    def randomforest_classify(self):
        pass

    def get_algorithm_params(self):
        if self.algorithm == "knn":
            return self.KNN_PARAMS
        elif self.algorithm == "mlp":
            return self.MLP_PARAMS
        elif self.algorithm == "randomforest":
            return self.RANDOMFOREST_PARAMS
