import numpy as np
import json


def export_data(data, train_data, train_labels, test_data, test_labels):
    np.save('data/{}_train_data'.format(data), train_data)
    np.save('data/{}_train_labels'.format(data), train_labels)
    np.save('data/{}_test_data'.format(data), test_data)
    np.save('data/{}_test_labels'.format(data), test_labels)


def export_bov(train_histograms, test_histograms, train_labels, test_labels):
    np.save('data/bov_train_data', train_histograms)
    np.save('data/bov_test_data', test_histograms)
    np.save('data/bov_train_data', train_labels)
    np.save('data/bov_test_data', test_labels)


def import_numpy(path):
    return np.load(path)


def export_classification(results, data, clf):
    with open('data/{}_{}_results'.format(data, clf), 'w') as fp:
        json.dump(results, fp)
    fp.close()
