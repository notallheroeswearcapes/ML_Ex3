import numpy as np
import json


def export_data(data, prefix, train_data, train_labels, test_data, test_labels):
    np.save('data/{}_{}_train_data'.format(data, prefix), train_data)
    np.save('data/{}_{}_train_labels'.format(data, prefix), train_labels)
    np.save('data/{}_{}_test_data'.format(data, prefix), test_data)
    np.save('data/{}_{}_test_labels'.format(data, prefix), test_labels)


def import_data(data, prefix):
    train_data = np.load('data/{}_{}_train_data'.format(data, prefix))
    train_labels = np.load('data/{}_{}_train_labels'.format(data, prefix))
    test_data = np.load('data/{}_{}_test_data'.format(data, prefix))
    test_labels = np.load('data/{}_{}_test_labels'.format(data, prefix))
    return (train_data, train_labels), (test_data, test_labels)


def export_result(results, data, clf):
    with open('data/{}_{}_results'.format(data, clf), 'w') as fp:
        json.dump(results, fp)
    fp.close()
