import numpy as np
import json
from pathlib import Path


def export_data(data, prefix, train_data=None, train_labels=None, test_data=None, test_labels=None):
    if train_data is not None:
        np.save('data/{}_{}_train_data'.format(data, prefix), train_data)
    if train_labels is not None:
        np.save('data/{}_{}_train_labels'.format(data, prefix), train_labels)
    if test_data is not None:
        np.save('data/{}_{}_test_data'.format(data, prefix), test_data)
    if test_labels is not None:
        np.save('data/{}_{}_test_labels'.format(data, prefix), test_labels)


def import_data(data, prefix):
    path = Path(__file__).parent.parent
    train_data = np.load(path / 'data/{}_{}_train_data.npy'.format(data, prefix))
    train_labels = np.load(path / 'data/{}_{}_train_labels.npy'.format(data, prefix))
    test_data = np.load(path / 'data/{}_{}_test_data.npy'.format(data, prefix))
    test_labels = np.load(path / 'data/{}_{}_test_labels.npy'.format(data, prefix))
    return (train_data, train_labels), (test_data, test_labels)


def export_result(results, data, clf):
    with open('data/{}_{}_results'.format(data, clf), 'w') as fp:
        json.dump(results, fp)
    fp.close()
