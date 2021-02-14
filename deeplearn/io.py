import numpy as np


def export_numpy(path, array):
    np.save(path, array)


def import_numpy(path):
    return np.load(path)
