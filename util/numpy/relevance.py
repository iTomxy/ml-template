import numpy as np


def sim_mat(label, label_2=None, sparse=False):
    if label_2 is None:
        label_2 = label
    if sparse:
        S = (label[:, np.newaxis] == label_2[np.newaxis, :])
    else:
        S = np.dot(label, label_2.T) > 0
    return S.astype(label.dtype)
