# evaluate.py
import numpy as np
from scipy.optimize import linear_sum_assignment
import sklearn.metrics as metrics

"""
clustering evaluation metrics
"""

def calc_cost_matrix(y_true, y_assign, n_classes, n_clusters):
    """calculate cost matrix W
    Input:
        y_true: [n], in {0, ..., n_classes - 1}
        y_assign: [n], in {0, ..., n_clusters - 1}
        n_classes: int, provide in case that y_true.max() != n_classes
        n_clusters: int, provide in case that y_assign.max() != n_clusters
    Output:
        W: [n_clusters, n_classes]
    """
    y_true = y_true.astype(np.int64)
    y_assign = y_assign.astype(np.int64)
    assert y_assign.size == y_true.size # n
    # C = np.zeros((y_assign.max() + 1, y_true.max() + 1), dtype=np.int64)
    C = np.zeros((n_clusters, n_classes), dtype=np.int64)
    for i in range(y_assign.size):
        C[y_assign[i], y_true[i]] += 1
    W = C.max() - C
    return W


def reorder_assignment(y_true, y_assign, n_classes, n_clusters):
    """(linear_sum_assignment) re-order y_assign to be y_adjust so that it has the same order as y_true
    Input:
        y_true: [n], in {0, ..., c - 1}
        y_assign: [n], in {0, ..., d - 1}
        n_classes: int, provide in case that y_true.max() != n_classes
        n_clusters: int, provide in case that y_assign.max() != n_clusters
    Output:
        y_adjust: [n], in {0, ..., c - 1}, in same order as y_true
    """
    W = calc_cost_matrix(y_true, y_assign, n_classes, n_clusters)
    row_idx, col_idx = linear_sum_assignment(W)
    map_a2t = np.zeros(n_clusters, dtype=np.int64)
    for i, j in zip(row_idx, col_idx):
        map_a2t[i] = j
    y_adjust = map_a2t[y_assign]
    return y_adjust


def purity(y_true, y_assign):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_assign):
        hist, _ = np.histogram(y_true[y_assign == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_assign == cluster] = winner

    return metrics.accuracy_score(y_true, y_voted_labels)


def evaluate(y_true, y_assign, n_classes, n_clusters, average='macro'):
    y_adjust = reorder_assignment(y_true, y_assign, n_classes, n_clusters)
    return {
        # clustering
        'purity': purity(y_true, y_assign),
        'nmi': metrics.normalized_mutual_info_score(y_true, y_assign),
        'ami': metrics.adjusted_mutual_info_score(y_true, y_assign),
        'ari': metrics.adjusted_rand_score(y_true, y_assign),
        # classification
        'acc': metrics.accuracy_score(y_true, y_adjust),
        'precision': metrics.precision_score(y_true, y_adjust, average=average),
        'recall': metrics.recall_score(y_true, y_adjust, average=average),
        'f1-score': metrics.f1_score(y_true, y_adjust, average=average)
    }
