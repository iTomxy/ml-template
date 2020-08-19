import os
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from args import args


def cos(A, B=None):
    """cosine"""
    An = normalize(A, norm='l2', axis=1)
    if (B is None) or (B is A):
        return np.dot(An, An.T)
    Bn = normalize(B, norm='l2', axis=1)
    return np.dot(An, Bn.T)


def hamming(A, B=None, discrete=False):
    """A, B: [None, bit]
    elements in {-1, 1}
    """
    if B is None: B = A
    bit = A.shape[1]
    kernel = bit - A.dot(B.T)
    if discrete:
        return (kernel.astype(np.int) // 2).astype(A.dtype)
    else:
        return 0.5 * kernel


def euclidean(A, B=None, sqrt=False):
    aTb = np.dot(A, B.T)
    if (B is None) or (B is A):
        aTa = np.diag(aTb)
        bTb = aTa
    else:
        aTa = np.diag(np.dot(A, A.T))
        bTb = np.diag(np.dot(B, B.T))
    D = aTa[:, np.newaxis] - 2.0 * aTb + bTb[np.newaxis, :]
    if sqrt:
        D = np.sqrt(D)
    return D


def sim_mat(label, label_2=None):
    if label_2 is None:
        label_2 = label
    return (np.dot(label, label_2.T) > 0).astype(np.float32)


def calc_mAP(qF, rF, qL, rL, what=0, k=-1):
    """calculate mAP for retrieval
    Args:
        qF: query feature/hash matrix
        rF: retrieval feature/hash matrix
        qL: query label matrix
        rL: retrieval label matrix
        what: {0: cos, 1: hamming, 2: euclidean}
        k: mAP@k, default `-1` means mAP@ALL
    """
    n_query = qF.shape[0]
    if k == -1 or k > rF.shape[0]:
        k = rF.shape[0]
    Gnd = sim_mat(qL, rL).astype(np.int)
    if what == 0:
        Rank = np.argsort(1 - cos(qF, rF))
    elif what == 1:
        Rank = np.argsort(hamming(qF, rF))
    elif what == 2:
        Rank = np.argsort(euclidean(qF, rF))
        
    AP = 0.0
    for it in range(n_query):
        gnd = Gnd[it]
        if np.sum(gnd) == 0:
            continue
        rank = Rank[it][:k]
        gnd = gnd[rank]
        if np.sum(gnd) == 0:
            continue
        pos = np.asarray(np.where(gnd == 1.)) + 1.0
        rel_cnt = np.arange(pos.shape[-1]) + 1.0
        AP += np.mean(rel_cnt / pos)

    mAP = AP / n_query
    return mAP


def ap_pc(y_true, y_score):
    """AP per class for multi-label classification
    input:
    - y_true: [n, c], ground-truth, in {0, 1}
    - y_score: [n, c], predict score, in [0, 1]
    output:
    - AP: [c], AP[c] = average precision along the c-th class
    """
    n, c = y_true.shape
    # sort along sample, minus for DESC
    Rank = np.argsort(- y_score, axis=0)
    AP = np.zeros([c])
    for i in range(c):
        gnd = y_true[:, i]
        if gnd.sum() == 0:
            continue
        rank = Rank[:, i]
        gnd = gnd[rank]
        pos = np.asarray(np.where(gnd == 1.)) + 1.0
        rel_cnt = np.arange(pos.shape[-1]) + 1.0
        AP[i] = (rel_cnt / pos).mean()

    return AP


def prfa(y_true, y_pred):
    """Precision, Recall, F1, Accuracy
    - micro: OP, OR, OF1
    - macro: CP, CR, CF1
    input:
    - y_true: [n, c], ground-truth, in {0, 1}
    - y_pred: [n, c], prediction, in {0, 1}
    output:
    - OP, OR, OF1, CP, CR, CF1, acc
    """
    true = (y_true > 0.5).astype(np.int)
    pred = (y_pred > 0.5).astype(np.int)
    same = (true == pred).astype(np.int)
    n, c = y_true.shape
    gnd = true.sum(0)
    pos = pred.sum(0)
    tp = (true * pred).sum(0)

    OP = tp.sum() / max(pos.sum(), 1)
    OR = tp.sum() / max(gnd.sum(), 1)
    OF1 = (2 * OP * OR) / (OP + OR)

    pos[pos == 0] = 1
    tp[tp == 0] = 1
    CP = (tp / pos).sum() / c
    CR = (tp / gnd).sum() / c
    CF1 = (2 * CP * CR) / (CP + CR)

    acc_pc = same.sum(0) / n
    acc = acc_pc.sum() / c

    return OP, OR, OF1, CP, CR, CF1, acc, acc_pc


def analyse_outcast(y_true, y_score):
    """accuracy, including the outcast
    compare: prediction of outcast, true outcast
    y_true: [n * c, c + 1], in {0, 1}
    y_score: [n * c, c + 1], in [0, 1]
    """
    big = np.max(y_score, 1, keepdims=True)
    y_pred = np.where(y_score == big, 1, 0).astype(np.int)
    y_true = y_true.astype(np.int)
    acc_pc = (y_true == y_pred).sum(0) / y_true.shape[0]
    true_pc = y_true.sum(0)
    pred_pc = y_pred.sum(0)

    return acc_pc, true_pc, pred_pc


def t_sne(F, L, title="tsne"):
    """T-SNE visualization
    F: [n, d], features
    L: [n], label id
    """
    tsne = TSNE(n_components=2, init="pca", random_state=0)
    F = tsne.fit_transform(F)
    fig = plt.figure()
    plt.title(title)
    plt.scatter(F[:, 0], F[:, 1], s=25, c=L, marker='.', cmap="rainbow")
    plt.show()
    fig.savefig(os.path.join(args.log_path, "{}.png".format(title)))
    plt.close(fig)


def vis_retrieval(F, L, title="retrieval"):
    """T-SNE visualization
    F: [1 + n, d], features, with the query sample at first
    L: [1 + n, c], one-hot label id
    """
    tsne = TSNE(n_components=2, init="pca", random_state=0)
    F = tsne.fit_transform(F)
    fig = plt.figure()
    plt.title(title)
    S = sim_mat(L[:1], L)[0]  # [1 + n]
    plt.scatter(F[:1, 0], F[:1, 1], s=40, c=S[:1], marker='*', cmap="rainbow")
    plt.scatter(F[1:, 0], F[1:, 1], s=25, c=S[1:], marker='.', cmap="rainbow")
    plt.colorbar()
    plt.show()
    fig.savefig(os.path.join(args.log_path, "{}.png".format(title)))
    plt.close(fig)


if __name__ == "__main__":
    # qB = np.array([[1, -1, 1, 1],
    #            [-1, -1, -1, 1],
    #            [1, 1, -1, 1],
    #            [1, 1, 1, -1]])
    # rB = np.array([[1, -1, 1, -1],
    #                [-1, -1, 1, -1],
    #                [-1, -1, 1, -1],
    #                [1, 1, -1, -1],
    #                [-1, 1, -1, -1],
    #                [1, 1, -1, 1]])
    # query_L = np.array([[0, 1, 0, 0],
    #                     [1, 1, 0, 0],
    #                     [1, 0, 0, 1],
    #                     [0, 1, 0, 1]])
    # retrieval_L = np.array([[1, 0, 0, 1],
    #                         [1, 1, 0, 0],
    #                         [0, 1, 1, 0],
    #                         [0, 0, 1, 0],
    #                         [1, 0, 0, 0],
    #                         [0, 0, 1, 0]])
    # print("mAP test:", calc_mAP(qB, rB, query_L, retrieval_L, what=1))
    import flickr

    dataset = flickr.Flickr()
    test_labels = dataset.load_label("test")
    ret_labels = dataset.load_label("ret")

    mAP_cos = calc_mAP(test_labels, ret_labels, test_labels, ret_labels, 0)
    mAP_ham = calc_mAP(test_labels, ret_labels, test_labels, ret_labels, 1)
    mAP_euc = calc_mAP(test_labels, ret_labels, test_labels, ret_labels, 2)

    print("--- label ret ---")
    print("cos:", mAP_cos, ", ham:", mAP_ham, ", euc:", mAP_euc)
