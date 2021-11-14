import sys
sys.path.append("..")
from util import MeanValue
import numpy as np


def eval_in_batch(X, L, dist_fn, eval_fn, sim_fn, k=-1, batch_size=2000):
    """evaluate multiple metrics with multiple k in batch
    Input:
        X: [n, d] feature matrix or 2-element tuple/list ([n, d], [m, d])
        L: [n, c] label matrix or 2-element tuple/list ([n, c], [m, c])
            If using sparse labels, the shapes are [n] or ([n], [m]).
            Notes that this function is oblivious of the sparsity of the labels,
            and such consideration is on those `sim_fn`s.
        dist_fn: distance function
            Called by `dist_fn(X1, X2)`.
        eval_fn: (list of) evaluation function/s
            Called by `eval_fn(DistMat, SimMat, k/[k1, k2, ...])`.
        sim_fn: (list of) similarity function/s
            Called by `sim_fn(L1, L2)`.
        k: (list of) int for position threshold/s
        batch_size: int
    Output:
        result: [#eval, #k], with result[i][j] = evaluation value of metirc_i@k_j
    """
    # preprocessings
    if not isinstance(eval_fn, (list, tuple)):
        eval_fn = [eval_fn]
    if not isinstance(sim_fn, (list, tuple)):
        sim_fn = [sim_fn]
    assert len(eval_fn) == len(sim_fn)
    if isinstance(k, int):
        k = [k]
    if isinstance(X, (list, tuple)):
        assert len(X) == 2
        X, X2 = X
    else:
        assert isinstance(X, np.ndarray)
        X2 = X
    if isinstance(L, (list, tuple)):
        assert len(L) == 2
        L, L2 = L
    else:
        assert isinstance(L, np.ndarray)
        L2 = L
    assert (X.shape[0] == L.shape[0]) and (X.shape[1] == X2.shape[1])
    if 2 == L.ndim:
        assert L.shape[1] == L2.shape[1]

    n, m = X.shape[0], X2.shape[0]
    n_metr, n_k = len(eval_fn), len(k)
    result = []
    for _1 in range(n_metr):
        _res = []
        for _2 in range(n_k):
            _res.append(MeanValue())
        result.append(_res)

    for i in range(0, n, batch_size):
        _X1 = X[i: i + batch_size]
        _L1 = L[i: i + batch_size]
        _bat_sz = int(_L1.shape[0])
        _D = dist_fn(_X1, X2)
        for i_metr, (_eval_fn, _sim_fn) in enumerate(zip(eval_fn, sim_fn)):
            _S = _sim_fn(_L1, L2)
            _res = _eval_fn(_D, _S, k)
            # assert isinstance(_res, (np.ndarray, np.float))
            if 0 == _res.ndim:  # single k
                result[i_metr][0].add(_res * _bat_sz, _bat_sz)
            else:  # multiple k
                for _k in range(len(k)):
                    result[i_metr][_k].add(_res[_k] * _bat_sz, _bat_sz)

    # reduce
    for i_metr in range(n_metr):
        for i_k in range(n_k):
            result[i_metr][i_k] = result[i_metr][i_k].value()[0]
    if 1 == n_k:
        for i_metr in range(n_metr):
            # assert len(result[i_metr]) == 1
            result[i_metr] = result[i_metr][0]
    # if 1 == n_metr:
    #     # assert len(result) == 1
    #     result = result[0]
    return result


if "__main__" == __name__:
    from _mAP import *
    from _nDCG import *
    from util.numpy import *
    import pprint

    N, M = 11, 23
    qF = np.random.randn(N, 3)
    rF = np.random.randn(M, 3)
    qB = np.random.randint(0, 2, size=(N, 3)) * 2 - 1
    rB = np.random.randint(0, 2, size=(M, 3)) * 2 - 1
    qL = np.random.randint(0, 2, size=(N, 7))
    rL = np.random.randint(0, 2, size=(M, 7))
    # sio.savemat("data/data.mat", {
        # "qB": qB, "rB": rB, "qL": qL, "rL": rL
    # })
    D_euc = euclidean(qF, rF)
    D_ham = hamming(qB, rB)
    Rel = qL.dot(rL.T)
    Sim = (Rel > 0).astype(np.int32)
    k_list = [1] #+ list(range(0, M + 1, 5)[1:])

    print("--- mAP")
    map1 = [mAP(D_euc, Sim, k) for k in k_list]
    print("map1:", map1)

    print("--- nDCG")
    ndcg1 = [nDCG(D_euc, Rel, k) for k in k_list]
    print("ndcg1:", ndcg1)

    sim_fn = lambda Lx, Ly: (Lx.dot(Ly.T) > 0).astype(np.int32)
    rel_fn = lambda Lx, Ly: Lx.dot(Ly.T)
    print("--- eval_in_batch")
    result = eval_in_batch(
        [qF, rF], [qL, rL], euclidean,
        [multi_mAP],# multi_nDCG],
        [sim_fn],# rel_fn],
        k_list, batch_size=7)
    # for _metr, _res in zip(["mAP", "nDCG"], result):
    #     print(_metr, ':', _res)
    for _metr, _res in zip(["mAP"], result):
        print(_metr, ':', _res)
