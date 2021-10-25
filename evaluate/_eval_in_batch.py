import sys
sys.path.append("..")
from util import MeanValue


def eval_in_batch(eval_fn, Dist, Sim, k=-1, batch_size=2000):
    assert Dist.shape[0] == Sim.shape[0]
    n = Dist.shape[0]
    single_k = isinstance(k, int) or (1 == len(k))
    if single_k:
        res = MeanValue()
    else:
        res = [MeanValue() for _ in range(len(k))]
    for i in range(0, n, batch_size):
        _D = Dist[i: i + batch_size]
        _S = Sim[i: i + batch_size]
        _bat_sz = int(_D.shape[0])
        _res = eval_fn(_D, _S, k)
        if single_k:
            res.add(_res * _bat_sz, _bat_sz)
        else:
            for _k in range(len(k)):
                res[_k].add(_res[_k] * _bat_sz, _bat_sz)

    if single_k:
        res = res.value()[0]
    else:
        for _k in range(len(k)):
            res[_k] = res[_k].value()[0]
    return res


if "__main__" == __name__:
    import numpy as np
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
    k_list = [1] + list(range(0, M + 1, 5)[1:])

    print("--- mAP")
    map1 = mAP(D_euc, Sim, k=k_list)
    map2 = eval_in_batch(mAP, D_euc, Sim, k_list, 20)
    print("map1:", map1)
    print("map2:", map2)

    print("--- mAP_tie")
    map_tie1 = mAP_tie(D_ham, Sim, k=k_list)
    map_tie2 = eval_in_batch(mAP_tie, D_ham, Sim, k_list, 20)
    print("map_tie1:", map_tie1)
    print("map_tie2:", map_tie2)

    print("--- nDCG")
    ndcg1 = nDCG(D_euc, Rel, k=k_list)
    ndcg2 = eval_in_batch(nDCG, D_euc, Rel, k_list, 3)
    print("ndcg1:", ndcg1)
    print("ndcg2:", ndcg2)

    print("--- nDCG_tie")
    ndcg_tie1 = nDCG_tie(D_ham, Rel, k=k_list)
    ndcg_tie2 = eval_in_batch(nDCG_tie, D_ham, Rel, k_list, 20)
    print("ndcg_tie1:", ndcg_tie1)
    print("ndcg_tie2:", ndcg_tie2)
