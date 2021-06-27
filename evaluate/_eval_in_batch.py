import sys
sys.path.append("..")
from tool import *


def eval_in_batch(eval_fn, Dist, Sim, k=-1, batch_size=2000):
    assert Dist.shape[0] == Sim.shape[0]
    n = Dist.shape[0]
    res = MeanValue()
    for i in range(0, n, batch_size):
        _D = Dist[i: i + batch_size]
        _S = Sim[i: i + batch_size]
        _bat_sz = int(_D.shape[0])
        _res = float(eval_fn(_D, _S, k))
        res.add(_res * _bat_sz, _bat_sz)

    return res.value()[0]
