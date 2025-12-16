import math
import numpy as np


class Record:
    """record (scalar) performance"""

    def __init__(self):
        self._best = {}
        self.prefer = {}  # {0: small, 1: big}
        self.seq = {}

    def best(self, key=None):
        """if `key` given, return the best value of that key (in number)
        else, return best records of all key (in string)
        """
        if key is not None:
            assert key in self._best
            return self._best[key]
        else:
            s = ""
            for k in self._best:
                v = self._best[k]
                if v != math.inf and v != - math.inf:
                    s += "{}: {}\n".format(k, v)
            return s

    def new(self, key=None):
        """if `key` given, return the newest value of that key (in number)
        else, return newest records of all key (in string)
        """
        if key is not None:
            assert key in self.seq
            return self.seq[key][-1]
        else:
            s = ""
            for k in self.seq:
                if len(self.seq[k]) > 0:
                    s += "{}: {}\n".format(k, self.seq[k][-1])
            return s

    def add_big(self, *args):
        for k in args:
            assert k not in self._best
            self._best[k] = - math.inf
            self.prefer[k] = 1
            self.seq[k] = []

    def add_small(self, *args):
        for k in args:
            assert k not in self._best
            self._best[k] = math.inf
            self.prefer[k] = 0
            self.seq[k] = []

    def update(self, key, value):
        _cmp = min if (0 == self.prefer[key]) else max
        self._best[key] = _cmp(self._best[key], value)
        self.seq[key].append(value)

    def near_mean(self, key, window=7):
        """mean of the nearest several records
        to recognize plateau & stop training
        """
        assert (window > 0) and (key in self.seq)
        if 0 == len(self.seq[key]):
            # print("Record: near_mean: not records yet")
            return None
        _list = self.seq[key][-window:]
        return sum(_list) / len(_list)


class MeanValue:
    """ref: torchnet.meter.AverageValueMeter
    mean(X) = sum_i { X_i } / n
    var(X) = sum_i { [X_i - mean(X)]^2 } / (n - 1)
           = sum_i { X_i^2 - 2 * mean(X) * X_i + mean(X)^2 } / (n - 1)
    """

    def __init__(self):
        self.reset()

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if 0 == self.n:
            self.mean, self.std = math.nan, math.nan
        elif 1 == self.n:
            self.mean, self.std = self.sum, math.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = math.sqrt(self.m_s / (self.n - 1.0))

    def value(self, prec=None):
        if isinstance(prec, int):
            return round(self.mean, prec), round(self.std, prec)
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = math.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = math.nan


def calc_stat(lst, percentages=[], prec=None, scale=None):
    """list of statistics: median, mean, standard error, min, max, percentiles
    It can be useful when you want to know these statistics of a list and
    dump them in a json log/string.
    Input:
        lst: list of number
        percentages: List[float] = [], what percentiles (quantile) to cauculate
        prec: int|None = None, round to which decimal place if it is an int
        scale: int|float|None = None, scale the elements in `lst` if it is an int or float
            Use it when `lst` contains normalised number (i.e. in [0, 1]) and you want to
            present them in percentage (i.e. 0.xyz -> xy.z%)
    """
    if isinstance(scale, (int, float)):
        lst = list(map(lambda x: scale * x, lst))

    ret = {
        "min": float(np.min(lst)),
        "max": float(np.max(lst)),
        "mean": float(np.mean(lst)),
        "std": float(np.std(lst)),
        "median": float(np.median(lst))
    }
    if len(percentages) > 0:
        percentages = [max(1e-7, min(p, 100 - 1e-7)) for p in percentages]
        percentiles = np.percentile(lst, percentages)
        for ptage, ptile in zip(percentages, percentiles):
            ret["p_{}".format(ptage)] = float(ptile)

    if isinstance(prec, int):
        ret = {k: round(v, prec) for k, v in ret.items()}

    return ret


class OnlineStatEstim:
    """online estimation of min, max, mean, median, percentile"""

    def __init__(self, percentages=[]):
        """percentages: List[float] in (0, 100)"""
        for p in percentages:
            assert 0 < p < 100, "percentages should be within open interval (0, 100)"
        self.percentages = [max(1e-7, min(p, 100 - 1e-7)) for p in percentages]
        self.avg_std = MeanMeter()
        self.reset()

    def __call__(self, x):
        """x: List[int|float]"""
        if isinstance(x, str):
            x = float(x)
        if isinstance(x, (int, float)):
            x = [x]

        x = np.asarray(x, dtype=float).flatten()

        # min, max
        self.min = min(self.min, float(x.min()))
        self.max = max(self.max, float(x.max()))
        # mean & standard deviation
        self.avg_std.add(x.sum(), x.size)
        # rank-based statistics: quantile/percentile
        self.digest.batch_update(x)

    def reset(self):
        self.min = math.inf
        self.max = - math.inf
        self.avg_std.reset() # mean & standard deviation
        from tdigest import TDigest # for online percentile estimation
        self.digest = TDigest() # rank-based statistics: quantile/percentile

    def value(self, prec=None):
        mean, std = self.avg_std.value(prec)
        median = self.digest.percentile(50)
        ans = {
            "min": round(self.min, prec) if isinstance(prec, int) else self.min,
            "max": round(self.max, prec) if isinstance(prec, int) else self.max,
            "mean": mean,
            "std": std,
            "median": round(median, prec) if isinstance(prec, int) else median
        }
        if len(self.percentages) > 0:
            ans["percentile"] = {}
            for p in self.percentages:
                v = self.digest.percentile(p)
                if isinstance(prec, int):
                    v = round(v, prec)

                ans["percentile"][str(p)] = v

        return ans
