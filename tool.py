import os
import time
import math
import csv
import itertools


def timestamp():
    """time-stamp string: Y-M-D-h-m"""
    t = time.localtime(time.time())
    # return time.strftime("%Y-%m-%d-%H-%M", t)
    return "{}-{}-{}-{}-{}".format(
        t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)


def enum_product(*args):
    """enumerate the product of several arrays with corresponding indices
    usage:
    - `for (i1, i2), (v1, v2) in enum_product(array1, array2):`
    ref:
    - https://stackoverflow.com/questions/56430745/enumerating-a-tuple-of-indices-with-itertools-product
    """
    for _index, _data in zip(
        itertools.product(*(range(len(x)) for x in args)),
        itertools.product(*args)
    ):
        yield _index, _data


class Logger:
    """log info in stdout & log file"""

    def __init__(self, log_path='.', file_name=None):
        self.log_path = log_path
        self.file_name = file_name
        self.log_file = None

    def __del__(self):
        if self.log_file is not None:
            self.log_file.write("end time: {}\n".format(time.asctime()))
            self.log_file.flush()
            self.log_file.close()
            # self.log_file = None

    def __call__(self, text, on_screen=True):
        if self.log_file is None:
            self.open()
        if on_screen:
            print(text)
        self.log_file.write(text + '\n')

    def open(self):
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if self.file_name is None:
            self.file_name = "log.{}.txt".format(timestamp())
        log_file_path = os.path.join(self.log_path, self.file_name)
        self.log_file = open(log_file_path, "w")
        assert self.log_file is not None
        self.log_file.write("begin time: {}\n".format(time.asctime()))

    def flush(self):
        if self.log_file:
            self.log_file.flush()


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

    def value(self):
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


def dict2csv(csv_file, dict_data):
    """write a dict to `csv_file`
    - dict_data: {(str) key name: (list) value list}
    """
    with open(csv_file, "w", newline='') as f:
        writer = csv.writer(f)
        for k in dict_data:
            v_list = [k]
            v_list.extend(list(dict_data[k]))
            writer.writerow(v_list)


if __name__ == "__main__":
    data = {"a": (1, 2, 3), "b": [4, 5, 6]}
    dict2csv("test.csv", data)
