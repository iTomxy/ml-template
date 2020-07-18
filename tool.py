import os
import time
import math


def timestamp():
    """time-stamp string: Y-M-D-h-m"""
    t = time.localtime(time.time())
    return "{}-{}-{}-{}-{}".format(
        t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)


class Logger:
    """log info in stdout & log file"""
    def __init__(self, args):
        if not os.path.exists(args.log_path):
            os.makedirs(args.log_path)
        log_file_path = os.path.join(args.log_path, "log.{}".format(timestamp()))
        self.log_file = open(log_file_path, "a")
        for k, v in args._get_kwargs():
            self.log_file.write("{}: {}\n".format(k, v))
        self.log_file.write("begin time: {}\n".format(time.asctime()))

    def __del__(self):
        self.stop()

    def log(self, text):
        print(text)
        self.log_file.write(text + '\n')

    def stop(self):
        self.log_file.write("end time: {}\n".format(time.asctime()))
        self.log_file.flush()
        self.log_file.close()


class Record:
    """record (scalar) performance"""
    def __init__(self):
        self.best = {}
        self.prefer = {}  # {0: small, 1: big}
        self.seq = {}

    def __str__(self):
        return self.log_best()

    def log_best(self):
        s = ""
        for k in self.best:
            v = self.best[k]
            if v != math.inf and v != - math.inf:
                s += "{}: {}\n".format(k, v)
        return s

    def log_new(self):
        s = ""
        for k in self.seq:
            if len(self.seq[k]) > 0:
                s += 

    def add_big(self, *args):
        for k in args:
            assert k not in self.best
            self.best[k] = - math.inf
            self.prefer[k] = 1
            self.seq[k] = []

    def add_small(self, *args):
        for k in args:
            assert k not in self.best
            self.best[k] = math.inf
            self.prefer[k] = 0
            self.seq[k] = []

    def update(self, key, value):
        _cmp = min if (0 == self.prefer[key]) else max
        self.best[key] = _cmp(self.best[key], value)
        self.seq[key].append(value)
