try:
    import __builtin__ # Python 2
except ImportError:
    import builtins as __builtin__ # Python 3
from collections.abc import Iterable
import csv
import functools
import itertools
import logging
import math
import os
import re
import time
import timeit


def timestamp(fmt="%Y%m%d-%H%M%S"):
    """time-stamp string"""
    t = time.localtime(time.time())
    # return "{}-{}-{}-{}-{}".format(
    #     t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)
    return time.strftime(fmt, t)


def human_time(seconds, prec=1):
    """transfer seconds to human readable time string
    Input:
        - seconds: float, time in second
        - prec: int, decimal precision of second to show
    Output:
        - str
    """
    prec = max(0, prec)
    seconds = round(seconds, prec)
    minutes = int(seconds // 60)
    hours = minutes // 60
    days = hours // 24

    seconds %= 60
    minutes %= 60
    hours %= 24

    str_list = []
    if days > 0:
        str_list.append("{:d}d".format(days))
    if hours > 0:
        str_list.append("{:d}h".format(hours))
    if minutes > 0:
        str_list.append("{:d}m".format(minutes))
    if seconds > 0:
        str_list.append("{0:.{1}f}s".format(seconds, prec))

    return ' '.join(str_list) if len(str_list) > 0 else "0s"


class tic_toc:
    """timer with custom message"""

    def __init__(self, message="time used"):
        self.msg = message

    def __enter__(self):
        self.tic = timeit.default_timer()

    def __exit__(self, exc_type, exc_val, exc_tb):
        n_second = timeit.default_timer() - self.tic
        print("{}:".format(self.msg), human_time(n_second))

    def __call__(self, f):
        """enable to use as a context manager
        ```python
        @tic_toc("foo")
        def bar:
            pass
        ```
        https://stackoverflow.com/questions/9213600/function-acting-as-both-decorator-and-context-manager-in-python
        """
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            with self:
                return f(*args, **kwargs)
        return decorated


def enum_product(*args):
    """enumerate the product of several arrays with corresponding indices
    usage:
    - `for (i1, i2), (v1, v2) in enum_product(array1, array2):`
    ref:
    - https://stackoverflow.com/questions/56430745/enumerating-a-tuple-of-indices-with-itertools-product
    """
    if len(args) == 1:
        iterator = enumerate(args[0])
    else:
        iterator = zip(
            itertools.product(*(range(len(x)) for x in args)),
            itertools.product(*args)
        )

    for _index, _data in iterator:
        yield _index, _data


def prog_bar(iter_obj, prefix=None, timing=True):
    """simple progress bar
    Will temporarily overload the built-in `print` inside and restore when finish.
    Input:
        - iter_obj: iter_obj: range, tuple, list, numpy.ndarray
        - prefix: str, some message to show
        - timing: bool, to time each iteration and show it
    Ref:
        - https://stackoverflow.com/questions/3002085/how-to-print-out-status-bar-and-percentage
        - https://stackoverflow.com/questions/550470/overload-print-python
    """
    if isinstance(iter_obj, range):
        _current, _stop, _step = iter_obj.start, iter_obj.stop, iter_obj.step
    elif isinstance(iter_obj, Iterable):
        _current, _stop, _step = 0, len(iter_obj), 1
    else:
        raise NotImplemented

    # template: <msg>: [<current iter> / <total iter>, <ETA>]
    #   ETA: Expected Time of Arrival, i.e. expected time left
    template = ''
    if prefix != None:
        template += prefix + ": "
    if timing:
        template += "[ {:d} / {:d}, ETA: {} ]"
        estim_time = 0
        momentum = 0.9
    else:
        template += "[ {:d} / {:d} ]"

    # overload built-in `print` temporarily
    _builtin_print = __builtin__.print
    class _overload_print:
        def __init__(self):
            self.prev_log = ""
        def update_prog(self, new_log):
            _builtin_print("\r{}\r".format(' ' * len(self.prev_log)) + new_log, end="", flush=True)
            self.prev_log = new_log
        def __call__(self, *args, **kwargs):
            # clean the previous progress log first if the previous print is a progress log
            _builtin_print("\r{}\r".format(' ' * len(self.prev_log)), end="")
            _builtin_print(*args, **kwargs)
            _builtin_print(self.prev_log, end="", flush=True)

    _print_obj = _overload_print()
    global print
    __builtin__.print = print = _print_obj

    # iteration
    _print_obj.update_prog(template.format(_current, _stop, "N/A"))
    if timing:
        tic = timeit.default_timer()
        for i, x in enumerate(iter_obj):
            t = timeit.default_timer() - tic
            estim_time = momentum * t + (1 - momentum) * estim_time if i > 1 else t
            _current += _step
            eta = estim_time * (_stop - _current + 1)

            _print_obj.update_prog(template.format(_current, _stop, human_time(eta) if i > 0 else "N/A"))

            tic = timeit.default_timer()
            yield x

        _print_obj.update_prog(template.format(_stop, _stop, "0"))
    else:
        for x in iter_obj:
            _current += _step
            _print_obj.update_prog(template.format(_current, _stop))
            yield x

        _print_obj.update_prog(template.format(_stop, _stop))

    # restore the built-in `print`
    __builtin__.print = print = _builtin_print
    del _print_obj
    # finishing
    print('')


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

    def __call__(self, *text, sep=' ', end='\n', on_screen=True):
        """mimic built-in `print`"""
        if self.log_file is None:
            self.open()
        _str = sep.join(map(str, text))
        if on_screen:
            print(_str, end=end)
        self.log_file.write(_str + end)

    def open(self):
        if self.file_name is None:
            self.file_name = "log.{}.txt".format(timestamp())
        log_file_path = os.path.join(self.log_path, self.file_name)
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        self.log_file = open(log_file_path, "w")
        assert self.log_file is not None
        self.log_file.write("begin time: {}\n".format(time.asctime()))

    def flush(self):
        if self.log_file:
            self.log_file.flush()


def get_logger(
    logger_name,
    log_file = None,
    log_console = True,
    fmt = '[%(asctime)s] - {%(filename)s:%(lineno)d} - %(levelname)s - %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S',
    logger_level = logging.DEBUG,
    log_file_level = logging.INFO,
    log_console_level = logging.DEBUG
):
    """using built-in logging module
    https://blog.csdn.net/weixin_39278265/article/details/115203933
    Args:
        logger_name: str, globally unique logger name, usually `__file__`
        log_file: str, log to file if provided, default = None
        log_console: bool, whether to log to console, default = True
        fmt: str, logging message format
        datefmt: str, date format
        logger_level, log_file_level, log_console_level: can be logging.NOTSET|DEBUG|INFO|WARNING|ERROR|CRITICAL
    Return:
        logger: logging.Logger
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logger_level)
    formatter = logging.Formatter(fmt, datefmt=datefmt)
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        fileHandler = logging.FileHandler(log_file, mode='w')
        fileHandler.setLevel(log_file_level)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)
    if log_console:
        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(log_console_level)
        consoleHandler.setFormatter(formatter)
        logger.addHandler(consoleHandler)
    return logger


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


def natural_sort_key(s, num_pattern=re.compile('([0-9]+)'), lower=False):
    """https://stackoverflow.com/questions/4836710/is-there-a-built-in-function-for-string-natural-sort"""
    if lower:
        return [int(text) if text.isdigit() else text.lower()
                for text in num_pattern.split(s)]
    else:
        return [int(text) if text.isdigit() else text#.lower()
                for text in num_pattern.split(s)]


if __name__ == "__main__":
    data = {"a": (1, 2, 3), "b": [4, 5, 6]}
    dict2csv("test.csv", data)

    print(human_time(2 * 24 * 60 * 60 + 50))
    print(human_time(0.1415, 0))

    print("before prog_bar:", id(print))
    for x in prog_bar("abcdefg", "timing"):
        print('1', x, id(print))
        time.sleep(1)
    for x in prog_bar(range(5, 15, 2), "no timing", False):
        print(x, id(print))
        time.sleep(1)
    print("after prog_bar:", id(print))
