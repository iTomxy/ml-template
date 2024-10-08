
import datetime, time, timeit


def timestamp(fmt="%Y%m%d-%H%M%S"):
    """time-stamp string"""
    return time.strftime(fmt, time.gmtime())


def human_time(seconds, prec=0):
    """transfer seconds to human readable time string
    Input:
        - seconds: float, time in second
        - prec: int, decimal precision of second to show
    Output:
        - str
    """
    # prec = max(0, prec)
    # seconds = round(seconds, prec)
    # minutes = int(seconds // 60)
    # hours = minutes // 60
    # days = hours // 24

    # seconds %= 60
    # minutes %= 60
    # hours %= 24

    # str_list = []
    # if days > 0:
    #     str_list.append("{:d}d".format(days))
    # if hours > 0:
    #     str_list.append("{:d}h".format(hours))
    # if minutes > 0:
    #     str_list.append("{:d}m".format(minutes))
    # if seconds > 0:
    #     str_list.append("{0:.{1}f}s".format(seconds, prec))

    # return ' '.join(str_list) if len(str_list) > 0 else "0s"
    return str(datetime.timedelta(seconds=round(seconds, prec)))


class tic_toc:
    """timer with custom message"""

    def __init__(self, message="time used", end='\n'):
        self.msg = message
        self.end = end

    def __enter__(self):
        self.tic = timeit.default_timer()

    def __exit__(self, exc_type, exc_val, exc_tb):
        n_second = timeit.default_timer() - self.tic
        print("{}: {}".format(self.msg, datetime.timedelta(seconds=int(n_second))), end=self.end)

    def __call__(self, f):
        """supports decorator-style usage, e.g.:
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


class EMATimer:
    """Measure the time of a repeated procedule (e.g. a training epoch/iteration).
    Also record the starting time on the instance creation.
    """

    def __init__(self, momentum=0.01):
        """
        momentum: float, in [0, 1], how slow to update the estimated time.
        """
        self.momentum = max(0, min(momentum, 1))
        self.start_time = time.asctime()
        self.estim_time = -1

    def __enter__(self):
        self.tic = timeit.default_timer()

    def __exit__(self, exc_type, exc_val, exc_tb):
        n_seconds = timeit.default_timer() - self.tic
        if self.estim_time < 0:
            self.estim_time = n_seconds
        else:
            self.estim_time = self.momentum * self.estim_time + (1 - self.momentum) * n_seconds

    def __call__(self, f):
        """supports decorator-style usage, e.g.:
        ```python
        timer = EMATimer(0.5)
        @timer
        def train_one_epoch:
            pass
        ```
        https://stackoverflow.com/questions/9213600/function-acting-as-both-decorator-and-context-manager-in-python
        """
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            with self:
                return f(*args, **kwargs)
        return decorated


class RepeatingEventTimer:
    """time a repeating event at a specific timing (e.g. right after validation in deep learning)
    Example:
    ```python
        retimer = RepeatingEventTimer(skip_first=2)
        validate()      # initial validation before training, skipped
        for epoch in range(EPOCH):
            train()     # training
            validate()  # validation
            retimer()   # <- call here, times training + validation
    ```
    """

    def __init__(self, momentum=0.1, skip_first=1):
        """
        skip_first: int, skip the 1st (several) occurence/s.
            In typical cases, it is 1. Because the interval between two occurences
            marks one loop, we start timing on the 1st occurence.
            It is designed as an int in case one wants to skip the first several occurences.
        """
        self.skip_first = skip_first
        self.momentum = max(0, min(momentum, 1))
        self.tic = None
        self.estim_time = -1

    def __call__(self):
        if self.skip_first > 0:
            self.skip_first -= 1
        else:
            t = timeit.default_timer() - self.tic
            if self.estim_time < 0:
                self.estim_time = t
            else:
                self.estim_time = self.momentum * self.estim_time + (1 - self.momentum) * t

        self.tic = timeit.default_timer()

    def time(self):
        return max(0, self.estim_time)
