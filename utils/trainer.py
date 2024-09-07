import os, os.path as osp, json, datetime, time, timeit, functools


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


class BaseTrainer:
    def __init__(self):
        self.epoch_timer = EMATimer()
        self.iter_timer = EMATimer()

    def time_epoch(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            with self.epoch_timer:
                return func(self, *args, **kwargs)
        return wrapper

    def time_iter(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            with self.iter_timer:
                return func(self, *args, **kwargs)
        return wrapper

    def per_epoch(self):
        """#seconds cost per epoch"""
        return self.epoch_timer.estim_time

    def per_iter(self):
        """#seconds cost per iteration"""
        return self.iter_timer.estim_time


if "__main__" == __name__:
    print("usage 1")
    class Trainer(BaseTrainer):
        def __init__(self):
            super().__init__()
            self.epoch = 7

        @BaseTrainer.time_epoch
        def train_epoch(self, epoch):
            nipe = 2
            for i in range(nipe):
                self.train_iter(epoch, i)
                eta = datetime.timedelta(seconds=int(
                    (self.epoch - epoch) * self.per_epoch() + (nipe - i) * self.per_iter()
                ))
                print("ETA:", eta, ", expected finish time:", datetime.datetime.now() + eta)

            print("epoch", self.per_epoch())

        @BaseTrainer.time_iter
        def train_iter(self, epoch, it):
            time.sleep(1)
            print("iter", epoch, it, self.per_iter())

        def train(self):
            for epoch in range(self.epoch):
                self.train_epoch(epoch)

    Trainer().train()


    print("usage 2")
    timer = EMATimer()
    @timer
    def do(i):
        time.sleep(i)
        print(i)

    for i in range(7):
        do(i)
        print(i, timer.estim_time)
