import os, os.path as osp, json, datetime, time, timeit, functools
from timing import EMATimer


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
