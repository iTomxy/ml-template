import json
import os
import os.path as osp
import time
import timeit

from .misc import timestamp, Logger


class BaseTrainer:
    def __init__(self,
                 time_estim_mometum=0.8,
                 log_dir='.',
                 log_file=None):
        self.log_freq = log_freq
        self.mometum = time_estim_mometum

        if log_file is None:
            log_file = "log.{}.json".format(timestamp())
        log_file = osp.join(log_dir, log_file)
        log_dir, log_file = osp.dirname(log_file), osp.basename(log_file)  # in case path in `log_file`
        os.makedirs(log_dir, exist_ok=True)
        self.logger = Logger(log_dir, log_file)

    def __del__(self):
        self.logger.flush()


class IterTrainer(BaseTrainer):
    def __init__(self,
                 n_iters,
                 log_freq=10,
                 time_estim_mometum=0.8,
                 log_dir='.',
                 log_file=None):
        super(time_estim_mometum, log_dir, log_file)
        self.n_iters = n_iters
        self.log_freq = log_freq
        self.iter = 0
        self.estim_time = {"iter": 0}

    def iter_begin(self):
        self.tic_iter = timeit.default_timer()

    def iter_end(self, log_dict=None):
        if (self.iter + 1) % self.log_freq == 0:
            elapsed_iter = timeit.default_timer() - self.tic_iter
            self.estim_time["iter"] = self.mometum * (elapsed_iter / self.log_freq) + (1 - self.mometum) * self.estim_time["iter"]
            expected_train_time = self.estim_time["iter"] * (self.n_iters - self.iter - 1)
            _log_dict = {
                "iter": self.iter + 1,
                "iter_time": int(self.estim_time["iter"]),
                "elapsed_since_last": elapsed_iter,
                "expected_train_time": int(expected_train_time),
            }
            if log_dict is not None:
                _log_dict.update(log_dict)
            self.logger(json.dumps(_log_dict))
        elif log_dict is not None:
            self.logger(json.dumps(log_dict))

        self.iter += 1


class EpochTrainer(BaseTrainer):
    def __init__(self,
                 n_epochs, n_iters_per_epoch,
                 log_freq=10,
                 time_estim_mometum=0.8,
                 log_dir='.',
                 log_file=None):
        super(time_estim_mometum, log_dir, log_file)
        self.n_epochs = n_epochs
        self.n_iters_per_epoch = n_iters_per_epoch
        self.log_freq = log_freq
        self.epoch = 0
        self.iter = 0
        self.estim_time = {"epoch": 0, "iter": 0}

    def epoch_begin(self):
        self.tic_epoch = timeit.default_timer()

    def iter_begin(self):
        self.tic_iter = timeit.default_timer()

    def iter_end(self, log_dict=None):
        if (self.iter + 1) % self.log_freq == 0:
            elapsed_iter = timeit.default_timer() - self.tic_iter
            self.estim_time["iter"] = self.mometum * (elapsed_iter / self.log_freq) + (1 - self.mometum) * self.estim_time["iter"]
            expected_epoch_time = self.estim_time["iter"] * (self.n_iters_per_epoch - self.iter - 1)
            expected_train_time = self.estim_time["iter"] * n_iters_per_epoch * (self.n_epochs - self.epoch - 1)
            _log_dict = {
                "epoch": self.epoch + 1, "iter": self.iter + 1,
                "iter_time": int(self.estim_time["iter"]),
                "elapsed_since_last": elapsed_iter,
                "expected_epoch_time": int(expected_epoch_time),
                "expected_train_time": int(expected_train_time),
            }
            if log_dict is not None:
                _log_dict.update(log_dict)
            self.logger(json.dumps(_log_dict))
        elif log_dict is not None:
            self.logger(json.dumps(log_dict))

        self.iter += 1

    def epoch_end(self, log_dict=None):
        elapsed_epoch = timeit.default_timer() - self.tic_epoch
        self.estim_time["epoch"] = self.mometum * elapsed_epoch + (1 - self.mometum) * self.estim_time["epoch"]
        expected_train_time = self.estim_time["epoch"] * (self.n_epochs - self.epoch - 1)
        _log_dict = {
            "epoch": self.epoch + 1,
            "epoch_time": int(self.estim_time["epoch"]),
            "expected_train_time": int(expected_train_time),
        }
        if log_dict is not None:
            _log_dict.update(log_dict)
        self.logger(json.dumps(_log_dict))

        self.iter = 0
        self.epoch += 1

