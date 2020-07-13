from os.path import join
import numpy as np
# import h5py
import scipy.io as sio
# import cv2
from PIL import Image
import torch
from torchvision import transforms
from args import *


class Flickr:
    def __init__(self,):
        # self.MEAN_PIX = join(args.data_path, "avgpix.{}.npy".format(args.dataset))
        self.LABELS = join(args.data_path, "labels.mat")
        self.IMAGES = join(args.data_path, "images")#.alexnet.npy")
        self.TEXTS = join(args.data_path, "texts.mat")

        split_path = join("./split", args.dataset)
        CENTRES = join(split_path, "{}.{}.npy".format(args.dataset, args.bit))
        self.idx_test = np.load(join(split_path, "idx_test.npy"))
        self.idx_ret = np.load(join(split_path, "idx_ret.npy"))
        self.idx_labeled = np.load(join(split_path, "idx_labeled.npy"))
        self.idx_unlabeled = np.load(join(split_path, "idx_unlabeled.npy"))
        if args.tuning:
            self._tuning_mode()

        self.indices_set = {
            "test": self.idx_test,
            "ret": self.idx_ret,
            "labeled": self.idx_labeled,
            "unlabeled": self.idx_unlabeled
        }

        # self.labels = np.load(self.LABELS)
        self.labels = torch.from_numpy(sio.loadmat(self.LABELS)["LAll"]).float()
        self.centres = torch.from_numpy(np.load(CENTRES)).float()  # in {-1, 1}
        # self.images = torch.from_numpy(np.load(self.IMAGES))
        # self.texts = torch.from_numpy(sio.loadmat(self.TEXTS)["YAll"])
        # mean_pix = np.load(self.MEAN_PIX).astype(np.float32)
        # self.mean_pix = torch.from_numpy(np.expand_dims(mean_pix, 0))  # [1, 224, 224, 3]

    def _tuning_mode(self):
        """k-fold to tune"""
        assert args.i_fold < args.n_fold
        n_unlabeled = self.idx_unlabeled.shape[0] // 2
        self.idx_unlabeled = self.idx_unlabeled[-n_unlabeled:]
        n_labeled = self.idx_labeled.shape[0]
        fold_size = n_labeled // args.n_fold
        begin, end = args.i_fold * fold_size, (args.i_fold + 1) * fold_size
        self.idx_test = self.idx_labeled[begin: end]
        self.idx_labeled = np.concatenate(
            (self.idx_labeled[:begin], self.idx_labeled[end:]))
        self.idx_ret = np.concatenate([self.idx_labeled, self.idx_unlabeled])
        
    def count(self, which):
        return self.indices_set[which].shape[0]

    def _load_images(self, indeces, transform=None):
        image_batch = []
        for idx in indeces:
            img = np.load(join(self.IMAGES, "{}.npy".format(idx)))  # [1, 224, 224, 3]
            if transform is not None:
                img = transform(Image.fromarray(img[0])).unsqueeze(0)
            else:
                img = torch.from_numpy(np.transpose(img, (0, 3, 1, 2)))
            image_batch.append(img)

        image_batch = torch.cat(image_batch, 0).float()
        return image_batch #- self.mean_pix

    def _next_batch_idx(self, ptr, batch_sz, indices, shuffle=False):
        """return (new_pointer, batch_indices_array)"""
        n_total = indices.shape[0]
        ptr_new = ptr + batch_sz
        if ptr_new <= n_total:
            idx = indices[ptr: ptr_new]
            if ptr_new == n_total:
                ptr_new = 0
        else:
            ptr_new = batch_sz - (n_total - ptr)
            idx = np.concatenate((indices[ptr:], indices[:ptr_new]))
            if shuffle:
                np.random.shuffle(indices)
        return ptr_new, idx

    def loop_set(self, which, transform=None, batch_size=None, shuffle=False):
        """cycle through the dataset"""
        batch_size = batch_size or args.batch_size
        indices = self.indices_set[which]
        if shuffle:
            indices = indices.copy()
        ptr = 0
        while True:
            ptr, idx = self._next_batch_idx(ptr, batch_size, indices, shuffle)
            image = self._load_images(idx, transform)
            # image = self.images[idx]
            yield self.labels[idx], image

    def iter_set(self, which, transform=None, batch_size=None, shuffle=False):
        """traverse the dataset once"""
        batch_size = batch_size or args.batch_size
        indices = self.indices_set[which]
        if shuffle:
            indices = indices.copy()
            np.random.shuffle(indices)
        for i in range(0, indices.shape[0], batch_size):
            idx = indices[i: i + batch_size]
            image = self._load_images(idx, transform)
            # image = self.images[idx]
            yield self.labels[idx], image

    def load_set(self, which, transform=None, shuffle=False):
        """load the whole set"""
        idx = self.indices_set[which]
        if shuffle:
            idx = idx.copy()
            np.random.shuffle(idx)
        image = self._load_images(idx, transform)
        # image = self.images[idx]
        return self.labels[idx], image

    def load_label(self, which):
        idx = self.indices_set[which]
        return self.labels[idx]

    def load_centre(self):
        return self.centres


if __name__ == "__main__":
    pass
