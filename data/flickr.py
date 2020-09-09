import numpy as np
import h5py
import scipy.io as sio
import cv2
from os.path import join
from args import args


class Flickr:
    def __init__(self):
        self.MEAN_PIX = join(
            args.data_path, "avgpix.{}.npy".format(args.dataset))
        self.LABELS = join(args.data_path, "labels.mat")
        self.IMAGES = join(args.data_path, "images")
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
            "train": self.idx_labeled,
            "ret": self.idx_ret,
            "labeled": self.idx_labeled,
            "unlabeled": self.idx_unlabeled
        }

        # self.labels = np.load(self.LABELS)
        self.labels = sio.loadmat(self.LABELS)["LAll"]
        self.centres = np.load(CENTRES).astype(np.float32)  # in {-1, 1}
        # self.images = np.load(self.IMAGES)
        # self.texts = sio.loadmat(self.TEXTS)["YAll"]
        # mean_pix = np.load(self.MEAN_PIX).astype(np.float32)
        # self.mean_pix = np.expand_dims(mean_pix, 0)  # [1, 224, 224, 3]
        self.mean = np.expand_dims(np.array([0.485, 0.456, 0.406]), (0, 1, 2))
        self.std = np.expand_dims(np.array([0.229, 0.224, 0.225]), (0, 1, 2))

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
            img = np.load(join(self.IMAGES, "{}.npy".format(idx)))  # [224, 224, 3]
            if transform is not None:
                img = transform(img)
            else:
                img = cv2.resize(img, args.image_size, interpolation=cv2.INTER_LINEAR)
            image_batch.append(img[np.newaxis, :])

        image_batch = np.concatenate(image_batch, 0).astype(np.float32)
        return image_batch

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
        if batch_size is None: batch_size = args.batch_size
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
        if batch_size is None: batch_size = args.batch_size
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


if __name__ == "__main__":
    dataset = Flickr()
    C = dataset.count("test")
    print("#test:", C)
