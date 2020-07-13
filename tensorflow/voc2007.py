import numpy as np
import pickle
import cv2
from os.path import join
from args import args


class VOC2007:
    def __init__(self, zero_as=0):
        """zero_as: in {0, 1}
        - 0: treat difficult as negative
        - 1: treat difficult as positive
        """

        self.LABELS = join(args.data_path, "labels.l.npy")
        self.IMAGES = join(args.data_path, "images")#.resnet101.npy")

        split_path = join("./split", args.dataset)
        self.idx_test = np.load(join(split_path, "idx_test.npy"))
        self.idx_train = np.load(join(split_path, "idx_train_val.npy"))
        if args.tuning:
            self._tuning_mode()

        self.indices_set = {
            "test": self.idx_test,
            "train": self.idx_train
        }

        self.labels = np.load(self.LABELS).astype(np.float32)
        if 1 == zero_as:
            self.labels[0 == self.labels] = 1
        self.labels[-1 == self.labels] = 0
        self.images = np.load(self.IMAGES).astype(np.float32)

    def _tuning_mode(self):
        """k-fold to tune"""
        assert args.i_fold < args.n_fold
        n_train = self.idx_train.shape[0]
        fold_size = n_train // args.n_fold
        begin, end = args.i_fold * fold_size, (args.i_fold + 1) * fold_size
        self.idx_test = self.idx_train[begin: end]
        self.idx_train = np.concatenate(
            (self.idx_train[:begin], self.idx_train[end:]))

    def _augment(self, image):
        """single image: [w, h, c] or [w, h]
        I -> resize ->
        [256, 256] -> random crop ->
        [224, 224] -> random flip -> I'
        ref:
        - https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html?highlight=flip#cv2.flip
        """
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)  # bilinear
        x = np.random.randint(0, 33)
        y = np.random.randint(0, 33)
        image = image[x: x + 224, y: y + 224]
        if np.random.randint(2):
            image = cv2.flip(img, 1)
        return image

    def count(self, which):
        return self.indices_set[which].shape[0]

    def _load_images(self, indeces, augment=False):
        image_batch = []
        for idx in indeces:
            # idx: 0-base, image file name: 1-base
            img = cv2.imread(join(self.IMAGES, "{:0>6}.jpg".format(idx + 1)))[:, :, ::-1]
            if augment:
                img = self._augment(img)
            else:
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
            image_batch.append(img[np.newaxis, :])

        image_batch = np.vstack(image_batch).astype(np.float32)
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

    def loop_set(self, which, batch_size=None, shuffle=False, augment=False):
        """cycle through the dataset"""
        if batch_size is None: batch_size = args.batch_size
        indices = self.indices_set[which]
        if shuffle:
            indices = indices.copy()
        ptr = 0
        while True:
            ptr, idx = self._next_batch_idx(ptr, batch_size, indices, shuffle)
            image = self._load_images(idx, augment=augment)
            yield self.labels[idx], image#self.images[idx]

    def iter_set(self, which, batch_size=None, shuffle=False, augment=False):
        """traverse the dataset once"""
        if batch_size is None: batch_size = args.batch_size
        indices = self.indices_set[which]
        if shuffle:
            indices = indices.copy()
            np.random.shuffle(indices)
        for i in range(0, indices.shape[0], batch_size):
            idx = indices[i: i + batch_size]
            image = self._load_images(idx, augment=augment)
            yield self.labels[idx], image#self.images[idx]

    def load_set(self, which, shuffle=False, augment=False):
        """load the whole set"""
        idx = self.indices_set[which]
        if shuffle:
            idx = idx.copy()
            np.random.shuffle(idx)
        image = self._load_images(idx, augment=augment)
        yield self.labels[idx], image#self.images[idx]

    def load_label(self, which):
        idx = self.indices_set[which]
        return self.labels[idx]

    def load_label_w2v(self):
        with open(args.w2v_file, "rb") as f:
            data = pickle.load(f)
        return data


if __name__ == "__main__":
    dataset = VOC2007()
    print("train:", dataset.idx_train.max(), dataset.idx_train.min())
    print("test:", dataset.idx_test.max(), dataset.idx_test.min())

    import matplotlib.pyplot as plt
    from models import to_matrix, to_vector
    L = dataset.load_label("train")
    one_pc = (L == 1).sum(0)
    zero_pc = (L == 0).sum(0)
    zo_pc = zero_pc / one_pc
    print("one:", one_pc)
    print("zero:", zero_pc)
    print("0 / 1:", zo_pc)

    def show(label, title):
        cnt = np.sum(label, axis=0)
        fig = plt.figure()
        plt.bar(np.arange(cnt.shape[0]), cnt)
        plt.title(title)
        # plt.show()
        fig.savefig("log/{}.png".format(title))

    # show(L, "one")
    # show(1 - L, "zero")

