import random
import numpy as np
import cv2


"""image transforms/augmentations
use them as you do with those torchvision.transforms,
and likewise, they process SINGLE image of shape [H, W, C].

input: numpy.ndarray (maybe reading by cv2)
output: numpy.ndarray

references:
- https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#Compose
- https://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html
- https://blog.csdn.net/thisiszdy/article/details/87028312
"""


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Resize:
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        if isinstance(size, int):
            self.size = [int(size), int(size)]
        else:
            self.size = size
        self.interpolation = interpolation

    def __call__(self, x):
        return cv2.resize(x, self.size, interpolation=self.interpolation)


class Crop:
    """image[x: x + w, y: y + h, :]"""

    def __init__(self, size, x, y):
        if isinstance(size, int):
            self.size = [int(size), int(size)]
        else:
            self.size = size
        self.x = x
        self.y = y

    def __call__(self, x):
        w, h = self.size
        return x[self.x: self.x + w, self.y: self.y + h, :]


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            x = cv2.flip(x, 1)
        return x


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            x = cv2.flip(x, 0)
        return x


class MultiScaleCrop:
    """something like random crop, but multi-scale
    NOTE: following `Resize` needed
    """

    def __init__(self, size, scales=[1.0, 0.875, 0.75, 0.66, 0.5],
                 max_distort=1, fix_crop=True, more_fix_crop=True):
        
        if isinstance(size, int):
            self.size = [int(size), int(size)]
        else:
            self.size = size
        self.scales = scales
        self.max_distort = max_distort
        # perform random crop
        # or select randomly from some deterministic cropping positions
        self.fix_crop = fix_crop
        # whether to generate more deterministic cropping positions to choose
        self.more_fix_crop = more_fix_crop

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        offsets = list()
        offsets.append((0, 0))  # upper left
        offsets.append((4 * w_step, 0))  # upper right
        offsets.append((0, 4 * h_step))  # lower left
        offsets.append((4 * w_step, 4 * h_step))  # lower right
        offsets.append((2 * w_step, 2 * h_step))  # center

        if self.more_fix_crop:
            offsets.append((0, 2 * h_step))  # center left
            offsets.append((4 * w_step, 2 * h_step))  # center right
            offsets.append((2 * w_step, 4 * h_step))  # lower center
            offsets.append((2 * w_step, 0 * h_step))  # upper center

            offsets.append((1 * w_step, 1 * h_step))  # upper left quarter
            offsets.append((3 * w_step, 1 * h_step))  # upper right quarter
            offsets.append((1 * w_step, 3 * h_step))  # lower left quarter
            offsets.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return random.choice(offsets)

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.size[1] if abs(
            x - self.size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.size[0] if abs(
            x - self.size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if self.fix_crop:
            w_offset, h_offset = self._sample_fix_offset(
                image_w, image_h, crop_pair[0], crop_pair[1])
        else:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def __call__(self, image):
        w, h, x, y = self._sample_crop_size(image.shape)
        return image[x: x + w, y: y + h, :]


class Normalize:
    """assert the input image is in range [0, 1]
    e.g., a `lambda x: x / 255` might has to be added before this
        if the original image is in range [0, 255].
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.expand_dims(np.asarray(mean, dtype=np.float32), [0, 1])
        self.std = np.expand_dims(std, dtype=np.float32), [0, 1])

    def __call__(self, x):
        return (x - self.mean) / self.std
