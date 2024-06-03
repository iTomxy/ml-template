import os, os.path as osp
import cv2
import numpy as np


def crop(img, blank=(255, 255, 255)):
    """remove blank edge of an image
    Input:
        img: (H, W[, C]), numpy.ndarray
        blank: int or (int, int, int), pixel value of blank edge to cut
    """
    pos = np.where(img != blank)
    xs, ys = pos[:2]
    l, r = xs.min(), xs.max()
    u, d = ys.min(), ys.max()
    return img[l: r+1, u: d+1]


def zoompad(image, size, interpolation=cv2.INTER_LINEAR):
    """resize image by zooming (keeping ratio) + padding 0
    Input:
        image: [H, W[, C]], C can be the any value (i.e. greyscale or RGB sequence)
        size: int of List[int, int]
        interpolation: see https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html
    Output:
        resized_image: [H', W'[, C]]
    """
    if isinstance(size, int):
        assert size > 0
        size = [size, size]
    elif isinstance(size, (tuple, list)):
        assert len(size) == 2 and size[0] > 0 and size[1] > 0
    h, w = image.shape[:2]
    scale_h, scale_w = float(size[0]) / image.shape[0], float(size[1]) / image.shape[1]
    scale = min(scale_h, scale_w)
    tmp_size = [ # clipping to ensure size
        min(int(image.shape[0] * scale), size[0]),
        min(int(image.shape[1] * scale), size[1])
    ]
    image = cv2.resize(image, (tmp_size[1], tmp_size[0]), interpolation=interpolation)
    pad_h, pad_w = size[0] - image.shape[0], size[1] - image.shape[1]
    if pad_h > 0 or pad_w > 0:
        pad_left, pad_right = pad_w // 2, (pad_w + 1) // 2
        pad_top, pad_bottom = pad_h // 2, (pad_h + 1) // 2
        image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), constant_values=0)
    return image


if "__main__" == __name__:
    img = cv2.imread("img.png")
    assert img is not None
    img2 = crop(img)
    print(img.shape, img2.shape)
    cv2.imwrite(crop-img.png, img2)
