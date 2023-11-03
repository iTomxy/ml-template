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


if "__main__" == __name__:
    img = cv2.imread("img.png")
    assert img is not None
    img2 = crop(img)
    print(img.shape, img2.shape)
    cv2.imwrite(crop-img.png, img2)
