import os, os.path as osp
import cv2
import numpy as np
import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform, apply_orientation


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


def reorient_nib(image_nii, target_orientation=('L', 'P', 'S'), dtype=None):
    """Change to target orientation if not already is.
    Input:
        image_nii: nibabel.nifti1.Nifti1Image
        target_orientation: Tuple(char) = ('L', 'P', 'S')
        dtype: numpy data type, cast the data if provided, otherwise reoriented
            volume will be float64 due to the nibabel `get_fdata()`.
    Output:
        image_nii: nibabel.nifti1.Nifti1Image
    """
    target_orientation = tuple(s.upper() for s in target_orientation)
    original_orientation = nib.aff2axcodes(image_nii.affine)
    if original_orientation == target_orientation:
        return image_nii

    # Convert orientation strings to orientation matrices
    orig_ornt = axcodes2ornt(original_orientation)
    target_ornt = axcodes2ornt(target_orientation)

    # Get the transform from original to target orientation
    transform = ornt_transform(orig_ornt, target_ornt)

    # Apply the orientation transform to the image
    image_data = apply_orientation(image_nii.get_fdata(), transform)
    if dtype is not None:
        image_data = image_data.astype(dtype)

    # Create new affine for the transformed image
    affine = image_nii.affine.copy()
    affine = nib.orientations.inv_ornt_aff(transform, image_nii.shape)
    affine = np.dot(image_nii.affine, affine)

    # Create new oriented image
    image_nii = image_nii.__class__(image_data, affine, image_nii.header)

    # Check orientation after reorientation
    assert nib.aff2axcodes(image_nii.affine) == target_orientation
    return image_nii


def reorient_3dgrid(vol, src_ornt, trg_ornt):
    """numpy-based reorientation
    Input:
        vol: np.ndarray, [H, W, L], original volume
        src_ornt: Tuple[char], original orientation, e.g. ('R', 'A', 'S')
        trg_ornt: Tuple[char], target orientation, e.g. ('L', 'P', 'S')
    Output:
        result: np.ndarray, [H', W', L'], reoriented volume
    """
    src_ornt = tuple(s.upper() for s in src_ornt)
    trg_ornt = tuple(s.upper() for s in trg_ornt)
    if src_ornt == trg_ornt:
        return vol

    opposites = {'L': 'R', 'R': 'L', 'P': 'A', 'A': 'P', 'I': 'S', 'S': 'I'}
    # Find axis mapping and flip requirements
    axis_order = []
    flip_flags = []

    for _t in trg_ornt:
        # Find which source axis corresponds to this target direction
        for src_axis, _s in enumerate(src_ornt):
            if _s == _t:
                # Same direction - no flip needed
                axis_order.append(src_axis)
                flip_flags.append(False)
                break
            elif _s == opposites[_t]:
                # Opposite direction - flip needed
                axis_order.append(src_axis)
                flip_flags.append(True)
                break

    # Apply axis permutation
    result = np.transpose(vol, axis_order)

    # Apply flips where needed
    for axis, flip in enumerate(flip_flags):
        if flip:
            result = np.flip(result, axis=axis)

    return result


def reorient_points(points, src_ornt, trg_ornt, axes_range):
    """reorient a set of 3D points
    Input:
        points: [..., 3], numpy.ndarray
        src_ornt: Tuple[char], original orientation, e.g. ('R', 'A', 'S')
        trg_ornt: Tuple[char], target orientation, e.g. ('L', 'P', 'S')
        axes_range: [(x1, x2), (y1, y2), (z1, z2)], range of point coordinates
            along each axis, all inclusive, in original axis order.
    Output:
        result: [..., 3], numpy.ndarray, reoriented points
    """
    src_ornt = tuple(s.upper() for s in src_ornt)
    trg_ornt = tuple(s.upper() for s in trg_ornt)
    if src_ornt == trg_ornt:
        return points

    # Define opposite directions
    opposites = {'L': 'R', 'R': 'L', 'P': 'A', 'A': 'P', 'I': 'S', 'S': 'I'}
    # Find axis mapping and flip requirements
    axis_order = []
    flip_flags = []
    for _t in trg_ornt:
        # Find which source axis corresponds to this target direction
        for src_axis, _s in enumerate(src_ornt):
            if _s == _t:
                # Same direction - no flip needed
                axis_order.append(src_axis)
                flip_flags.append(False)
                break
            elif _s == opposites[_t]:
                # Opposite direction - flip needed
                axis_order.append(src_axis)
                flip_flags.append(True)
                break

    # re-order axes
    result = points[..., axis_order]
    axes_range = [axes_range[i] for i in axis_order]
    # flip axes
    for coord_axis, (flip, (_min, _max)) in enumerate(zip(flip_flags, axes_range)):
        if flip:
            result[..., coord_axis] = _min + _max - result[..., coord_axis]

    return result


if "__main__" == __name__:
    img = cv2.imread("img.png")
    assert img is not None
    img2 = crop(img)
    print(img.shape, img2.shape)
    cv2.imwrite("crop-img.png", img2)
