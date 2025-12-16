import os, os.path as osp
import cv2
import numpy as np
import nibabel as nib
import SimpleITK as sitk
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


def determine_reorient(src_ornt, trg_ornt):
    """Determine whether a reorientation is needed, calculate the operation parameters if yes.
    Orientation = axis order + direction along each axis. Correspondence:
        - x: L, R
        - y: A, P
        - z: S, I
    Input:
        src_ornt: Tuple[char], original orientation, e.g. ('R', 'A', 'S')
        trg_ornt: Tuple[char], target orientation, e.g. ('L', 'P', 'S')
    Output:
        need: bool, False if no operation is needed (i.e. already the wanted orientation)
        axis_order: int[3], in {0, 1, 2}, the new axis order
        flip_flags: bool[3], whether this axis should be flipped, same order as trg_ornt
    """
    src_ornt = tuple(s.upper() for s in src_ornt)
    trg_ornt = tuple(s.upper() for s in trg_ornt)
    to_axis = {
        'L': 'X', 'R': 'X', 'X': 'X', # x-axis
        'A': 'Y', 'P': 'Y', 'Y': 'Y', # y-axis
        'S': 'Z', 'I': 'Z', 'Z': 'Z', # z-axis
    }
    assert len(set([to_axis(s) for s in src_ornt])) == 3, "Duplicated axes in `src_ornt': {}".format(src_ornt)
    assert len(set([to_axis(t) for t in trg_ornt])) == 3, "Duplicated axes in `trg_ornt': {}".format(trg_ornt)
    flag_diff = False
    for s, t, in zip(src_ornt, trg_ornt):
        if to_axis[s] != to_axis[t] or ( # diff axis order
            s != t and t not in "XYZ"    # same axis order, but diff direction
        ):
            flag_diff = True
            break

    if not flag_diff:
        return False, None, None

    opposites = {'L': 'R', 'R': 'L', 'P': 'A', 'A': 'P', 'I': 'S', 'S': 'I'}
    axis_order = []
    flip_flags = []
    for t in trg_ornt:
        for src_axis, s in enumerate(src_ornt):
            if to_axis[s] == to_axis[t]:
                axis_order.append(src_axis)
                if t in "XYZ":
                    flip_flags.append(False)
                else: # t not in {X, Y, Z}
                    assert s not in "XYZ", "Cannot resolve: {} -> {}".format(s, t)
                    flip_flags.append(s == opposites[t])

                break # found corresponding axis pair

    # NOTE first perform axis re-ordering, then flip,
    # Because `flip_flags' is in the same order as `trg_ornt'.
    return True, axis_order, flip_flags


def reorient_3dgrid(vol, src_ornt, trg_ornt):
    """numpy-based reorientation, support using x, y, z to indicate axis order but keep direction
    Orientation = axis order + direction along each axis. Correspondence:
        - x: L, R
        - y: A, P
        - z: S, I
    Input:
        vol: np.ndarray, [H, W, L], original volume
        src_ornt: Tuple[char], original orientation, e.g. ('R', 'A', 'S')
        trg_ornt: Tuple[char], target orientation, e.g. ('L', 'P', 'S')
    Output:
        result: np.ndarray, [H', W', L'], reoriented volume
    """
    flag_need, axis_order, flip_flags = determine_reorient(src_ornt, trg_ornt)
    if not flag_need:
        return vol

    # 1. Apply axis permutation
    result = np.transpose(vol, axis_order)
    # 2. Apply flips where needed
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
    flag_need, axis_order, flip_flags = determine_reorient(src_ornt, trg_ornt)
    if not flag_need:
        return points

    # 1. re-order axes
    result = points[..., axis_order]
    axes_range = [axes_range[i] for i in axis_order]
    # 2. flip axes
    for coord_axis, (flip, (_min, _max)) in enumerate(zip(flip_flags, axes_range)):
        if flip:
            result[..., coord_axis] = _min + _max - result[..., coord_axis]

    return result


def ct_window(image, level, width, norm=False, to_image=0):
    """apply windowing on a CT slice
    Input:
        image: float[H, W], numpy.ndarray raw CT slice
        level: float, window level (centre)
        width: float, window width
        norm: bool = False, normalise intensity (HU value) to [0, 1] or not
        to_image: int{0, 1, 3} = 0, convert to image format or not
            - 0: do not convert, float[H, W]
            - 1: convert to grey scale, uint8[H, W] in [0, 255]
            - 3: convert to RGB, uint8[H, W, 3] in [0, 255]
    Output:
        image: numpy.ndarray, transformed image
    """
    assert to_image in (0, 1, 3)
    level = float(level)
    width = float(width)
    image = np.clip(image, level - width, level + width)
    if norm or to_image > 0:
        assert width > 0
        image -= level - width # [level - width, level + width] -> [0, 2 * width]
        image /= 2 * width # -> [0, 1]
        image = np.clip(image, 0, 1)
        if to_image > 0:
            image = np.clip(image * 255, 0, 255).astype(np.uint8) # grey scale
            if 3 == to_image:
                image = np.asarray(Image.fromarray(image).convert('RGB'))

    return image


def adjust_spacing(points, old_spacing, new_spacing):
    """Adjust point coordinates for new spacing.
    Args:
        points: [n, 3] array of point coordinates
        old_spacing: [3] array of old spacing (z, y, x) or (h, w, l)
        new_spacing: [3] array of new spacing
    Returns:
        adjusted_points: [n, 3] array with adjusted coordinates
    """
    old_spacing = np.array(old_spacing)
    new_spacing = np.array(new_spacing)

    # Scale factor from old to new spacing
    scale_factor = old_spacing / new_spacing

    # Apply scaling
    adjusted_points = points * scale_factor

    return adjusted_points


def normalise_pc(points, centroid=None, radius=None, return_params=False):
    """normalise point cloud: (points - centroid) / radius
    Input:
        points: float[n, 3], numpy.ndarray
        centroid: float[3] = None
        radius: float = None
        return_params: bool = False, whether return `centroid` and `radius`
    """
    points = points.astype(np.float32)

    if centroid is None:
        centroid = np.mean(points, axis=0)
    else:
        centroid = np.asarray(centroid, dtype=np.float32)

    points = points - centroid

    if radius is None:
        radius = np.max(np.sqrt(np.sum(points**2, axis=1)))
    else:
        radius = float(radius)

    points = points / radius

    if return_params:
        return points, centroid, radius
    return points


def axcodes2dir(axcode):
    """Convert axcode string to direction cosine matrix.
    Ref: https://nipy.org/nibabel/reference/nibabel.orientations.html#nibabel.orientations.aff2axcodes
    Input:
        orientation: str|Tuple[char], e.g. "RAI", ('L', 'P', 'S')
    Output:
       direction: float[9]: serialised 3x3 direction matrix in row-major order
    """
    assert len(set(axcode)) == 3
    axis_map = {
        'R': [1, 0, 0], 'L': [-1, 0, 0],
        'A': [0, 1, 0], 'P': [0, -1, 0],
        'S': [0, 0, 1], 'I': [0, 0, -1]
    }
    direction = []
    for code in axcode:
        direction.extend(axis_map[code.upper()])

    return direction


def np2sitk(image, axcode, spacing=(1.0, 1.0, 1.0)):#, origin=(0.0, 0.0, 0.0)):
    """Convert numpy array to SimpleITK image.
    Input:
        image: [L, H, W], numpy array
        axcode: str[3], orientation of `image', e.g. "RAI"
        spacing: float[3], spacing of each axis
    Output:
        SimpleITK image
    """
    # first reorient `image` to z-y-x order, expected by SimpleITK
    image = reorient_3dgrid(image, axcode, "ZYX")
    # re-order `spacing` accordingly
    flag, axis_order, flip_flags = determine_reorient(axcode, "ZYX")
    if flag:
        spacing = tuple(spacing[i] for i in axis_order)
        # origin = tuple(origin[i] for i in axis_order) # FIXME: how to adjust origin?

    image_sitk = sitk.GetImageFromArray(image)
    image_sitk.SetSpacing(spacing)
    # image_sitk.SetOrigin(origin)
    image_sitk.SetDirection(axcodes2dir(axcode))
    return image_sitk


def np2nifti(image, axcode, spacing=(1.0, 1.0, 1.0)):
    """Convert numpy array to nibabel.Nifti1Image.
    Input:
        image: [L, H, W], numpy array
        axcode: str[3], orientation of `image', e.g. "RAI"
        spacing: float[3], spacing of each axis
    Output:
        nibabel.Nifti1Image
    """
    affine = np.eye(4)
    direction = axcodes2dir(axcode)
    direction_matrix = np.array(direction).reshape(3, 3)
    for i in range(3):
        affine[:3, i] = direction_matrix[:, i] * spacing[i]

    return nib.Nifti1Image(image, affine=affine)


if "__main__" == __name__:
    img = cv2.imread("img.png")
    assert img is not None
    img2 = crop(img)
    print(img.shape, img2.shape)
    cv2.imwrite("crop-img.png", img2)
