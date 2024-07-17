import os, math
import numpy as np
from PIL import Image


def get_palette(n_classes, pil_format=True):
    """Returns the color map for visualizing the segmentation mask.
    Example:
        ```python
        palette = get_palette(n_classes, True)
        seg_mask = seg_model(image) # int, [H, W], in [0, n_classes]
        seg_img = PIL.Image.fromarray(seg_mask)
        seg_img.putpalette(palette)
        seg_img.convert("RGB").save("seg.jpg")
        ```
    Args:
        n_classes: int, number of classes
        pil_format: bool, whether in format suitable for `PIL.Image.putpalette`.
            see: https://pillow.readthedocs.io/en/stable/reference/ImagePalette.html
    Returns:
        palette: [(R_i, G_i, B_i)] if `pil_format` is False, or
            [R1, G1, B1, R2, G2, B2, ...] if `pil_format` is True
    """
    n = n_classes
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3

    if pil_format:
        return palette

    res = []
    for i in range(0, len(palette), 3):
        res.append(tuple(palette[i: i+3]))
    return res


def color_seg(label, n_classes=0):
    """put colour on the segmentation mask
    Input:
        label: [H, W], int numpy.ndarray
        n_classes: int, num of classes (including background), inferred from `label` if not provided
    Output:
        label_rgb: [H, W, 3], PIL.Image
    """
    if n_classes < 1:
        n_classes = math.ceil(np.max(label)) + 1
    label_rgb = Image.fromarray(label.astype(np.int32)).convert("L")
    label_rgb.putpalette(get_palette(n_classes))
    return label_rgb.convert("RGB")


def blend_seg(image, label, n_classes=0, alpha=0.7, rescale=False, transparent_bg=True, save_file=""):
    """blend image & pixel-level label/prediction
    Input:
        image: [H, W] or [H, W, C], int numpy.ndarray, in [0, 255]
        label: [H, W], int numpy.ndarray
        n_classes: int, num of classes (including background), inferred from `label` if not provided
        alpha: float in (0, 1)
        rescale: bool, normalise & scale to [0, 255] if True
        transparent_bg: bool, don't colour (i.e. use original image pixel value for) background pixels if True
        save_file: str, path to save the blended image
    Output:
        blended_image: PIL.Image
    """
    if rescale:
        denom = image.max() - image.min()
        if 0 != denom:
            image = (image - image.min()) / denom * 255
        image = np.clip(image, 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(image).convert("RGB")
    lab_pil = color_seg(label, n_classes)
    blended_image = Image.blend(img_pil, lab_pil, alpha)
    if transparent_bg:
        blended_image = Image.fromarray(np.where(
            (0 == label)[:, :, np.newaxis],
            np.asarray(img_pil),
            np.asarray(blended_image)
        ))
    if save_file:
        blended_image.save(save_file)
    return blended_image


def show_nii(nii_file):
    """show medical volume in .nii/.nii.gz format
    Input:
        nii_file: str, path to .nii/.nii.gz file
    """
    if not os.path.isfile(nii_file):
        print("No such file:", nii_file)
        return
    import nibabel as nib
    from nibabel.viewers import OrthoSlicer3D
    img = nib.load(nii_file)
    OrthoSlicer3D(img.dataobj).show()


def compact_image_grid(image_list, exact=False):
    """adaptively arrange images in a compactest 2D grid (for better visualisation)
    Input:
        image_list: list of images in format of [h, w] or [h, w, c] numpy.ndarray
        exact: bool, subjest to #grids = #images or not.
            If False, #grids > #images may happen for a more compact view.
    Output:
        grid: [H, W] or [H, W, c], compiled images
    """
    n = len(image_list)
    if 1 == n:
        return image_list[0]

    # max image resolution
    max_h, max_w = 0, 0
    for im in image_list:
        h, w = im.shape[:2]
        max_h = max(max_h, h)
        max_w = max(max_w, w)

    # find compactest layout
    nr, nc = 1, n
    min_peri = nr * max_h + nc * max_w # 1 row
    for r in range(2, n + 1):
        if exact and n % r != 0:
            continue
        c = math.ceil(n / r)
        assert r * c >= n and r * (c - 1) <= n
        peri = r * max_h + c * max_w
        if peri < min_peri:
            nr, nc, min_peri = r, c, peri
    assert nr * nc >= n

    grid_shape = (nr * max_h, nc * max_w) + image_list[0].shape[2:]
    grid = np.zeros(grid_shape, dtype=image_list[0].dtype)
    for i, img in enumerate(image_list):
        r, c = i // nc, i % nc
        h, w = img.shape[:2]
        grid[r*max_h: r*max_h+h, c*max_w: c*max_w+w] = img

    return grid


if "__main__" == __name__:
    nc, sz = 24, 17
    palette = get_palette(nc, False)
    print(palette)
    img = np.zeros((sz, nc * sz, 3), dtype=np.uint8)
    for c, p in enumerate(palette):
        img[:, c*sz: (c+1)*sz] = p
    Image.fromarray(img).save("palette.jpg")

    img = np.load("mr_train_1014_image_63.npy")
    lab = np.load("mr_train_1014_label_63.npy")
    img = (img - img.min()) / (img.max() - img.min()) * 255 # normalisation
    img = img.astype(np.uint8)
    blend_seg(img, lab, save_file="blend.png")

    show_nii("sub-verse004_seg-vert_msk.nii.gz")

    p = os.path.expanduser(r"~\Pictures\wallpaper")
    img_list = []
    for i, f in enumerate(os.listdir(p)):
        img = np.asarray(Image.open(os.path.join(p, f)).resize((224, 224)))
        if img.ndim < 3: continue
        img_list.append(img[:, :, :3])
        if len(img_list) >= 17: break
    Image.fromarray(compact_image_grid(img_list, False)).save("grid.png")
    Image.fromarray(compact_image_grid(img_list, True)).save("grid-exact.png")
