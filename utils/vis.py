import os, os.path as osp
import numpy as np
from PIL import Image


def get_palette(n_classes, pil_format=True):
    """ Returns the color map for visualizing the segmentation mask.
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
    if n_classes < 1:
        n_classes = np.max(label) + 1
    lab_pil = Image.fromarray(label).convert("L")
    lab_pil.putpalette(get_palette(n_classes))
    blended_image = Image.blend(img_pil, lab_pil.convert("RGB"), alpha)
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
    if not osp.isfile(nii_file):
        print("No such file:", nii_file)
        return
    import nibabel as nib
    from nibabel.viewers import OrthoSlicer3D
    img = nib.load(nii_file)
    OrthoSlicer3D(img.dataobj).show()


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
