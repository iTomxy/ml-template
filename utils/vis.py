def get_palette(n_classes, pil_format=False):
    """ Returns the color map for visualizing the segmentation mask.
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


if "__main__" == __name__:
    import numpy as np
    from PIL import Image
    nc, sz = 24, 17
    palette = get_palette(nc, False)
    print(palette)
    img = np.zeros((sz, nc * sz, 3), dtype=np.uint8)
    for c, p in enumerate(palette):
        img[:, c*sz: (c+1)*sz] = p
    Image.fromarray(img).save("palette.jpg")
