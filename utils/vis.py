import os, math, itertools, multiprocessing as mp
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


def color_seg(label, n_classes=0, palette=[]):
    """put colour on the segmentation mask
    Input:
        label: [H, W], int numpy.ndarray
        n_classes: int = 0, num of classes (including background), inferred from `label` if not provided.
        palette: List[RGB] = [], PIL-format palette, use this if provided, else use generated one.
    Output:
        label_rgb: [H, W, 3], PIL.Image
    """
    if n_classes < 1:
        n_classes = math.ceil(np.max(label)) + 1
    label_rgb = Image.fromarray(label.astype(np.int32)).convert("L")
    if len(palette) > 0:
        assert len(palette) >= 3 * n_classes # RGB for each class
    else:
        palette = get_palette(n_classes)
    label_rgb.putpalette(palette)
    return label_rgb.convert("RGB")


def blend_seg(image, label, n_classes=0, alpha=0.7, rescale=False, transparent_ids=[], palette=[], save_file=""):
    """blend image & pixel-level label/prediction
    Input:
        image: [H, W] or [H, W, C], int numpy.ndarray, in [0, 255]
        label: [H, W], int numpy.ndarray
        n_classes: int = 0, num of classes (including background), inferred from `label` if not provided.
        alpha: float = 0.7, in (0, 1)
        rescale: bool = False, normalise & scale image to [0, 255] if True.
        transparent_ids: List[int] = [], don't blend pixels with these label IDs, use original image pixel instead.
        palette: List[RGB] = [], PIL-format palette, use this if provided, else use generated one.
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
    lab_pil = color_seg(label, n_classes, palette)
    blended_image = Image.blend(img_pil, lab_pil, alpha)
    if len(transparent_ids) > 0:
        blended_image = Image.fromarray(np.where(
            np.isin(label, transparent_ids)[:, :, np.newaxis],
            np.asarray(img_pil),
            np.asarray(blended_image)
        ))

    if save_file:
        os.makedirs(os.path.dirname(save_file) or '.', exist_ok=True)
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


def compact_image_grid(image_list, exact=False, high_first=False):
    """adaptively arrange images in a compactest 2D grid (for better visualisation)
    Input:
        image_list: list of images in format of [h, w] or [h, w, c] numpy.ndarray
        exact: bool, subjest to #grids = #images or not.
            If False, #grids > #images may happen for a more compact view.
        high_first: bool = True, if two layouts have the same compatness, high triumph wide
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
    if high_first:
        # consider transposed case
        max_h, max_w = max_w, max_h

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

    if high_first:
        # transpose back
        nr, nc = nc, nr
        max_h, max_w = max_w, max_h

    assert nr * nc >= n

    grid_shape = (nr * max_h, nc * max_w) + image_list[0].shape[2:]
    grid = np.zeros(grid_shape, dtype=image_list[0].dtype)
    for i, img in enumerate(image_list):
        r, c = i // nc, i % nc
        h, w = img.shape[:2]
        grid[r*max_h: r*max_h+h, c*max_w: c*max_w+w] = img

    return grid


def show_ply(ply_file):
    """show .ply file with open3d"""
    if not os.path.isfile(ply_file):
        print("No such file:", ply_file)
        return

    import open3d as o3d
    pcd = o3d.io.read_point_cloud(ply_file)
    o3d.visualization.draw_geometries([pcd])


def vis_point_cloud(xyz, label=None, n_classes=0, palette=None, window_name="Open3D"):
    """visualise 1 point cloud (label or prediction)
    xyz: int|float[n, 3], numpy.ndarray
    label: int[n] = None
    n_classes: int = 0
    palette: int[m, 3] = None, m >= n_classes
    window_name: str = "Open3D"
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if label is not None:
        assert len(label) == xyz.shape[0]
        label = np.asarray(label)
        if n_classes < 1:
            n_classes = label.max() + 1
        if palette is None:
            palette = np.asarray(get_palette(n_classes, False))
        elif not isinstance(palette, np.ndarray):
            palette = np.asarray(palette)
        assert palette.shape[0] >= n_classes and 3 == palette.shape[1], "Palette ({}) too small for {} classes".format(palette.shape, n_classes)
        colors = np.asarray([palette[c] for c in label])
        colors = colors.astype(np.float32) / 255
        pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd], window_name=window_name)


def vis_multi_pc(xyzs, labels=[], class_nums=[], palettes=[], windows_name=[]):
    """visualise multiple point clouds (label or prediction) simultaneously by calling `vis_point_cloud` with multi-processing
    xyzs: list of int|float[n, 3], numpy.ndarray
    labels: list of int[n] = None
    class_nums: list of int = 0
    palettes: list of int[m, 3] = None, m >= n_classes
    windows_name: list of str = "Open3D"
    """
    assert isinstance(xyzs, (list, tuple))
    if len(labels) == 0:
        labels = [None] * len(xyzs)
    if len(class_nums) == 0:
        class_nums = [0] * len(xyzs)
    if len(palettes) == 0:
        palettes = [None] * len(xyzs)
    if len(windows_name) == 0:
        windows_name = ["Open3D {}".format(i) for i in range(len(xyzs))]
    assert len(xyzs) == len(labels) == len(class_nums) == len(palettes) == len(windows_name)

    p_list = []
    for xyz, label, nc, palette, wn in zip(xyzs, labels, class_nums, palettes, windows_name):
        p = mp.Process(target=vis_point_cloud, args=(xyz, label, nc, palette, wn))
        p.start()
        p_list.append(t)
        # p.join() # do NOT join here

    for p in p_list:
        p.join()


def bbox3d_points(point1, point2):
    """Generate all integer positions (xyz) of a 3D bounding-box defined by its two diagonal points.
    Input:
        point1: List or tuple [x1, y1, z1] representing one diagonal corner.
        point2: List or tuple [x2, y2, z2] representing the opposite diagonal corner.
    Output:
        return: int[n, 3]
    """
    # Get min/max values
    x_min, x_max = min(point1[0], point2[0]), max(point1[0], point2[0])
    y_min, y_max = min(point1[1], point2[1]), max(point1[1], point2[1])
    z_min, z_max = min(point1[2], point2[2]), max(point1[2], point2[2])

    # Generate 8 vertices
    vertices = list(itertools.product([x_min, x_max], [y_min, y_max], [z_min, z_max]))

    # Define edges using pairs of vertex indices
    edge_indices = np.array([
        [0, 1], [0, 2], [0, 4],
        [1, 3], [1, 5],
        [2, 3], [2, 6],
        [3, 7],
        [4, 5], [4, 6],
        [5, 7], [6, 7]
    ])

    # Store all edge points
    all_edge_points = set()

    # Generate integer points along each edge
    for edge in edge_indices:
        start, end = np.array(vertices[edge[0]]), np.array(vertices[edge[1]])
        # Get range for each coordinate axis
        x_range = np.arange(start[0], end[0] + np.sign(end[0] - start[0]), np.sign(end[0] - start[0])) if start[0] != end[0] else [start[0]]
        y_range = np.arange(start[1], end[1] + np.sign(end[1] - start[1]), np.sign(end[1] - start[1])) if start[1] != end[1] else [start[1]]
        z_range = np.arange(start[2], end[2] + np.sign(end[2] - start[2]), np.sign(end[2] - start[2])) if start[2] != end[2] else [start[2]]

        # Create points along the edge
        for x, y, z in zip(
            np.broadcast_to(x_range, max(len(x_range), len(y_range), len(z_range))),
            np.broadcast_to(y_range, max(len(x_range), len(y_range), len(z_range))),
            np.broadcast_to(z_range, max(len(x_range), len(y_range), len(z_range)))
        ):
            all_edge_points.add((x, y, z))

    return np.array(list(all_edge_points))


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

    show_ply(r"C:\Users\24647087\OneDrive - UTS\Desktop\test\ptcloud\pred-507-err.ply")
