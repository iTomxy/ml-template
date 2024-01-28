import random
import torch
import torchvision.transforms.functional as F
from .misc import seed_everything


class MultiCompose:
    """Extension of torchvision.transforms.Compose that accepts multiple inputs
    and ensures the same random seed is applied on each of these inputs at each transforms.
    This can be useful when simultaneously transforming images & segmentation masks.

    Usage:
        ```python
        train_trans = MultiCompose([
            # interpolation: image uses `bilinear`, label uses `nearest`
            [transforms.Resize((224, 256), transforms.InterpolationMode.BILINEAR),
             transforms.Resize((224, 256), transforms.InterpolationMode.NEAREST)],
            transforms.RandomAffine(30, (0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
            # apply `ColorJitter` on image but not on label (thus `None`)
            (transforms.ColorJitter(0.1, 0.2, 0.3, 0.4), None),
        ])

        # apply augmentations on both `image` and `seg_label`
        image, seg_label = train_trans(image, seg_label)
        ```
    """

    # numpy.random.seed range error:
    #   ValueError: Seed must be between 0 and 2**32 - 1
    MIN_SEED = 0 # - 0x8000_0000_0000_0000
    MAX_SEED = min(2**32 - 1, 0xffff_ffff_ffff_ffff)

    def __init__(self, transforms):
        no_op = lambda x: x  # i.e. identity function
        self.transforms = []
        for t in transforms:
            if isinstance(t, (tuple, list)):
            	# convert `None` to `no_op` for convenience
                self.transforms.append([no_op if _t is None else _t for _t in t])
            else:
                self.transforms.append(t)

    def __call__(self, *images):
        for t in self.transforms:
            if isinstance(t, (tuple, list)):
                # `<=` allows redundant transforms
                assert len(images) <= len(t), f"#inputs: {len(images)} v.s. #transforms: {len(self.transforms)}"
            else:
                t = [t] * len(images)

            _aug_images = []
            _seed = random.randint(self.MIN_SEED, self.MAX_SEED)
            for _im, _t in zip(images, t):
                seed_everything(_seed)
                _aug_images.append(_t(_im))

            images = _aug_images

        if len(images) == 1:
            images = images[0]
        return images


class ResizeZoomPad:
    """resize by zooming (to keep ratio aspect) & padding (to ensure size)
    Parameter:
        size: int or (int, int)
        interpolation: str / torchvision.transforms.functional.InterpolationMode
            can be {"nearest", "bilinear", "bicubic", "box", "hamming", "lanczos"}
    """
    def __init__(self, size, interpolation="bilinear"):
        if isinstance(size, int):
            assert size > 0
            self.size = [size, size]
        elif isinstance(size, (tuple, list)):
            assert len(size) == 2 and size[0] > 0 and size[1] > 0
            self.size = size

        if isinstance(interpolation, str):
            assert interpolation.lower() in {"nearest", "bilinear", "bicubic", "box", "hamming", "lanczos"}
            interpolation = {
                "nearest": F.InterpolationMode.NEAREST,
                "bilinear": F.InterpolationMode.BILINEAR,
                "bicubic": F.InterpolationMode.BICUBIC,
                "box": F.InterpolationMode.BOX,
                "hamming": F.InterpolationMode.HAMMING,
                "lanczos": F.InterpolationMode.LANCZOS
            }[interpolation.lower()]
        self.interpolation = interpolation

    def __call__(self, image):
        """image: [C, H, W]"""
        scale_h, scale_w = float(self.size[0]) / image.size(1), float(self.size[1]) / image.size(2)
        scale = min(scale_h, scale_w)
        tmp_size = [ # clipping to ensure size
            min(int(image.size(1) * scale), self.size[0]),
            min(int(image.size(2) * scale), self.size[1])
        ]
        image = F.resize(image, tmp_size, self.interpolation)
        assert image.size(1) <= self.size[0] and image.size(2) <= self.size[1]
        pad_h, pad_w = self.size[0] - image.size(1), self.size[1] - image.size(2)
        if pad_h > 0 or pad_w > 0:
            pad_left, pad_right = pad_w // 2, (pad_w + 1) // 2
            pad_top, pad_bottom = pad_h // 2, (pad_h + 1) // 2
            image = F.pad(image, (pad_left, pad_top, pad_right, pad_bottom))
        return image


class PermutePatch:
    """divide image into patches & permute patch order"""

    def __init__(self, h_div, w_div, ensure_displace=False):
        """
        Input:
            h_div: int, #patches to divide into along heigth
            w_div: int, #patches to divide into along width
            ensure_displace: bool, ensure NO patch stay in its original position
        """
        assert isinstance(h_div, int) and isinstance(w_div, int)
        self.h_div = h_div
        self.w_div = w_div
        self.ensure_displace = ensure_displace

    def permute_axis(self, image, axis, div):
        """
        Input:
            image: [C, H, W], torch.Tensor
            axis: int, in {1, 2} (i.e. {height, width}), along wich axis to permute patches
            div: # of patches to divide along this axis
        Output:
            the permuted image, [C, H, W], torch.Tensor
        """
        assert axis in (1, 2) # only h(1) & w(2)
        image = torch.moveaxis(image, axis, 0)
        tmp = torch.zeros_like(image)
        div = min(div, image.size(0))
        rnd_idx = torch.randperm(div)
        if self.ensure_displace:
            # adjust `rnd_idx` to ensure that rnd_idx[i] != i for all i
            idx = torch.arange(div)
            fixed_mask = (rnd_idx == idx)
            fixed_num = fixed_mask.sum() # num of fixed point
            if 1 == fixed_num: # find another one to swap
                fixed_pos = idx[fixed_mask].item()
                for pos in range(rnd_idx.size(0)):
                    if pos != fixed_pos and rnd_idx[fixed_pos] != pos and rnd_idx[pos] != fixed_pos:
                        rnd_idx[fixed_pos], rnd_idx[pos] = rnd_idx[pos].item(), rnd_idx[fixed_pos].item()
                        break
            elif fixed_num > 1: # simply left shift
                fixed_idx = rnd_idx[fixed_mask]
                rnd_idx[fixed_mask] = torch.cat([fixed_idx[1:], fixed_idx[:1]])
            assert (rnd_idx == idx).sum() == 0
        step = image.size(0) // div
        lens = [step] * (div - 1) + [image.size(0) - step * (div - 1)]
        starts = [0]
        for i in range(1, div):
            starts.append(starts[i - 1] + lens[i - 1])
        cur = 0
        for idx in rnd_idx:
            tmp[cur: cur + lens[idx]] = image[starts[idx]: starts[idx] + lens[idx]]
            cur += lens[idx]
        return torch.moveaxis(tmp, 0, axis)

    def __call__(self, image):
        """image: [C, H, W], torch.Tensor"""
        image = self.permute_axis(image, 1, self.h_div) # permute vertically
        image = self.permute_axis(image, 2, self.w_div) # permute horizontally
        return image
