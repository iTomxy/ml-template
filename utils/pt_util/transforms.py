import random
import torchvision.transforms.functional as F
from .misc import seed_everything


def to_multi(trfm):
    """wrap a transform to extend to multiple input with synchronised random seed
    Input:
        trfm: transformation function/object (custom or from torchvision.transforms)
    Output:
        _multi_transform: function
    """
    # numpy.random.seed range error:
    #   ValueError: Seed must be between 0 and 2**32 - 1
    min_seed = 0 # - 0x8000_0000_0000_0000
    max_seed = min(2**32 - 1, 0xffff_ffff_ffff_ffff)
    def _multi_transform(*images):
        """images: [C, H, W]"""
        if len(images) == 1:
            return trfm(images)
        _seed = random.randint(min_seed, max_seed)
        res = []
        for img in images:
            seed_everything(_seed)
            res.append(trfm(img))
        return tuple(res)

    return _multi_transform


class MultiCompose:
    """Extension of torchvision.transforms.Compose that accepts multiple input.
    Usage is the same as torchvision.transforms.Compose. This class will wrap input
    transforms with `to_multi` to support simultaneous multiple transformation.
    This can be useful when simultaneously transforming images & segmentation masks.
    """
    def __init__(self, transforms):
        """transforms should be wrapped by `to_multi`"""
        self.transforms = [to_multi(t) for t in transforms]

    def __call__(self, *images):
        for t in self.transforms:
            if len(images) == 1:
                images = t(images)
            else:
                images = t(*images)
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
