# ml-template

ML code templates for quick project starting with, or easy conversion between, these frameworks:

- tensorflow 1.12
- tensorflow 2.1
- pytorch 1.4

**Note**: The hierarchy here **DOES NOT** represent the real one. I usually put these files in one directory.<br/>
Cause multiple frameworks are involved here, files are organized according to their generality.

# TODO

1. add model saving & resuming

# Notes

1. Seems that for **PyTorch 0.3**, performing advanced indexing with `numpy.ndarray` is **NOT** supported, which is OK in PyTorch 1.4. So in *pytorch/voc2007.py*, indices (those `idx_*`) are converted to `torch.Tensor` before being used, representing the use in PyTorch 0.3.