# ml-template

ML code templates for quick project starting with, or easy conversion between, these frameworks:

- tensorflow 1.12
- tensorflow 2.1
- pytorch 1.4

**Note**: The hierarchy here **DOES NOT** represent the real one. I usually put these files in one directory.<br/>
Cause multiple frameworks are involved here, files are organized according to their generality.

# TODO

1. add model saving & resuming, with info string indicating the information of model, loss, epoch, performance, etc.
2. add forward/backward hook to diagnose model, see [3]
3. hard mining for triplet, see [5]

# References

1. [AlexNet implementation + weights in TensorFlow](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/)
2. [mikechen66/AlexNet_TensorFlow2.0-2.2](https://github.com/mikechen66/AlexNet_TensorFlow2.0-2.2)
3. [Finding source of NaN in forward pass](https://discuss.pytorch.org/t/finding-source-of-nan-in-forward-pass/51153)
4. [torch.autograd.detect_anomaly](https://pytorch.org/docs/1.4.0/autograd.html#torch.autograd.detect_anomaly)
5. [deep-cross-modal-hashing/torchcmh/dataset/base/triplet.py](https://github.com/WangGodder/deep-cross-modal-hashing/blob/master/torchcmh/dataset/base/triplet.py)