# ml-template

ML code templates for quick project starting with, or easy conversion between, these frameworks:

- tensorflow 1.12
- tensorflow 2.1
- pytorch 1.4
- matlab R2018a

**Note**: The hierarchy here **DOES NOT** represent the real one. I usually put these files in one directory.<br/>
Cause multiple frameworks are involved here, files are organized according to their generality.

- *+itom/*: some tool functions that might be used in matlab codes.

# TODO

1. add model saving & resuming, with info string indicating the information of model, loss, epoch, performance, etc.
2. add forward/backward hook to diagnose model, see [3]
3. hard mining for triplet, see [5]

# Docker

1. [TensorFlow 1.12](https://hub.docker.com/layers/tensorflow/tensorflow/1.12.0-gpu-py3/images/sha256-413b9533f92a400117a23891d050ab829e277a6ff9f66c9c62a755b7547dbb1e?context=explore)
2. [TensorFlow 2.1](https://hub.docker.com/layers/tensorflow/tensorflow/2.1.0-gpu-py3/images/sha256-1010e051dde4a9b62532a80f4a9a619013eafc78491542d5ef5da796cc2697ae?context=explore)
3. [PyTorch 1.4](https://hub.docker.com/layers/pytorch/pytorch/1.4-cuda10.1-cudnn7-runtime/images/sha256-ee783a4c0fccc7317c150450e84579544e171dd01a3f76cf2711262aced85bf7?context=explore)
4. [PyTorch 0.3](https://hub.docker.com/layers/floydhub/pytorch/0.3.1-gpu.cuda9cudnn7-py3.38/images/sha256-f130384d52e5e5542a78db8b7d7ead8885fd73a84cca8cc5a7c7a755a192da37?context=explore)

# hosts

some entries to be added in *hosts* file:

```
140.82.114.4	github.com
199.232.5.194	github.global.ssl.fastly.net
199.232.68.133 raw.githubusercontent.com
```

# References

1. [AlexNet implementation + weights in TensorFlow](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/)
2. [mikechen66/AlexNet_TensorFlow2.0-2.2](https://github.com/mikechen66/AlexNet_TensorFlow2.0-2.2)
3. [Finding source of NaN in forward pass](https://discuss.pytorch.org/t/finding-source-of-nan-in-forward-pass/51153)
4. [torch.autograd.detect_anomaly](https://pytorch.org/docs/1.4.0/autograd.html#torch.autograd.detect_anomaly)
5. [deep-cross-modal-hashing/torchcmh/dataset/base/triplet.py](https://github.com/WangGodder/deep-cross-modal-hashing/blob/master/torchcmh/dataset/base/triplet.py)
6. [【BUG】[nltk_data] Error loading punkt: ＜urlopen error [Errno 11004] [nltk_data]](https://blog.csdn.net/xiangduixuexi/article/details/108601873)