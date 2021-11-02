# ml-template

ML code templates for quick project starting with, or easy conversion between, these frameworks:

- tensorflow 1.12
- tensorflow 2.1
- pytorch 1.4
- matlab R2018a

**Note**: The hierarchy here **DOES NOT** represent the real one. I usually put these files in one directory.<br/>
Cause multiple frameworks are involved here, files are organized according to their generality.

- *+itom/*: some tool functions that might be used in matlab codes.
- *requirements.txt*: some common packages I met that a docker image may not contains.

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

some entries to be added in *hosts* file for [GitHub](https://github.com/) accessing:

```
140.82.114.4	github.com
199.232.5.194	github.global.ssl.fastly.net
199.232.68.133	raw.githubusercontent.com
```

see [9] for automatic *hosts* updating.

# pip

change the source of `pip`, see [7, 8].

While you can add the configuration file yourself, a quicker way would be using one of the following commands:

```shell
# Tsinghua
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# Alibaba
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# Tencent
pip config set global.index-url http://mirrors.cloud.tencent.com/pypi/simple
# Douban
pip config set global.index-url http://pypi.douban.com/simple/
```

# CPU cores

## linux

- [Linux查看物理CPU个数、核数、逻辑CPU个数](https://www.cnblogs.com/emanlee/p/3587571.html)

```shell
# 总核数 = 物理CPU个数 X 每颗物理CPU的核数 
# 总逻辑CPU数 = 物理CPU个数 X 每颗物理CPU的核数 X 超线程数

# 查看物理CPU个数
cat /proc/cpuinfo | grep "physical id" | sort | uniq | wc -l

# 查看每个物理CPU中core的个数(即核数)
cat /proc/cpuinfo | grep "cpu cores" | uniq

# 查看逻辑CPU的个数
cat /proc/cpuinfo | grep "processor" | wc -l
```

## python

- [python 查看cpu的核数](https://blog.csdn.net/m0_37360684/article/details/104048542)

```python
from multiprocessing import cpu_count

print(cpu_count())
```



# References

1. [AlexNet implementation + weights in TensorFlow](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/)
2. [mikechen66/AlexNet_TensorFlow2.0-2.2](https://github.com/mikechen66/AlexNet_TensorFlow2.0-2.2)
3. [Finding source of NaN in forward pass](https://discuss.pytorch.org/t/finding-source-of-nan-in-forward-pass/51153)
4. [torch.autograd.detect_anomaly](https://pytorch.org/docs/1.4.0/autograd.html#torch.autograd.detect_anomaly)
5. [deep-cross-modal-hashing/torchcmh/dataset/base/triplet.py](https://github.com/WangGodder/deep-cross-modal-hashing/blob/master/torchcmh/dataset/base/triplet.py)
6. [【BUG】[nltk_data] Error loading punkt: ＜urlopen error [Errno 11004] [nltk_data]](https://blog.csdn.net/xiangduixuexi/article/details/108601873)
7. [Python 修改 pip 源为国内源](https://www.cnblogs.com/lsgxeva/p/12978981.html)
8. [python - pip换源，更换pip源到国内镜像](https://blog.csdn.net/xuezhangjun0121/article/details/81664260)
9. [521xueweihan/GitHub520](https://github.com/521xueweihan/GitHub520)