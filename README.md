# ml-template

My personal code snippet library of helper functions and classes, some with example.

- [+itom/](+itom): Tool functions that might be used in MATLAB codes.
The leading `+` will add this folder into the searching path.

- [configurations/](configurations): Configuration files.

- [containers/](containers): Definition files of Docker and Singularity image.

- [git/](git): git-related command showcases and *.gitignore* template.

- [scripts/](scripts): Utility scripts.

- [utils/](utils): Utilities.

- [args.py](args.py): An example of using the `argparse` package in Python.

- [config.yaml](config.yaml): An example of writing arguments in YAML.
Can be used together with [utils/config.py](utils/config.py).

- [requirements.txt](requirements.txt): Common packages I met that a Docker image may not contain.

# badge ![badge](https://img.shields.io/badge/badge-purple)

Sometimes you may want to create a badge in your GitHub repository README.
Use [Shields.io](https://github.com/badges/shields) to make one.
See [Static Badge](https://shields.io/badges/static-badge) for basic usage.
In markdown, you make one by inserting a:
```md
![SOME_ALTERNATIVE_TEXT](https://img.shields.io/badge/:badgeContent)
```
where `:badgeContent` has to be replaced by some fields.
For example,
a badge of this repository can be ![iTomxy/ml-template](https://img.shields.io/badge/iTomxy-ml--template-blue?logo=github&link=https%3A%2F%2Fgithub.com%2FiTomxy%2Fml-template):
```md
![iTomxy/ml-template](https://img.shields.io/badge/iTomxy-ml--template-blue?logo=github)
```
where:
- `iTomxy-ml--template-blue` is a `-`-separated string,
containing 3 fields: 1) left part text, 2) right part text, and 3) right part background colour.
- `logo` specifies the logo shown in the left part. The logo name should be from [Simple Icons](https://simpleicons.org/).

If you also want to add a hyperlink on the badge instead of letting it be a pure icon,
you just wrap it with `[badge-code](link)` [![iTomxy/ml-template](https://img.shields.io/badge/iTomxy-ml--template-blue?logo=github)](https://github.com/iTomxy/ml-template):
```md
[![iTomxy/ml-template](https://img.shields.io/badge/iTomxy-ml--template-blue?logo=github)](https://github.com/iTomxy/ml-template)
```
Althoug the Shilds.io badge supports inserting link by itself
(see [Static Badge](https://shields.io/badges/static-badge)),
but it seems that this is not usable in GitHub
(see [How to specify the link of left and right on GitHub #5593](https://github.com/badges/shields/discussions/5593)).


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
10. [解决pip安装超时的问题](https://blog.csdn.net/qq_39161804/article/details/81191977)
11. [screen](https://zhuanlan.zhihu.com/p/592016896)
12. [亲测！screen好看好用的配置（Linux）](https://www.jianshu.com/p/89607ef31493)
13. [How can I run Tensorboard on a remote server?](https://stackoverflow.com/questions/37987839/how-can-i-run-tensorboard-on-a-remote-server)
