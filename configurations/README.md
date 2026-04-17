# vim

The vim colour theme files (e.g. *solarized.vim*) should be placed under *VIM_CONFIG_PATH/colors/*.
The standard VIM_CONFIG_PATH is `%USERPROFILE%\vimfiles\` on Windows and `~/.vim/` on Linux and MacOS.

# hosts

some entries to be added in *hosts* file for [GitHub](https://github.com/) accessing:

```
140.82.114.4	github.com
199.232.5.194	github.global.ssl.fastly.net
199.232.68.133	raw.githubusercontent.com
```

see [3] for automatic *hosts* updating.
Use [7] to check IP addresses.

# pip

change the source of `pip`, see [1,2].

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

# powershell

By default, the powershell is prohibited to run ps1 files
(with error saying `running scripts is disabled on this system`).
Run `Get-ExecutionPolicy -List` to see the current policy.
If you run powershell commands and/or scripts sometimes,
run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` to grant priviledge for the current user (you).

For one-time execution, use
`powershell -ExecutionPolicy Bypass -File <THE_SCRIPT>.ps1`
to bypass the policy setting.

If a script is blocked, run
`Unblock-File <THE_SCRIPT>.ps1`.


# References

1. [Python 修改 pip 源为国内源](https://www.cnblogs.com/lsgxeva/p/12978981.html)
2. [python - pip换源，更换pip源到国内镜像](https://blog.csdn.net/xuezhangjun0121/article/details/81664260)
3. [521xueweihan/GitHub520](https://github.com/521xueweihan/GitHub520)
4. [解决pip安装超时的问题](https://blog.csdn.net/qq_39161804/article/details/81191977)
5. [screen](https://zhuanlan.zhihu.com/p/592016896)
6. [亲测！screen好看好用的配置（Linux）](https://www.jianshu.com/p/89607ef31493)
7. [ipaddress.com](https://www.ipaddress.com/)
