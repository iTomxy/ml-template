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

# cmd

To patch the Windows cmd.exe to be more powerful,
e.g. let it complete like bash that to the common prefix of multiple candidates,
install clink [8]: `winget install clink`.

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

# opencode

Run [opencode](https://opencode.ai/) with `--auto` to automatically approve permission requests that are not explicitly denied.
Can also configure `~/.config/opencode/opencode.jsonc` to achieve so (see [Config](https://opencode.ai/docs/config/)):
- add these fields to the file, not simplily overwriting the whole.
- `--auto` = `"*": "allow"` (see [CLI](https://opencode.ai/docs/cli/)), can be risky.
```json
{
    "*": "allow",
    "autoupdate": true
}
```

To avoid crush when running multiple opencode over NFS (e.g. the UTS iHPC cluster) due to shared SQLite DB,
one can redirect the data directory to a node-specific folder:
```shell
# on a node of the server cluster
export XDG_DATA_HOME=/tmp/opencode-$(whoami)-$(hostname)
mkdir -p "$XDG_DATA_HOME"
opencode
```
Or better, to restrict this only when opencode is called,
wrap `opencode` as a function in ~/.bashrc:
- Note: this function blocks you from executing `opencode auth login`.
  One can either
    1. login before adding this function to ~/.bashrc, or
    2. login with full path `~/.opencode/bin/opencode auth login` or by-passing shell function `command opencode auth login`.
```shell
# ~/.bashrc
opencode() { XDG_DATA_HOME=/tmp/opencode-$(whoami)-$(hostname) $HOME/.opencode/bin/opencode --auto "$@"; }
```

# References

1. [Python 修改 pip 源为国内源](https://www.cnblogs.com/lsgxeva/p/12978981.html)
2. [python - pip换源，更换pip源到国内镜像](https://blog.csdn.net/xuezhangjun0121/article/details/81664260)
3. [521xueweihan/GitHub520](https://github.com/521xueweihan/GitHub520)
4. [解决pip安装超时的问题](https://blog.csdn.net/qq_39161804/article/details/81191977)
5. [screen](https://zhuanlan.zhihu.com/p/592016896)
6. [亲测！screen好看好用的配置（Linux）](https://www.jianshu.com/p/89607ef31493)
7. [ipaddress.com](https://www.ipaddress.com/)
8. [chrisant996/clink](https://github.com/chrisant996/clink)
