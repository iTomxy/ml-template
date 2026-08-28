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
    "*": "allow"
}
```

Running several opencode instances against one SQLite DB (`opencode.db`) breaks in two distinct ways [9-12]:
- On *any* filesystem, local disk included: the DB is opened with `PRAGMA busy_timeout = 0`,
  so a write that meets a lock fails at once instead of retrying
  and the session dies silently mid-turn on `SQLITE_BUSY` [10].
  Instances in the same project also step on each other's session state [11].
- On an NFS-mounted home (e.g. the UTS iHPC cluster), additionally: SQLite locks through `fcntl()`,
  which is unreliable over NFS, and WAL mode needs a shared-memory `-shm` mapping that does not work
  across hosts. The DB is then not merely contended but *corrupted*
  (`database disk image is malformed`, alongside stale `.nfs*` handles) [9].

Only the second one is cured by redirecting the data directory to a node-specific folder
(`/tmp` is local to each node, so each node gets a DB of its own):
```shell
# on a node of the server cluster
export XDG_DATA_HOME=/tmp/opencode-$(whoami)-$(hostname)
mkdir -p "$XDG_DATA_HOME"
opencode
```
This moves the *whole* data directory `$XDG_DATA_HOME/opencode/`, which also holds the credentials `auth.json`,
so every node ends up asking for its own login.
To login once (e.g. on the login node) and reuse it everywhere,
keep `auth.json` in the NFS-shared home and symlink it into the node-local data directory:
```shell
# ~/.bashrc
opencode() {
    local share="$HOME/.local/share/opencode"        # NFS-shared, holds auth.json
    local xdg="/tmp/opencode-$(whoami)-$(hostname)"  # node-local, holds opencode.db
    mkdir -p "$share" "$xdg/opencode"
    # a *real* (non-symlink) auth.json here means a login was written node-locally:
    # push it back to the shared home, but only if it is newer than the shared one
    # (`-nt` is also true when the shared one does not exist yet)
    if [ -f "$xdg/opencode/auth.json" ] && [ ! -L "$xdg/opencode/auth.json" ] \
       && [ "$xdg/opencode/auth.json" -nt "$share/auth.json" ]; then
        mv -f "$xdg/opencode/auth.json" "$share/auth.json"
    fi
    ln -sfn "$share/auth.json" "$xdg/opencode/auth.json"
    XDG_DATA_HOME="$xdg" "$HOME/.opencode/bin/opencode" --auto "$@"
}
```
- `opencode auth login` now works *through* the wrapper: it writes through the symlink into
  `~/.local/share/opencode/auth.json`, so the previous work-arounds (login before adding the function,
  or `command opencode auth login`) are no longer needed.
  A dangling symlink is fine for the very first login, the target is created on write.
- Only `auth.json` is shared. `opencode.db`, `log/` and `project/*/storage/` stay under `/tmp`,
  i.e. per node, so session history is *not* shared across nodes and is lost when `/tmp` is cleaned.
- `bin/` (downloaded helpers such as ripgrep and LSP servers) is per node as well and re-downloaded
  after each `/tmp` clean; symlink it the same way if that annoys:
  `ln -sfn "$share/bin" "$xdg/opencode/bin"`.
- OAuth logins (Claude Pro/Max, GitHub Copilot) refresh their tokens by rewriting `auth.json`,
  so many nodes running at once do race on that single file.
  Plain API-key providers can skip `auth.json` altogether by exporting e.g. `ANTHROPIC_API_KEY` in `~/.bashrc`.
- Two opencode on the *same* node still share `/tmp/opencode-$(whoami)-$(hostname)/opencode/opencode.db`,
  so the `SQLITE_BUSY` death [10] is *not* covered. `OPENCODE_DB=/tmp/opencode-$$.db` gives every process
  its own DB [11], at the price of losing all session history.
  It is undocumented, so check that the installed version honours it before relying on it:
  `OPENCODE_DB=/tmp/oc-test.db opencode run 'hi' && ls -l /tmp/oc-test.db`.
- None of this is officially supported: [9] is open and unanswered, [10] was closed as *not planned*,
  and `OPENCODE_DATA_DIR`, the sanctioned way to move the data directory, is still an unmerged PR [12].
  Mind that opencode updates itself by default: should a later build stop honouring these variables,
  it will not complain, it will quietly fall back to the DB on NFS.

# References

1. [Python 修改 pip 源为国内源](https://www.cnblogs.com/lsgxeva/p/12978981.html)
2. [python - pip换源，更换pip源到国内镜像](https://blog.csdn.net/xuezhangjun0121/article/details/81664260)
3. [521xueweihan/GitHub520](https://github.com/521xueweihan/GitHub520)
4. [解决pip安装超时的问题](https://blog.csdn.net/qq_39161804/article/details/81191977)
5. [screen](https://zhuanlan.zhihu.com/p/592016896)
6. [亲测！screen好看好用的配置（Linux）](https://www.jianshu.com/p/89607ef31493)
7. [ipaddress.com](https://www.ipaddress.com/)
8. [chrisant996/clink](https://github.com/chrisant996/clink)
9. [SQLite database corruption (database disk image is malformed) when running concurrent sessions on NFS #14970](https://github.com/anomalyco/opencode/issues/14970)
10. [opencode run: concurrent sessions crash with SQLITE_BUSY (busy_timeout=0) #21215](https://github.com/anomalyco/opencode/issues/21215)
11. [Multiple opencode instances in the same project share the same session via SQLite database #31307](https://github.com/anomalyco/opencode/issues/31307)
12. [feat: add OPENCODE_DATA_DIR and friends for portable mode #8963](https://github.com/anomalyco/opencode/pull/8963)
