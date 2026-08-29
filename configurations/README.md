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

The two are cured by two independent splits of the data directory.
opencode already keys session *storage* by project (`project/<id>/storage/`),
but `opencode.db`, `log/` and `bin/` are shared by every project and every node:
- splitting the data directory *per project* removes the cross-project contention [10,11];
- redirecting it to a node-local folder (`/tmp` is local to each node) removes the NFS damage [9].

The per-project split is the more useful of the two, and the one described below.
Every project gets its own data directory, at the project's own absolute path
mirrored under `~/.local/share/opencode/`, and only `auth.json` is shared:

```
~/.local/share/opencode/
├── auth.json                    # the single login, shared by every project
├── home/tom/codes/pointcept/    # the project /home/tom/codes/pointcept/
│   └── opencode/                # its data directory
│       ├── auth.json            # -> the shared auth.json above
│       ├── opencode.db
│       ├── log/
│       ├── bin/
│       └── project/
└── project/open-points/
    └── opencode/
        └── ...
```

```shell
# ~/.bashrc
opencode() {
    local share="$HOME/.local/share/opencode"  # holds auth.json and the mirrored tree
    local base="$share"                        # root of the mirrored tree
    # on a cluster with an NFS home, keep the DBs off NFS instead:
    # base="/tmp/opencode-$(whoami)-$(hostname)"

    # the project: the git root if there is one, else the current directory
    local proj; proj="$(git rev-parse --show-toplevel 2>/dev/null || pwd -P)"
    local dir="$base$proj/opencode"            # $proj is absolute, hence no separator

    mkdir -p "$share" "$dir"
    # a *real* (non-symlink) auth.json here means a login was written into this project:
    # push it back to the shared one, but only if it is newer
    # (`-nt` is also true when the shared one does not exist yet)
    if [ -f "$dir/auth.json" ] && [ ! -L "$dir/auth.json" ] \
       && [ "$dir/auth.json" -nt "$share/auth.json" ]; then
        mv -f "$dir/auth.json" "$share/auth.json"
    fi
    ln -sfn "$share/auth.json" "$dir/auth.json"

    XDG_DATA_HOME="$base$proj" "$HOME/.opencode/bin/opencode" --auto "$@"
}
```
- `XDG_DATA_HOME` is the *parent* of the data directory, opencode always appends `opencode/` to it.
  Mirroring the path makes that forced level useful rather than something to work around:
  it holds a project's files in a fixed leaf, so a repo *nested* inside another
  (a submodule, or `~/codes` being a repo of its own) lands in a sibling directory
  instead of intermixing with its parent's `opencode.db` and `log/`.
  Only a repo named `opencode` itself would collide.
- The project is the repo the current directory belongs to, and outside a repo the current
  directory itself: `git rev-parse` then writes nothing to stdout and fails, and `pwd -P` takes
  over, so git is used where it helps without being required. Both print a path with symlinks
  already resolved, hence no `realpath` and no way for one project to end up with two directories.
- Grouping by repo means a launch from a *subdirectory* shares the history of one from the repo
  root. The cost is the opposite case: everything under a `~` that happens to be a dotfiles repo
  collapses into a single directory. Use `proj="$(pwd -P)"` alone for a directory per directory.
- No name-length limit to worry about: every component keeps its original length,
  so only `PATH_MAX` (4096 for the whole path) applies, against `NAME_MAX` (255 bytes per component)
  for a flattened name. Nothing is escaped or rewritten either, so the mapping is exact both ways.
  In exchange, listing every project is `find "$share" -name opencode.db` rather than a plain `ls`.
- Run *without* the wrapper (`command opencode`, or from a shell that does not define it) and opencode
  writes `opencode.db`, `log/` and `project/` straight into `~/.local/share/opencode/`,
  i.e. beside the mirror, where its `project/` meets a mirrored `/project/...`.
  Set `base="$share/by-path"` to keep the mirror out of reach if that matters.
- `opencode auth login` works *through* the symlink: it writes into `~/.local/share/opencode/auth.json`,
  so there is no need to login outside the wrapper.
  A dangling symlink is fine for the very first login, the target is created on write.
- Two opencode in *different* projects now have separate DBs, so neither the `SQLITE_BUSY` death [10]
  nor the shared-session surprise [11] can happen between them, on one node or across nodes.
  Session history is per project and lives in the home directory,
  so it survives a `/tmp` clean and follows you from node to node.
- Two opencode in the *same* project still share `$dir/opencode.db`, so [10,11] are *not* covered there.
  `OPENCODE_DB=/tmp/opencode-$$.db` gives every process its own DB [11],
  at the price of losing all session history. It is undocumented, so check that the installed version
  honours it before relying on it: `OPENCODE_DB=/tmp/oc-test.db opencode run 'hi' && ls -l /tmp/oc-test.db`.
- The DBs are back on NFS, which is exactly what [9] is about: keeping history in the shared home means
  the *same* project opened on two nodes at once can corrupt its DB. Uncomment the node-local `base`
  to keep the per-project split and put the DBs on `/tmp` again; `auth.json` stays shared either way,
  history stops being shared across nodes.
- `bin/` (downloaded helpers such as ripgrep and LSP servers) is now per project, i.e. downloaded again
  for every new project. Share it like `auth.json` if that annoys:
  `mkdir -p "$share/bin"; ln -sfn "$share/bin" "$dir/bin"`.
- OAuth logins (Claude Pro/Max, GitHub Copilot) refresh their tokens by rewriting `auth.json`,
  so many instances running at once do race on that single file.
  Plain API-key providers can skip `auth.json` altogether by exporting e.g. `ANTHROPIC_API_KEY` in `~/.bashrc`.
- None of this is officially supported: [9] is open and unanswered, [10] was closed as *not planned*,
  and `OPENCODE_DATA_DIR`, the sanctioned way to move the data directory, is still an unmerged PR [12].
  Mind that opencode updates itself by default: should a later build stop honouring `XDG_DATA_HOME`,
  it will not complain, it will quietly fall back to the single DB in the home directory.

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
