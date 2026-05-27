Syncthing [1] is a P2P synchronising software.
Compared to [Nutstore](https://www.jianguoyun.com/),
it is:
- free of charge
- no upload / download data limit:
not depend on a central server,
only need to run it on computers that need to synchronise.
- support custom filtering rule with [.stignore](stignore) file [4]

This can be useful when synchronising a folder with frequently updated contents
(e.g. projects progress managing folder)
but also want to ignore some folders/files from synchronising
(e.g. .git/, auxiliary/temporary/log files).

# Usage

See [1]:

1. Install and start Syncthing on all computers to synchronise across,
e.g. the one at school/company and one at home.

2. Link other devices with device ID on each device.

3. Add folders to sync on one device,
then accept the synchronisation on other devices.

# Auto-start

See [3] for how to let Windows launch Syncthing on start automatically without manual operation,
using the Windows built-in Task Scheduler.
Note that the `--no-console` mentioned in [3] does NOT really let Syncthing run in background.
A concole window will still be launched, although minimised,
closing which will terminate Syncthing.
To really let it run in background,
run [syncthing_bg.vbs](syncthing_bg.vbs) with *C:\Windows\System32\wscript.exe*
in the Action step in [3].

Run `syncthing.exe browser` to open the control website.

# Ignoring

Use a `.stignore` file to exclude some folders and/or files from synchronising,
just like the `.gitignore` for git.
Example [.stignore](stignore).
One can also refer [.gitignore](../git/gitignore) for items to ignore.

The ignoring rules can also be edit on the website/GUI within the sync folder property editting.


# References

1. [syncthing](https://github.com/syncthing/syncthing)
2. [Getting Started](https://docs.syncthing.net/intro/getting-started.html)
3. [Starting Syncthing Automatically](https://docs.syncthing.net/users/autostart.html)
4. [Ignoring Files](https://docs.syncthing.net/users/ignoring.html)
