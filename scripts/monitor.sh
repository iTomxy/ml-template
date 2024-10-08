# monitor command on signals
# https://www.linuxjournal.com/content/bash-trap-command
# https://phoenixnap.com/kb/bash-trap-command

# on error
trap 'echo \[`date`\] `whoami`@`hostname`:`realpath $0` | mail -s "cmd error" tom@tomix.org' ERR
# on interrupted/killed
trap 'echo \[`date`\] `whoami`@`hostname`:`realpath $0` | mail -s "cmd interrupted" tom@tomix.org' INT TERM HUP
# on exit
trap 'echo \[`date`\] `whoami`@`hostname`:`realpath $0` | mail -s "cmd done" tom@tomix.org' EXIT
