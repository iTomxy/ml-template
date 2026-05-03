@echo off
@REM https://stackoverflow.com/questions/37987839/how-can-i-run-tensorboard-on-a-remote-server
@REM https://stackoverflow.com/questions/29936948/ssh-l-forward-multiple-ports
@REM Usage
@REM   1. modify parameters of this script
@REM   2. open with local browser, e.g. `localhost:16006` for tensorboard

@REM remote host where tensorboard/jupyter is launched
set USER=itom
set IP=1.2.3.4
@REM remote tensorboard port
set TB_PORT=6006
@REM remote jupyter notebook port
set JN_PORT=8888

@REM local port prefix to avoid conflict with local tensorboard/jupyter
set PORT_PREF=1

ssh -L %PORT_PREF%%TB_PORT%:127.0.0.1:%TB_PORT% ^
    -L %PORT_PREF%%JN_PORT%:127.0.0.1:%JN_PORT% ^
    %USER%@%IP%
