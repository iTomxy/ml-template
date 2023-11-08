@echo off
@REM https://stackoverflow.com/questions/37987839/how-can-i-run-tensorboard-on-a-remote-server
@REM https://stackoverflow.com/questions/29936948/ssh-l-forward-multiple-ports

set USER=itom
set IP=1.2.3.4
@REM remote tensorboard port
set TB_PORT=6006
@REM remote jupyter notebook port
set JN_PORT=8888
ssh -L 1%TB_PORT%:127.0.0.1:%TB_PORT% ^
    -L 1%JN_PORT%:127.0.0.1:%JN_PORT% ^
    %USER%@%IP%
