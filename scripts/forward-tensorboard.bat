@echo off
@REM https://stackoverflow.com/questions/37987839/how-can-i-run-tensorboard-on-a-remote-server

set USER=itom
set IP=1.2.3.4
set REMOTE_PORT=6006
set LOCAL_PORT=16006
ssh -L %LOCAL_PORT%:127.0.0.1:%REMOTE_PORT% %USER%@%IP%

