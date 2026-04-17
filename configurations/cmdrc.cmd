@REM cmdrc.cmd
@REM Similar to .bashrc for bash, let cmd.exe run these commands at starting.
@echo off

@REM Change code page to UTF-8
chcp 65001 >nul

@REM Configurate the prompt of the current cmd.exe window.
@REM Use `prompt /?` to see how.
@REM Default prompt is `$P$G`.
@REM To set the default prompt (just like setting `$PS1` in .bashrc on linux),
@REM add and edit the `PROMPT` environment variable.
prompt $P$_$G$S

@REM Alias
@REM append `$*` if the command accepts arguments
doskey clear=cls
doskey del=del /p $*
doskey erase=erase /p $*
doskey jlab=jupyter lab --ip=0.0.0.0 $*
doskey jlabn=jupyter lab --ip=0.0.0.0 --no-browser $*
doskey ls=dir /w $*
doskey ll=dir $*
doskey mlrun=matlab -nodesktop -nosplash -r $*
doskey psh=powershell.exe -ExecutionPolicy Bypass -File $*
doskey tb=tensorboard --bind_all --logdir $*
