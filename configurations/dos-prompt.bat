@echo off

@REM Configurate the prompt of the current cmd.exe window.
@REM Use `prompt /?` to see how.
@REM Default prompt is `$P$G`.

prompt $P$_$G$S

@REM To set the default prompt (just like setting `$PS1` in .bashrc on linux),
@REM add and edit the `PROMPT` environment variable.
