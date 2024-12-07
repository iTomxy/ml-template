@echo off
setlocal enabledelayedexpansion

@REM
@REM Execute command & check return code, alert on error.
@REM %1: command to be executed
@REM %2: (optional) file name of calling script (%~nx0)
@REM

@REM use `%~1' instead of `%1' to remove enclosing quote
set "cmd=%~1"
@REM convert embedded double-quote to quote (e.g. those used to wrap path)
set "cmd=%cmd:""="%"
@REM echo %cmd%

@REM execute command
%cmd%
@REM cache return code
set ret=%errorlevel%
@REM echo %ret%

@REM alert on error
if %ret% NEQ 0 (
    @REM show calling script name if available
    if NOT "%~2" == "" (
        set "errmsg=[%date%, %time%] Error(%ret%): %cmd% @ %~2"
    ) else (
        set "errmsg=[%date%, %time%] Error(%ret%): %cmd%"
    )
    @REM echo !errmsg!

    if exist "%SystemRoot%\System32\msg.exe" (
        msg * !errmsg!
    ) else (
        if "%~2" == "" (
            for /F "delims=" %%f in ("%0") do set fn=%%~nf
        ) else (
            for /F "delims=" %%f in ("%~2") do set fn=%%~nf
        )
        set "logf=%USERPROFILE%\__err__.!fn!.txt"
        echo !errmsg! > "!logf!"
        notepad "!logf!"
        del "!logf!" > nul 2>&1
    )
)

@REM return the return code of the executed command
exit /b %ret%
