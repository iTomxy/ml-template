@echo off

@REM set screen brightness
@REM https://blog.csdn.net/HackerTom/article/details/136127369

@REM powershell script to set
set PS_BN=ps_brightness.ps1

if "%~1" == "" (
        set b=50
) else (
        set b=%~1
)
@REM echo %b%
powershell -executionpolicy bypass -File "%PS_BN%" %b%
