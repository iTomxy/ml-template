@REM https://blog.csdn.net/HackerTom/article/details/136127369
@echo off
if "%~1" == "" (
        set b=50
) else (
        set b=%~1
)
@REM echo %b%
powershell -executionpolicy bypass -File ps_brightness.ps1 %b%
