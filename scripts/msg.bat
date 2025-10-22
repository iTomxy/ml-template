@echo off

@REM Method 1
@REM launch a message box with mshta.exe, more general than powershell and msg.exe
if "%1" NEQ "" (
    mshta vbscript:msgbox^("%*",0,"No Title"^)^(window.close^)
)


@REM Method 2
@REM launch a message box (using powershell). Alternative of the `msg.exe` on Windows 10 Pro.
if "%1" NEQ "" (
	powershell -ExecutionPolicy Bypass -File ".\msg.ps1" -Message "%*"
)


@REM Method 3
@REM launch via msg.exe, only available in limited distribution (e.g. Windows 10 Pro)
if "%1" NEQ "" (
	msg * %*
)
