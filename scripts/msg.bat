@echo off

@REM launch a message box (using powershell). Alternative of the `msg.exe` on Windows 10 Pro.

set PS_MSG=D:\bin\msg.ps1

if "%1" NEQ "" (
	powershell -ExecutionPolicy Bypass -File "%PS_MSG%" -Message "%*"
)
