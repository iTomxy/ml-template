@echo off

if "%1" NEQ "" (
	powershell -ExecutionPolicy Bypass -File D:\bin\msg.ps1 -Message "%*"
)

