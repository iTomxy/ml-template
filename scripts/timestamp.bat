@echo off


@REM Method 1: wmic (built-in from Windows 7)
for /f %%a in ('wmic os get LocalDateTime ^| find "."') do set dt=%%a
set year=%dt:~0,4%
set month=%dt:~4,2%
set day=%dt:~6,2%
set hour=%dt:~8,2%
set minute=%dt:~10,2%
set second=%dt:~12,2%


@REM Method 2: use powershell
for /f "tokens=1-6" %%a in ('powershell -Command "Get-Date -Format 'yyyy MM dd HH mm ss'"') do (
    set year=%%a
    set month=%%b
    set day=%%c
    set hour=%%d
    set minute=%%e
    set second=%%f
)
