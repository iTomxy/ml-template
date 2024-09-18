@echo off
setlocal enabledelayedexpansion


@REM Monitor jobs actively by fetching finish marker file.
@REM
@REM Suppose your time-consuming code will produce an `END.txt` on finish,
@REM and the URL of it will be `tom@1.2.3.4:/home/tom/project1/log/exp2/END.txt`.
@REM Then this script will try to download this `END.txt` to judge whether this job is done.
@REM
@REM To do so, create a text file with `.monitor` extension in the same folder as this script,
@REM write the URL in one line in this text file and save, and run this script.
@REM This script will invoke a message box on successful detecting `END.txt`.
@REM
@REM This script depends on `pscp` (provided along with putty.exe).
@REM One have to put the generated putty public ssh key to target server to support it.
@REM
@REM A better solution is to send an email at the end of the job code automatically!


set KEY=%USERPROFILE%\.ssh\putty-pri.ppk

:loop
for %%f in (*.monitor) do (
	set "fns=%%~nf"
	for /f "delims=" %%a in (%%f) do (
		set "sn=_!fns!_%%~nxa"
		pscp -i %KEY% "%%a" "!sn!"
		if exist "!sn!" (
			msg * [%date% %time%] "cmd" done: %%a
			del "%%f"
			del "!sn!"
		)
	)
)

for %%f in (*.monitor) do (
	echo sleep @ %date% %time%
	timeout /t 900 /nobreak
	goto :loop
)

msg * [%date% %time%] All monitored jobs done.
