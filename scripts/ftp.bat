@echo off
setlocal enabledelayedexpansion

set KEY=%USERPROFILE%\.ssh\putty-pri.ppk

set IP=1.2.3.4
set USER=tom

set HOME=/home/%USER%
set REMOTE=%HOME%/abc.txt

set "DESKTOP=%USERPROFILE%\Desktop"
set "LOCAL=%DESKTOP%"


echo fetch
call :fetch "%REMOTE%" "%LOCAL%"


goto :eof

:fetch
setlocal
	set "REMOTE=%~1"
	set "LOCAL=%~2"
	if not exist "%LOCAL%" md "%LOCAL%"
	pscp -i %KEY% %USER%@%IP%:%REMOTE% "%LOCAL%"
endlocal
exit /b

:send
setlocal
	set "LOCAL=%~1"
	set "REMOTE=%~2"
	pscp -i %KEY% "%LOCAL%" %USER%@%IP%:"%REMOTE%"
endlocal
exit /b
