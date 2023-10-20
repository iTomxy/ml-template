@echo off
setlocal enabledelayedexpansion
@REM https://blog.csdn.net/HackerTom/article/details/130260546

@REM servers
set servers[0]=QUIT
set servers[1]=itom@1.2.3.4
set servers[2]=jtom@5.6.7.8
set servers[3]=ktom@9.10.11.12

:connect
@REM show servers' id, user & ip
for /l %%n in (0,1,3) do (
   echo [%%n] !servers[%%n]!
)

@REM prompt, input server id & connect
set /p "sid=which: "
if defined servers[%sid%] (
	if %sid% EQU 0 (
		goto :eof
	) else (
		ssh !servers[%sid%]!
		@REM cls
		for /l %%i in (0,1,7) do echo.
		goto :connect
	)
) else (
	echo No such server: %sid%
	goto :connect
)
