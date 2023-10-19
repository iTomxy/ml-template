@echo off
setlocal enabledelayedexpansion
@REM https://blog.csdn.net/HackerTom/article/details/130260546

@REM servers
set servers[0]=itom@1.2.3.4
set servers[1]=tomsss@5.6.7.8
set servers[2]=tomascat@9.10.11.12
set servers[3]=tommy@13.14.15.16

@REM show servers' id, user & ip
for /l %%n in (0,1,3) do (
   echo [%%n] !servers[%%n]!
)

:connect
@REM prompt, input server id & connect
set /p "sid=which: "
if defined servers[%sid%] (
	ssh !servers[%sid%]!
    @REM goto :connect
) else (
	echo No such server: %sid%
	goto :connect
)
