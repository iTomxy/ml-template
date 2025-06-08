@echo off
@REM fo.bat <- Fzf Open
setlocal enabledelayedexpansion

@REM move the the specified root directory %1 if provided.
if "%~1" NEQ "" (
	if exist !%~1!\NUL (
		@REM %1 is a folder
		cd /d "%~1"
	)

	if "%~2" NEQ "" (
		if "%~2" EQU "1" (
			@REM open folder %1
			explorer .
		) else if "%~2" EQU "2" (
			@REM search patterns in file %1
			type "%~1" | fzf
		)
		goto :eof
	)
)

fzf --preview "bat -n --color=always {}" ^
	--preview-window=right:50%%:wrap ^
	--walker=file,dir,hidden,follow ^
	--bind "alt-d:become(fo.bat {})" ^
	--bind "alt-v:execute(vim {})" ^
	--bind "ctrl-d:execute(fo.bat {} 1)" ^
	--bind "ctrl-p:change-preview-window(down|hidden|)" ^
	--bind "ctrl-r:reload(dir /b)" ^
	--bind "ctrl-s:execute(fo.bat {} 2)" ^
	--bind "ctrl-y:execute(echo {} | clip)" ^
	--bind "enter:execute(start {})"
