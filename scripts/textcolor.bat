@echo off
echo Text Styles with ANSI Escape Codes ^(^>= Windows 10^)

for /f %%a in ('echo prompt $E^| cmd') do set "ESC=%%a"

echo %ESC%[0m reset %ESC%[0m
echo %ESC%[1m bold %ESC%[0m
echo %ESC%[4m underline %ESC%[0m
echo %ESC%[7m invert FG ^& BG %ESC%[0m
echo.
echo %ESC%[30m blacK %ESC%[0m, %ESC%[90m strong blacK %ESC%[0m
echo %ESC%[31m Red %ESC%[0m, %ESC%[91m strong Red %ESC%[0m
echo %ESC%[32m Green %ESC%[0m, %ESC%[92m strong Green %ESC%[0m
echo %ESC%[33m Yellow %ESC%[0m, %ESC%[93m strong Yellow %ESC%[0m
echo %ESC%[34m Blue %ESC%[0m, %ESC%[94m strong Blue %ESC%[0m
echo %ESC%[35m magenta %ESC%[0m, %ESC%[95m strong magenta %ESC%[0m
echo %ESC%[36m cyan %ESC%[0m, %ESC%[96m strong cyan %ESC%[0m
echo %ESC%[37m White %ESC%[0m, %ESC%[97m strong White %ESC%[0m
echo %ESC%[39m reset foreground color %ESC%[0m
echo.
echo %ESC%[40m blacK %ESC%[0m, %ESC%[100m strong blacK %ESC%[0m
echo %ESC%[41m Red %ESC%[0m, %ESC%[101m strong Red %ESC%[0m
echo %ESC%[42m Green %ESC%[0m, %ESC%[102m strong Green %ESC%[0m
echo %ESC%[43m Yellow %ESC%[0m, %ESC%[103m strong Yellow %ESC%[0m
echo %ESC%[44m Blue %ESC%[0m, %ESC%[104m strong Blue %ESC%[0m
echo %ESC%[45m magenta %ESC%[0m, %ESC%[105m strong magenta %ESC%[0m
echo %ESC%[46m cyan %ESC%[0m, %ESC%[106m strong cyan %ESC%[0m
echo %ESC%[47m White %ESC%[0m, %ESC%[107m strong White %ESC%[0m
