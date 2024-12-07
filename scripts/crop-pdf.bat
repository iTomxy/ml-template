@echo off
@REM Depend on `pdfcrop.exe` provided in texlive. See: https://www.tug.org/texlive/
@REM Usage:
@REM     crop-pdf.bat foo.pdf

if "%~1" == "" (
	echo * No file specificed
	exit
)

for /f %%f in ("%~1") do (
	set "basename=%%~nf"
	set "p=%%~dpf"
)
REM echo %1
REM echo %basename%
REM echo %p%

@REM cropped copy saved as `<FILENAME>-crop.pdf`
pdfcrop "%~1"
@REM delect original pdf
del /q "%~1"
@REM rename `<FILENAME>-crop.pdf` (back) to <FILENAME>.pdf
rename "%p%%basename%-crop.pdf" "%basename%.pdf"
