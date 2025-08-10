@echo off

set IN=main.tex
set OUT=combined.tex

latexpand -o %OUT% --out-encoding 'encoding(UTF-8)' %IN%
