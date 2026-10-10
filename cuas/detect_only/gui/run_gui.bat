@echo off
chcp 65001 > nul
cd /d "%~dp0"
set PY=python
if exist "%USERPROFILE%\anaconda3\python.exe" set PY="%USERPROFILE%\anaconda3\python.exe"
%PY% -X utf8 detect_gui.py %*
pause
