@echo off
rem Starts the web app. Double-click this file.
cd /d "%~dp0"
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat
python web_app.py
pause
