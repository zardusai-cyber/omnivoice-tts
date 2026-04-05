@echo off
call "%~dp0venv\Scripts\activate.bat"
cd /d "%~dp0"
python server.py
pause
