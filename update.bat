@echo off
call "%~dp0venv\Scripts\activate.bat"
cd /d "%~dp0"
echo Updating OmniVoice...
pip install --upgrade git+https://github.com/k2-fsa/OmniVoice.git
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/xpu --upgrade
echo Update complete!
pause
