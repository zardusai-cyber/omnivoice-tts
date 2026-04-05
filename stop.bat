@echo off
taskkill /f /im python.exe /fi "WINDOWTITLE eq OmniVoice*"
taskkill /f /im python.exe /fi "WINDOWTITLE eq *server.py*"
echo OmniVoice server stopped.
pause
