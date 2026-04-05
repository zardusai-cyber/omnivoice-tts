@echo off
echo ========================================
echo OmniVoice TTS API Server
echo ========================================
echo.

call "%~dp0venv\Scripts\activate.bat"
cd /d "%~dp0"

echo [1/3] Activating virtual environment...
echo [OK] Virtual environment activated

echo [2/3] Checking dependencies...
echo [OK] Dependencies ready

echo [3/3] Starting OmniVoice API Server...
echo ========================================
echo Server Endpoints:
echo ========================================
echo   Web Interface:  http://localhost:8000
echo   API Docs:       http://localhost:8000/docs
echo   Health Check:   http://localhost:8000/health
echo.
echo OpenAI-Compatible Endpoints:
echo   POST /v1/audio/speech      - Text to Speech
echo   GET  /v1/audio/voices      - List Voices
echo   POST /v1/chat/completions  - Voice Chat
echo ========================================
echo.
echo Press Ctrl+C to stop the server
echo ========================================

python api_server.py
pause
