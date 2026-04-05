@echo off
title OmniVoice TTS API Server (INT8 Quantized - XPU)

echo ================================================
echo OmniVoice TTS API Server (INT8 Quantized)
echo ================================================

set SYCL_CACHE_PERSISTENT=1

cd /d "%~dp0"
call "C:\ComfyUI\comfyui_venv\Scripts\activate.bat"

echo [OK] Virtual environment activated
echo [OK] PyTorch XPU: 2.12.0 nightly
echo.
echo Server Endpoints:
echo   Web Interface:  http://localhost:8000
echo   API Docs:       http://localhost:8000/docs
echo   Health Check:   http://localhost:8000/health
echo.
echo Quantization: INT8 weight-only (TorchAO)
echo Backend:      PyTorch XPU
echo.
echo Press Ctrl+C to stop the server
echo ================================================

python api_server_int8.py
pause
