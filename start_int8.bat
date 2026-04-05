@echo off
title OmniVoice TTS Server (INT8 Quantized - XPU)

echo ================================================
echo OmniVoice TTS Server (INT8 Quantized)
echo ================================================

set SYCL_CACHE_PERSISTENT=1

cd /d "%~dp0"
call "C:\ComfyUI\comfyui_venv\Scripts\activate.bat"

echo [OK] Virtual environment activated
echo [OK] PyTorch XPU: 2.12.0 nightly
echo [OK] Quantization: INT8 weight-only (TorchAO)
echo.
echo Press Ctrl+C to stop the server
echo ================================================

python server_int8.py
pause
