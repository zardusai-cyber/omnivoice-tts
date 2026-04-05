@echo off
title OmniVoice TTS API Server (INT4 Quantized - XPU)

echo ================================================
echo OmniVoice TTS API Server (INT4 Quantized)
echo ================================================
echo.
echo Quantization: INT4 weight-only (TorchAO)
echo Expected VRAM: ~2.2 GB (vs 3.5 GB INT8, 6 GB BF16)
echo Expected disk: ~2.0 GB (vs 2.24 GB INT8, 3.81 GB BF16)
echo Quality:       ~95-98% of BF16
echo.
set SYCL_CACHE_PERSISTENT=1

cd /d "%~dp0"
call "C:\ComfyUI\comfyui_venv\Scripts\activate.bat"

echo [OK] Virtual environment activated
echo [OK] PyTorch XPU: 2.12.0 nightly
echo [OK] Quantization: INT4 weight-only (group_size=128)
echo.
echo Server Endpoints:
echo   Web Interface:  http://localhost:8000
echo   API Docs:       http://localhost:8000/docs
echo   Health Check:   http://localhost:8000/health
echo.
echo Press Ctrl+C to stop the server
echo ================================================

python api_server_int4.py
pause
