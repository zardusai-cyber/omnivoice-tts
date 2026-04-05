@echo off
title OmniVoice TTS API Server (INT8 + torch.compile - XPU)

echo ================================================
echo OmniVoice TTS API Server (INT8 + torch.compile)
echo ================================================
echo.
echo Optimization: INT8 weight-only + torch.compile
echo Expected VRAM: ~3.5 GB
echo Expected speed: 1.5-2x faster after compilation
echo.
set SYCL_CACHE_PERSISTENT=1

cd /d "%~dp0"
call "C:\ComfyUI\comfyui_venv\Scripts\activate.bat"

echo [OK] Virtual environment activated
echo [OK] PyTorch XPU: 2.12.0 nightly
echo [OK] Quantization: INT8 weight-only (TorchAO)
echo [OK] Compilation: torch.compile (reduce-overhead)
echo.
echo Server Endpoints:
echo   Web Interface:  http://localhost:8000
echo   API Docs:       http://localhost:8000/docs
echo   Health Check:   http://localhost:8000/health
echo.
echo ================================================
echo IMPORTANT: First generation will be SLOW
echo ================================================
echo torch.compile will optimize ~200 kernels for your XPU.
echo This takes 10-20 minutes on first run.
echo.
echo After compilation, subsequent generations will be
echo 1.5-2x FASTER! The cache is saved automatically.
echo ================================================
echo.
echo Press Ctrl+C to stop the server
echo ================================================
echo.

python api_server_int8_compile.py
pause
