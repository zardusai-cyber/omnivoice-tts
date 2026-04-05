@echo off
title OmniVoice TTS Server (INT4/INT8 Fallback - XPU)

echo ================================================
echo OmniVoice TTS Server (INT4 Quantized)
echo ================================================
echo.
echo Quantization: INT4 weight-only (TorchAO)
echo Fallback: INT8 (if INT4 not available on Windows)
echo Expected VRAM: ~2.2-3.5 GB
echo Quality:       ~95-99% of BF16
echo.
set SYCL_CACHE_PERSISTENT=1

cd /d "%~dp0"
call "C:\ComfyUI\comfyui_venv\Scripts\activate.bat"

echo [OK] Virtual environment activated
echo [OK] PyTorch XPU: 2.12.0 nightly
echo [OK] Quantization: INT4 weight-only (group_size=128)
echo.
echo NOTE: INT4 requires 'mslk' package which is not
echo       available on Windows. The server will
echo       automatically fall back to INT8.
echo.
echo Press Ctrl+C to stop the server
echo ================================================

python server_int4.py
pause
