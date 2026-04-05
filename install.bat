@echo off
echo ============================================
echo OmniVoice TTS Server - Installation Script
echo ============================================
echo.

cd /d "%~dp0"

echo [1/4] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

echo.
echo [2/4] Activating virtual environment...
call "%~dp0venv\Scripts\activate.bat"

echo.
echo [3/6] Installing PyTorch XPU nightly...
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/xpu

echo.
echo [4/6] Installing dependencies and OmniVoice...
pip install soundfile scipy transformers accelerate safetensors gradio omegaconf einops
pip install git+https://github.com/k2-fsa/OmniVoice.git

echo.
echo [5/6] Reinstalling XPU torch and torchaudio (fixes dependency conflict)...
pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu --force-reinstall

echo.
echo ============================================
echo Installation complete!
echo.
echo To start the server, run: start.bat
echo ============================================
pause
