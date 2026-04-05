# OmniVoice TTS Server

OmniVoice TTS server with **Intel Arc iGPU / dGPU** support via PyTorch nightly XPU. Includes multiple quantization variants (BF16, INT8, INT4) for different VRAM budgets.

## Features

- **OpenAI-compatible API** - Drop-in replacement for OpenAI TTS
- **Intel XPU acceleration** - Optimized for Intel Arc GPUs via `torch.xpu`
- **Multiple precision modes** - BF16, INT8, INT4 for different VRAM/performance tradeoffs
- **torch.compile support** - Optimized inference with kernel compilation caching
- **Voice cloning** - Use any WAV/MP3/FLAC reference audio (15+ seconds)
- **80+ pre-trained voices** included
- **Dynamic voice reload** - Add voices without restarting via `POST /v1/audio/voices/reload`

## Prerequisites

- **Windows 10/11** with Intel Arc GPU (or compatible Intel GPU)
- **Python 3.11+**
- **PyTorch nightly with XPU support**
- **Intel oneAPI BaseKit** (for XPU runtime)
- **ffmpeg** (for voice companion tools)

## Quick Start

### 1. Install Dependencies

```bat
install.bat
```

This creates a virtual environment and installs all required packages including PyTorch XPU nightly.

### 2. Download the Model

Download the BF16 model weights from Hugging Face:

👉 **https://huggingface.co/drbaph/OmniVoice-bf16**

You only need this one. The INT8 and INT4 quantized versions are generated locally from these weights using the provided quantization scripts.

After downloading, place the weights in the appropriate folder:

**BF16 (full precision, ~6GB VRAM):**
```
OmniVoice/model.safetensors
OmniVoice/audio_tokenizer/model.safetensors
```

**INT8 (quantized, ~3.5GB VRAM):**
```
OmniVoice_INT8/model.safetensors
OmniVoice_INT8/audio_tokenizer/model.safetensors
```
Run `quantize_model.py` to create INT8 weights from BF16.

**INT4 (max quantized, ~2.2GB VRAM):**
```
OmniVoice_INT4/model.safetensors
OmniVoice_INT4/audio_tokenizer/model.safetensors
```
Run `quantize_model_int4.py` to create INT4 weights from BF16.

### 3. Start the Server

Choose the variant that fits your VRAM:

| Script | Precision | VRAM | Speed | Quality |
|--------|-----------|------|-------|---------|
| `start_api.bat` | BF16 | ~6 GB | Fast | Best |
| `start_api_int8.bat` | INT8 | ~3.5 GB | Faster | ~98% |
| `start_api_int8_compile.bat` | INT8 + compile | ~3.5 GB | Fastest | ~98% |
| `start_api_int4.bat` | INT4 | ~2.2 GB | Medium | ~95% |

```bat
start_api_int8_compile.bat
```

Server starts at `http://localhost:8000`

## API Endpoints

### Text to Speech
```
POST /v1/audio/speech
Content-Type: application/json

{
  "model": "omnivoice",
  "input": "Hello, this is a test.",
  "voice": "david-attenborough-original",
  "speed": 1.0
}
```

### List Voices
```
GET /v1/audio/voices
```

### Reload Voices (no restart needed)
```
POST /v1/audio/voices/reload
```

### Health Check
```
GET /health
```

### API Docs
```
GET /docs
```

## Adding Custom Voices

1. Place a WAV/MP3/FLAC file (15+ seconds of clean speech) in the `voices/` folder
2. Name it descriptively (e.g., `my-voice.wav`)
3. Call `POST /v1/audio/voices/reload` or restart the server

## Server Variants

### `api_server.py` (BF16)
- Full precision model
- Best quality
- Highest VRAM usage (~6GB)
- Uses `OmniVoice/` model folder

### `api_server_int8.py` (INT8)
- INT8 weight-only quantization via TorchAO
- ~40% VRAM savings
- ~98% quality retention
- Uses `OmniVoice_INT8/` model folder

### `api_server_int8_compile.py` (INT8 + torch.compile)
- INT8 quantization + torch.compile with reduce-overhead mode
- Compilation caching for fast startup after first run
- Best inference speed
- Uses `OmniVoice_INT8/` model folder

### `api_server_int4.py` (INT4)
- INT4 weight-only quantization (group_size=128)
- ~63% VRAM savings
- ~95% quality retention
- Uses `OmniVoice_INT4/` model folder

## Quantization

To create quantized models from the BF16 weights:

```bat
quantize_model.py        # Creates INT8 weights
quantize_model_int4.py   # Creates INT4 weights
```

## Project Structure

```
omnivoice-tts/
├── api_server.py                 # BF16 API server
├── api_server_int8.py            # INT8 API server
├── api_server_int8_compile.py    # INT8 + torch.compile API server
├── api_server_int4.py            # INT4 API server
├── server.py                     # BF16 Gradio server
├── server_int8.py                # INT8 Gradio server
├── server_int8_compile.py        # INT8 + compile Gradio server
├── server_int4.py                # INT4 Gradio server
├── quantize_model.py             # INT8 quantization script
├── quantize_model_int4.py        # INT4 quantization script
├── install.bat                   # Install dependencies
├── start_api*.bat                # Start API servers
├── start_int*.bat                # Start Gradio servers
├── stop.bat                      # Stop all servers
├── update.bat                    # Update dependencies
├── OmniVoice/                    # BF16 model configs (weights not included)
├── OmniVoice_INT8/               # INT8 model configs
├── OmniVoice_INT4/               # INT4 model configs (generated)
└── voices/                       # Voice reference audio
```

## License

MIT
