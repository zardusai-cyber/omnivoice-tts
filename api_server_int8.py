"""
OmniVoice INT8 Quantized API Server (XPU + TorchAO + torch.compile)

Mirrors api_server.py but loads the INT8 quantized model from OmniVoice_INT8/
and applies torch.compile() for optimized inference.
"""

import os
import gc
import io
import json
import time
import numpy as np
from pathlib import Path
from typing import Optional
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import soundfile as sf

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from omnivoice import OmniVoice

BASE_DIR = Path(__file__).parent
MODEL_SRC = BASE_DIR / "OmniVoice"
MODEL_INT8 = BASE_DIR / "OmniVoice_INT8"
VOICES_DIR = BASE_DIR / "voices"

app = FastAPI(title="OmniVoice TTS Server (INT8)", version="1.0.0")


def apply_quantization(model):
    from torchao.quantization import quantize_, Int8WeightOnlyConfig

    print("[2/5] Applying INT8 weight-only quantization...")
    quantize_(model.llm, Int8WeightOnlyConfig())
    quantize_(model.audio_heads, Int8WeightOnlyConfig())
    print("[OK] Quantization applied")

    quantized_count = sum(
        1 for _, m in model.named_modules() if "Int8WeightOnly" in type(m).__name__
    )
    print(f"     Quantized modules: {quantized_count}")
    return model


def load_quantized_state(model, state_path):
    print(f"[3/5] Loading quantized state from {state_path}...")
    state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    print("[OK] Quantized state loaded")


def apply_compile(model):
    print("[4/5] Applying torch.compile() for kernel fusion...")
    model = torch.compile(model)
    print("[OK] torch.compile() applied (first run will be slow due to compilation)")
    return model


print("=" * 60)
print("OmniVoice TTS API Server (INT8 Quantized)")
print("=" * 60)
print(f"[1/5] Loading model structure from: {MODEL_SRC}")
model = OmniVoice.from_pretrained(
    str(MODEL_SRC), device_map="xpu", dtype=torch.bfloat16
)

model = apply_quantization(model)

state_path = MODEL_INT8 / "quantized_state.pt"
if state_path.exists():
    load_quantized_state(model, state_path)
else:
    print(
        "[WARNING] quantized_state.pt not found. Model is quantized but using random weights."
    )

model = apply_compile(model)

print("[5/5] Scanning voices...")
voices = {}
if VOICES_DIR.exists():
    for f in VOICES_DIR.iterdir():
        if f.suffix.lower() in [".wav", ".mp3", ".flac"]:
            name = f.stem.lower().replace("_", "-")
            voices[name] = str(f)
print(f"[OK] Found {len(voices)} voices")
print("[OK] Server ready")
print("=" * 60)


def process_audio(audio_tensor):
    audio_np = audio_tensor.cpu().numpy()
    if audio_np.ndim > 1:
        audio_np = audio_np.squeeze()
    audio_np = np.clip(audio_np, -1.0, 1.0)
    return audio_np


def normalize_voice_id(voice_id):
    if not voice_id:
        return None
    voice_id = voice_id.lower()
    for ext in [".wav", ".mp3", ".flac", ".ogg"]:
        if voice_id.endswith(ext):
            voice_id = voice_id[: -len(ext)]
            break
    return voice_id.replace("_", "-").replace(" ", "-")


def generate_tts(text, voice=None, speed=1.0, num_steps=32):
    normalized_voice = normalize_voice_id(voice)

    if normalized_voice and normalized_voice in voices:
        print(f"[INFO] Using voice: {normalized_voice}")
        audio = model.generate(
            text=text,
            ref_audio=voices[normalized_voice],
            num_step=num_steps,
            speed=speed,
        )
    elif voice:
        print(f"[WARNING] Voice '{voice}' not found. Using default.")
        audio = model.generate(text=text, num_step=num_steps, speed=speed)
    else:
        audio = model.generate(text=text, num_step=num_steps, speed=speed)
    return process_audio(audio[0])


class SpeechRequest(BaseModel):
    model: str = "omnivoice"
    input: str
    voice: Optional[str] = None
    response_format: str = "wav"
    speed: float = 1.0


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "omnivoice"
    messages: list
    voice: Optional[str] = None
    speed: float = 1.0


@app.get("/")
async def root():
    return {
        "message": "OmniVoice TTS Server (INT8 Quantized)",
        "docs": "/docs",
        "voices": "/v1/audio/voices",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": "int8_quantized", "voices": len(voices)}


@app.post("/v1/audio/voices/reload")
async def reload_voices():
    global voices
    voices = {}
    if VOICES_DIR.exists():
        for vf in VOICES_DIR.iterdir():
            if vf.suffix.lower() in [".wav", ".mp3", ".flac"]:
                voices[vf.stem.lower().replace("_", "-").replace(" ", "-")] = str(vf)
    return {"status": "ok", "voices": len(voices), "reloaded": True}


@app.get("/v1/audio/voices")
async def list_voices():
    voice_list = []
    for voice_id in sorted(voices.keys()):
        display_name = voice_id.replace("-", " ").title()
        voice_list.append(
            {
                "voice_id": voice_id,
                "name": display_name,
                "preview_url": f"/voices/{voice_id}/preview",
                "type": "custom",
            }
        )
    return {"voices": voice_list}


@app.post("/v1/audio/speech")
async def text_to_speech(req: SpeechRequest):
    try:
        audio = generate_tts(req.input, req.voice, req.speed)
        buffer = io.BytesIO()
        sf.write(buffer, audio, 24000, format="WAV")
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=speech.wav"},
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    try:
        text = ""
        for msg in req.messages:
            if msg.get("role") == "user":
                text = msg.get("content", "")
                break
        if not text:
            text = req.messages[-1].get("content", "") if req.messages else ""
        audio = generate_tts(text, req.voice, req.speed)
        buffer = io.BytesIO()
        sf.write(buffer, audio, 24000, format="WAV")
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=chat_response.wav"},
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/docs")
async def docs():
    return {
        "endpoints": {
            "GET /": "Server info",
            "GET /health": "Health check",
            "GET /v1/audio/voices": "List available voices",
            "POST /v1/audio/speech": "Text to speech (body: {model, input, voice?, speed?})",
            "POST /v1/chat/completions": "Chat with TTS (body: {model, messages, voice?, speed?})",
        },
        "quantization": "INT8 weight-only (TorchAO)",
        "voices_available": len(voices),
        "sample_request": {
            "url": "/v1/audio/speech",
            "method": "POST",
            "body": {
                "model": "omnivoice",
                "input": "Hello, this is a test.",
                "voice": "morgan-freeman",
                "speed": 1.0,
            },
        },
    }


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Server Endpoints:")
    print("=" * 60)
    print("  Web Interface:  http://localhost:8000")
    print("  API Docs:       http://localhost:8000/docs")
    print("  Health Check:   http://localhost:8000/health")
    print("  List Voices:    http://localhost:8000/v1/audio/voices")
    print("=" * 60)
    print("OpenAI-Compatible Endpoints:")
    print("  POST /v1/audio/speech      - Text to Speech")
    print("  GET  /v1/audio/voices      - List Voices")
    print("  POST /v1/chat/completions  - Voice Chat")
    print("=" * 60)
    print(f"Voices Loaded: {len(voices)}")
    print(f"Quantization:  INT8 weight-only (TorchAO)")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
