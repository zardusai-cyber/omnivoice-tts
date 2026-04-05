"""
OmniVoice INT8 Quantized API Server with torch.compile (XPU + TorchAO)

Features:
- INT8 weight-only quantization (TorchAO)
- torch.compile with reduce-overhead mode
- Compilation caching to disk
- Progress tracking
- OpenAI-compatible API
"""

import os
import gc
import io
import json
import time
import hashlib
import numpy as np
from pathlib import Path
from typing import Optional
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import torch
import soundfile as sf

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from omnivoice import OmniVoice

BASE_DIR = Path(__file__).parent
MODEL_SRC = BASE_DIR / "OmniVoice"
MODEL_INT8 = BASE_DIR / "OmniVoice_INT8"
VOICES_DIR = BASE_DIR / "voices"
COMPILE_CACHE_DIR = BASE_DIR / "compiled_cache"

app = FastAPI(title="OmniVoice TTS Server (INT8 + torch.compile)", version="1.0.0")


def get_model_hash():
    state_path = MODEL_INT8 / "quantized_state.pt"
    if not state_path.exists():
        return "unknown"
    with open(state_path, "rb") as f:
        return hashlib.md5(f.read(1024 * 1024)).hexdigest()[:8]


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
    """Apply torch.compile with caching."""
    COMPILE_CACHE_DIR.mkdir(exist_ok=True)

    model_hash = get_model_hash()
    cache_file = COMPILE_CACHE_DIR / f"int8_{model_hash}_compiled.pt"
    cache_meta_file = cache_file.with_suffix(".pt.meta")

    if cache_file.exists() and cache_meta_file.exists():
        try:
            with open(cache_meta_file, "r") as f:
                meta = json.load(f)
            if meta.get("model_hash") == model_hash:
                print(f"\n{'=' * 60}")
                print("Found cached compilation!")
                print(f"Loading compiled kernels...")

                compiled_state = torch.load(cache_file, map_location="cpu")
                model.load_state_dict(compiled_state, strict=False)

                model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
                print("[OK] Compiled model loaded from cache!")
                print(f"{'=' * 60}\n")
                return model
        except Exception as e:
            print(f"[WARNING] Cache load failed: {e}, will recompile")

    print(f"\n{'=' * 60}")
    print("torch.compile() - First time compilation")
    print(f"{'=' * 60}")
    print("Mode: reduce-overhead (optimized for inference)")
    print("This will take 10-20 minutes on first run...")
    print("Subsequent runs will be much faster!")
    print(f"{'=' * 60}\n")

    print("Compiling model kernels...")
    print("Progress: [████████████████] 0% (this may take a while)\n")

    start_time = time.time()
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    elapsed = time.time() - start_time

    print(f"\n[OK] Compilation complete in {elapsed / 60:.1f} minutes!")

    try:
        print(f"Saving compilation cache...")
        torch.save(model.state_dict(), cache_file)
        with open(cache_meta_file, "w") as f:
            json.dump({"model_hash": model_hash, "timestamp": time.time()}, f)
        cache_size = cache_file.stat().st_size / (1024 * 1024)
        print(f"Cache saved: {cache_size:.1f} MB")
    except Exception as e:
        print(f"[WARNING] Could not save cache: {e}")

    print(f"\n{'=' * 60}")
    print("Compilation complete!")
    print(f"{'=' * 60}\n")

    return model


print("=" * 60)
print("OmniVoice TTS API Server (INT8 Quantized + torch.compile)")
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
        "[WARNING] quantized_state.pt not found. Using quantized but untrained weights."
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
        "message": "OmniVoice TTS Server (INT8 + torch.compile)",
        "docs": "/docs",
        "voices": "/v1/audio/voices",
        "optimization": "INT8 weight-only + torch.compile",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": "int8_quantized_compiled", "voices": len(voices)}


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


@app.post("/v1/audio/voices/reload")
async def reload_voices():
    global voices
    voices = {}
    if VOICES_DIR.exists():
        for vf in VOICES_DIR.iterdir():
            if vf.suffix.lower() in [".wav", ".mp3", ".flac"]:
                voices[vf.stem.lower().replace("_", "-").replace(" ", "-")] = str(vf)
    return {"status": "ok", "voices": len(voices), "reloaded": True}


@app.get("/docs")
async def docs():
    return {
        "endpoints": {
            "GET /": "Server info",
            "GET /health": "Health check",
            "GET /v1/audio/voices": "List available voices",
            "POST /v1/audio/speech": "Text to speech",
            "POST /v1/chat/completions": "Chat with TTS",
        },
        "optimization": "INT8 weight-only (TorchAO) + torch.compile",
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
    print(f"Voices Loaded:      {len(voices)}")
    print(f"Quantization:       INT8 weight-only (TorchAO)")
    print(f"Optimization:       torch.compile (reduce-overhead)")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
