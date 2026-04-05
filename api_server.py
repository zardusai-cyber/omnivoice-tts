import os
import gc
import io
import json
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
MODEL_PATH = BASE_DIR / "OmniVoice"
VOICES_DIR = BASE_DIR / "voices"

app = FastAPI(title="OmniVoice TTS Server", version="1.0.0")

print("=" * 60)
print("OmniVoice TTS API Server")
print("=" * 60)
print(f"[1/3] Loading model from: {MODEL_PATH}")
model = OmniVoice.from_pretrained(
    str(MODEL_PATH), device_map="xpu", dtype=torch.float16
)
print("[OK] Model loaded successfully")

print(f"[2/3] Scanning voices from: {VOICES_DIR}")
voices = {}
if VOICES_DIR.exists():
    for f in VOICES_DIR.iterdir():
        if f.suffix.lower() in [".wav", ".mp3", ".flac"]:
            name = f.stem.lower().replace("_", "-")
            voices[name] = str(f)
print(f"[OK] Found {len(voices)} voices")

print("[3/3] Server ready")
print("=" * 60)


def process_audio(audio_tensor):
    audio_np = audio_tensor.cpu().numpy()
    if audio_np.ndim > 1:
        audio_np = audio_np.squeeze()
    audio_np = np.clip(audio_np, -1.0, 1.0)
    return audio_np


def normalize_voice_id(voice_id: Optional[str]) -> Optional[str]:
    """Normalize voice ID to match how voices are stored."""
    if not voice_id:
        return None
    # Remove file extension if present
    voice_id = voice_id.lower()
    for ext in [".wav", ".mp3", ".flac", ".ogg"]:
        if voice_id.endswith(ext):
            voice_id = voice_id[: -len(ext)]
            break
    # Replace underscores/spaces with hyphens
    return voice_id.replace("_", "-").replace(" ", "-")


def generate_tts(
    text: str, voice: Optional[str] = None, speed: float = 1.0, num_steps: int = 32
):
    # Normalize voice ID to match stored format
    normalized_voice = normalize_voice_id(voice)

    if normalized_voice and normalized_voice in voices:
        print(f"[INFO] Using voice: {normalized_voice} (requested: {voice})")
        audio = model.generate(
            text=text,
            ref_audio=voices[normalized_voice],
            num_step=num_steps,
            speed=speed,
        )
    elif voice:
        # Log warning but continue with default voice
        print(
            f"[WARNING] Voice '{voice}' (normalized: '{normalized_voice}') not found. Available: {list(voices.keys())[:5]}..."
        )
        print(f"[INFO] Using default voice (no reference audio)")
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
        "message": "OmniVoice TTS Server",
        "docs": "/docs",
        "voices": "/v1/audio/voices",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": "loaded", "voices": len(voices)}


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
    """OpenAI-compatible voices list endpoint - matches pocket-tts format."""
    voice_list = []
    for voice_id in sorted(voices.keys()):
        # Extract display name from voice ID
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
    """OpenAI-compatible TTS endpoint - matches pocket-tts behavior."""
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
    """OpenAI-compatible chat completions with voice output."""
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
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
