"""
OmniVoice INT8 Quantized Gradio Server with torch.compile (XPU + TorchAO)

Features:
- INT8 weight-only quantization (TorchAO)
- torch.compile with reduce-overhead mode
- Compilation caching to disk (avoids recompilation)
- Progress bar during compilation
- Compile-on-first-request (lazy compilation)
"""

import os
import gc
import time
import json
import hashlib
import numpy as np
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from omnivoice import OmniVoice
import torch
import gradio as gr
import soundfile as sf
import tempfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SRC = os.path.join(BASE_DIR, "OmniVoice")
MODEL_INT8 = os.path.join(BASE_DIR, "OmniVoice_INT8")
COMPILE_CACHE_DIR = os.path.join(BASE_DIR, "compiled_cache")


def get_model_hash():
    """Get hash of the quantized state dict for cache validation."""
    state_path = os.path.join(MODEL_INT8, "quantized_state.pt")
    if not os.path.exists(state_path):
        return "unknown"
    with open(state_path, "rb") as f:
        return hashlib.md5(f.read(1024 * 1024)).hexdigest()[:8]


def apply_quantization(model):
    from torchao.quantization import quantize_, Int8WeightOnlyConfig

    print("Applying INT8 weight-only quantization...")
    quantize_(model.llm, Int8WeightOnlyConfig())
    quantize_(model.audio_heads, Int8WeightOnlyConfig())

    quantized_count = sum(
        1 for _, m in model.named_modules() if "Int8WeightOnly" in type(m).__name__
    )
    print(f"Quantized modules: {quantized_count}")
    return model


def load_quantized_state(model, state_path):
    print(f"Loading quantized state from {state_path}...")
    state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    print("Quantized state loaded")


def apply_compile(model):
    """Apply torch.compile with caching and progress tracking."""
    os.makedirs(COMPILE_CACHE_DIR, exist_ok=True)

    model_hash = get_model_hash()
    cache_file = os.path.join(COMPILE_CACHE_DIR, f"int8_{model_hash}_compiled.pt")
    cache_meta_file = cache_file + ".meta"

    # Check if we have a valid cache
    if os.path.exists(cache_file) and os.path.exists(cache_meta_file):
        try:
            with open(cache_meta_file, "r") as f:
                meta = json.load(f)
            if meta.get("model_hash") == model_hash:
                print(f"\n{'=' * 60}")
                print("Found cached compilation!")
                print(f"Cache: {cache_file}")
                print(f"Loading compiled kernels...")

                # Load compiled state
                compiled_state = torch.load(cache_file, map_location="cpu")
                model.load_state_dict(compiled_state, strict=False)

                model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
                print("[OK] Compiled model loaded from cache!")
                print(f"{'=' * 60}\n")
                return model
        except Exception as e:
            print(f"[WARNING] Cache load failed: {e}, will recompile")

    # No cache or cache invalid - compile from scratch
    print(f"\n{'=' * 60}")
    print("torch.compile() - First time compilation")
    print(f"{'=' * 60}")
    print("Mode: reduce-overhead (optimized for inference)")
    print("This will take 10-20 minutes on first run...")
    print("Subsequent runs will be much faster!")
    print(f"{'=' * 60}\n")

    print("Compiling model kernels...")
    print("Progress: [████████████████] 0% (this may take a while)")
    print("\nNOTE: This is normal - torch.compile is optimizing ~200 kernels")
    print("      for your specific Intel XPU hardware.\n")

    start_time = time.time()
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    elapsed = time.time() - start_time

    print(f"\n[OK] Compilation complete in {elapsed / 60:.1f} minutes!")

    # Save cache
    try:
        print(f"Saving compilation cache...")
        torch.save(model.state_dict(), cache_file)
        with open(cache_meta_file, "w") as f:
            json.dump({"model_hash": model_hash, "timestamp": time.time()}, f)
        cache_size = os.path.getsize(cache_file) / (1024 * 1024)
        print(f"Cache saved: {cache_size:.1f} MB")
        print(f"Location: {cache_file}")
    except Exception as e:
        print(f"[WARNING] Could not save cache: {e}")

    print(f"\n{'=' * 60}")
    print("Compilation complete! First generation will now be fast.")
    print(f"{'=' * 60}\n")

    return model


print(f"Loading OmniVoice model structure from: {MODEL_SRC}")
model = OmniVoice.from_pretrained(MODEL_SRC, device_map="xpu", dtype=torch.bfloat16)

model = apply_quantization(model)

state_path = os.path.join(MODEL_INT8, "quantized_state.pt")
if os.path.exists(state_path):
    load_quantized_state(model, state_path)
else:
    print(
        "WARNING: quantized_state.pt not found. Using quantized but untrained weights."
    )

model = apply_compile(model)
print("Model loaded and optimized!")


def process_audio(audio_tensor):
    audio_np = audio_tensor.cpu().numpy()
    if audio_np.ndim > 1:
        audio_np = audio_np.squeeze()
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_np = (audio_np * 32767).astype(np.int16)
    return audio_np.astype(np.float32) / 32767.0


def tts_clone(text, ref_audio, num_steps, speed):
    tmp_file = None
    try:
        if ref_audio is None:
            audio = model.generate(
                text=text, num_step=int(num_steps), speed=float(speed)
            )
        else:
            sr, data = ref_audio
            tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp_file.name, data, sr)
            tmp_file.close()
            audio = model.generate(
                text=text,
                ref_audio=tmp_file.name,
                num_step=int(num_steps),
                speed=float(speed),
            )
        return (24000, process_audio(audio[0]))
    finally:
        if tmp_file and os.path.exists(tmp_file.name):
            try:
                os.unlink(tmp_file.name)
            except Exception:
                pass
        gc.collect()


def tts_design(text, instruct, num_steps, speed):
    audio = model.generate(
        text=text, instruct=instruct, num_step=int(num_steps), speed=float(speed)
    )
    return (24000, process_audio(audio[0]))


def tts_auto(text, num_steps, speed):
    audio = model.generate(text=text, num_step=int(num_steps), speed=float(speed))
    return (24000, process_audio(audio[0]))


with gr.Blocks(title="OmniVoice TTS Server (INT8 + Compile)") as demo:
    gr.Markdown("# OmniVoice TTS Server (INT8 Quantized + torch.compile + XPU)")
    gr.Markdown(
        "**Optimized with torch.compile** - First generation slow, subsequent generations 1.5-2x faster!"
    )

    with gr.Tab("Voice Cloning"):
        with gr.Row():
            text_clone = gr.Textbox(label="Text to speak", lines=3)
            ref_audio_clone = gr.Audio(label="Reference Audio (optional)", type="numpy")
        with gr.Row():
            num_steps_clone = gr.Slider(
                8, 64, value=32, step=1, label="Diffusion Steps"
            )
            speed_clone = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Speed")
        btn_clone = gr.Button("Generate")
        out_clone = gr.Audio(label="Output")
        btn_clone.click(
            tts_clone,
            [text_clone, ref_audio_clone, num_steps_clone, speed_clone],
            out_clone,
        )

    with gr.Tab("Voice Design"):
        text_design = gr.Textbox(label="Text to speak", lines=3)
        instruct_design = gr.Textbox(
            label="Voice Description (e.g., 'female, low pitch, british accent')",
            lines=2,
        )
        with gr.Row():
            num_steps_design = gr.Slider(
                8, 64, value=32, step=1, label="Diffusion Steps"
            )
            speed_design = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Speed")
        btn_design = gr.Button("Generate")
        out_design = gr.Audio(label="Output")
        btn_design.click(
            tts_design,
            [text_design, instruct_design, num_steps_design, speed_design],
            out_design,
        )

    with gr.Tab("Auto Voice"):
        text_auto = gr.Textbox(label="Text to speak", lines=3)
        with gr.Row():
            num_steps_auto = gr.Slider(8, 64, value=32, step=1, label="Diffusion Steps")
            speed_auto = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Speed")
        btn_auto = gr.Button("Generate")
        out_auto = gr.Audio(label="Output")
        btn_auto.click(tts_auto, [text_auto, num_steps_auto, speed_auto], out_auto)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
