"""
Quantize OmniVoice BF16 model to INT8 weight-only using TorchAO.

This script:
1. Loads the BF16 model from OmniVoice/
2. Applies INT8 weight-only quantization to all Linear layers in the LLM backbone + audio head
3. Saves the quantized state dict to OmniVoice_INT8/
4. Copies config, tokenizer, and audio_tokenizer files

Usage:
    python quantize_model.py              # Uses OmniVoice/ as source
    python quantize_model.py --source OmniVoice_FP32/  # Custom source
    python quantize_model.py --no-save    # Quantize and test without saving
"""

import os
import sys
import shutil
import argparse
import time
import json
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch

BASE_DIR = Path(__file__).parent


def get_memory_usage():
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return f"XPU memory: {torch.xpu.memory_allocated() / 1e9:.2f} GB allocated, {torch.xpu.memory_reserved() / 1e9:.2f} GB reserved"
    except Exception:
        pass
    return "Memory info unavailable"


def main():
    parser = argparse.ArgumentParser(
        description="Quantize OmniVoice to INT8 weight-only"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="OmniVoice",
        help="Source BF16 model directory (default: OmniVoice)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="OmniVoice_INT8",
        help="Output INT8 model directory (default: OmniVoice_INT8)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Quantize and test without saving to disk",
    )
    args = parser.parse_args()

    source_dir = BASE_DIR / args.source
    output_dir = BASE_DIR / args.output

    if not source_dir.exists():
        print(f"[ERROR] Source directory not found: {source_dir}")
        sys.exit(1)

    print("=" * 60)
    print("OmniVoice INT8 Weight-Only Quantization")
    print("=" * 60)
    print(f"Source:  {source_dir}")
    print(f"Output:  {output_dir}")
    print(f"Method:  TorchAO Int8WeightOnlyConfig")
    print(f"Backend: PyTorch XPU")
    print("=" * 60)

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        print("[WARNING] XPU not available. Falling back to CPU (very slow).")
        print("          Install PyTorch XPU nightly for proper Intel GPU support.")
        device = "cpu"
    else:
        device = "xpu"
        print(f"[OK] XPU available: {torch.xpu.get_device_name(0)}")

    try:
        from torchao.quantization import quantize_, Int8WeightOnlyConfig
    except ImportError:
        print("[ERROR] torchao not installed. Run: pip install torchao")
        sys.exit(1)

    try:
        from omnivoice import OmniVoice
    except ImportError:
        print("[ERROR] omnivoice not installed.")
        sys.exit(1)

    print(f"\n[1/4] Loading BF16 model from {source_dir}...")
    t0 = time.time()
    model = OmniVoice.from_pretrained(
        str(source_dir),
        device_map=device,
        dtype=torch.bfloat16,
    )
    load_time = time.time() - t0
    print(f"[OK] Model loaded in {load_time:.1f}s")
    print(f"     {get_memory_usage()}")

    print(f"\n[2/4] Applying INT8 weight-only quantization...")
    print(f"     Quantizing LLM backbone (197 Linear layers)...")
    t0 = time.time()

    quantize_(model.llm, Int8WeightOnlyConfig())

    print(f"     Quantizing audio head (1 Linear layer)...")
    quantize_(model.audio_heads, Int8WeightOnlyConfig())

    quant_time = time.time() - t0
    print(f"[OK] Quantization complete in {quant_time:.1f}s")
    print(f"     {get_memory_usage()}")

    quantized_count = 0
    total_params = 0
    quantized_params = 0
    for name, module in model.named_modules():
        param_count = sum(p.numel() for p in module.parameters())
        total_params += param_count
        if "Int8WeightOnly" in type(module).__name__:
            quantized_count += 1
            quantized_params += param_count

    print(f"     Quantized modules: {quantized_count}")
    print(
        f"     Quantized params: {quantized_params / 1e6:.1f}M / {total_params / 1e6:.1f}M ({quantized_params / total_params * 100:.0f}%)"
    )

    print(f"\n[3/4] Running test generation...")
    t0 = time.time()
    try:
        audio = model.generate(
            text="Hello, this is a test of INT8 quantized OmniVoice.",
            num_step=16,
            speed=1.0,
        )
        gen_time = time.time() - t0
        audio_duration = audio[0].shape[-1] / 24000
        rtf = gen_time / audio_duration if audio_duration > 0 else float("inf")
        print(f"[OK] Test generation successful")
        print(
            f"     Generated {audio_duration:.1f}s of audio in {gen_time:.1f}s (RTF: {rtf:.2f})"
        )
    except Exception as e:
        print(f"[WARNING] Test generation failed: {e}")
        print("          Model is still quantized. You can try saving anyway.")

    if args.no_save:
        print(f"\n[4/4] Skipping save (--no-save flag)")
        print("Done! Model is quantized in memory but not saved to disk.")
        return

    print(f"\n[4/4] Saving quantized model to {output_dir}...")

    if output_dir.exists():
        print(f"     Removing existing {output_dir}...")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    print(f"     Saving quantized state dict...")
    t0 = time.time()
    state_path = output_dir / "quantized_state.pt"
    torch.save(model.state_dict(), str(state_path))
    save_time = time.time() - t0
    state_size = state_path.stat().st_size / (1024**3)
    print(f"[OK] State dict saved in {save_time:.1f}s ({state_size:.2f} GB)")

    files_to_copy = [
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "README.md",
        ".gitattributes",
    ]
    for fname in files_to_copy:
        src = source_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            print(f"     Copied {fname}")

    audio_tokenizer_src = source_dir / "audio_tokenizer"
    if audio_tokenizer_src.exists():
        audio_tokenizer_dst = output_dir / "audio_tokenizer"
        if audio_tokenizer_dst.exists():
            shutil.rmtree(audio_tokenizer_dst)
        shutil.copytree(audio_tokenizer_src, audio_tokenizer_dst)
        print(f"     Copied audio_tokenizer/")

    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        config["quantization"] = {
            "method": "torchao.Int8WeightOnlyConfig",
            "quantized_modules": quantized_count,
            "quantized_params": quantized_params,
            "total_params": total_params,
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"     Updated config.json with quantization metadata")

    print(f"\n{'=' * 60}")
    print(f"Quantization complete!")
    print(f"{'=' * 60}")
    print(
        f"Source BF16 size:  {sum(f.stat().st_size for f in source_dir.rglob('*') if f.is_file()) / (1024**3):.2f} GB"
    )
    print(
        f"Output INT8 size:  {sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file()) / (1024**3):.2f} GB"
    )
    print(f"Model path:        {output_dir}")
    print(f"{'=' * 60}")
    print(f"\nTo use this model, run:")
    print(f"  start_int8.bat        (Gradio web UI)")
    print(f"  start_api_int8.bat    (REST API server)")


if __name__ == "__main__":
    main()
