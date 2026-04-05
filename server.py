import os
import gc
import numpy as np

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from omnivoice import OmniVoice
import torch
import gradio as gr
import soundfile as sf
import tempfile

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OmniVoice")

print(f"Loading OmniVoice model from: {MODEL_PATH}")
print(f"Audio tokenizer path: {os.path.join(MODEL_PATH, 'audio_tokenizer')}")

model = OmniVoice.from_pretrained(MODEL_PATH, device_map="xpu", dtype=torch.float16)
print("Model loaded successfully!")


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
            except:
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


with gr.Blocks(title="OmniVoice TTS Server") as demo:
    gr.Markdown("# OmniVoice TTS Server (XPU)")

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
