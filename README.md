---
title: FakeArt Detector
emoji: 🎨
colorFrom: pink
colorTo: red
sdk: gradio
sdk_version: 5.49.1
python_version: "3.11"
app_file: app.py
pinned: false
suggested_hardware: t4-small
thumbnail: https://huggingface.co/spaces/onlineshihab/FakeArtDetector/resolve/main/assets/link_preview.png
models:
  - onlineshihab/fake-artwork-llava-lora
---

# Fake Artwork Detector

A Gradio application for the `onlineshihab/fake-artwork-llava-lora` adapter.

## Hardware

A CUDA GPU is required. Use a Hugging Face T4 Small Space or higher.

## Build notes

- Gradio is pinned by `sdk_version: 5.49.1`.
- Do not add `gradio>=5.0,<6.0` inside `requirements.txt`.
- Python is pinned to 3.11 so `torch==2.4.1` and `torchvision==0.19.1` install correctly.
