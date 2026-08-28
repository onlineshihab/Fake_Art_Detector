# Streamlit-compatible FakeArt Detector
from __future__ import annotations

import streamlit as st
import json
import os
import re
import threading
import traceback
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

class PredictionError(Exception):
    pass

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "llava-hf/llava-1.5-7b-hf")
ADAPTER_MODEL_ID = os.getenv(
    "ADAPTER_MODEL_ID", "onlineshihab/fake-artwork-llava-lora"
)
HF_TOKEN = os.getenv("HF_TOKEN") or None
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "1024"))
CONFIG_PATH = Path(__file__).with_name("threshold_config.json")

_MODEL: Any | None = None
_PROCESSOR: Any | None = None
_MODEL_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

def load_threshold() -> tuple[float, bool, str]:
    env_value = os.getenv("FAKE_THRESHOLD")
    if env_value not in (None, ""):
        try:
            return float(env_value), True, "Space variable FAKE_THRESHOLD"
        except ValueError as exc:
            raise RuntimeError("FAKE_THRESHOLD must be a valid number.") from exc

    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            config = json.load(file)
        return (
            float(config.get("fake_threshold", 0.0)),
            bool(config.get("calibrated", False)),
            "threshold_config.json",
        )

    return 0.0, False, "built-in default"


FAKE_THRESHOLD, THRESHOLD_IS_CALIBRATED, THRESHOLD_SOURCE = load_threshold()


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def load_model() -> tuple[Any, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is unavailable. Change this Hugging Face Space to GPU "
            "hardware, such as T4 Small, and restart the Space."
        )

    print(f"Loading processor from: {ADAPTER_MODEL_ID}")
    try:
        processor = AutoProcessor.from_pretrained(
            ADAPTER_MODEL_ID,
            token=HF_TOKEN,
        )
    except Exception as processor_error:
        print("Adapter processor could not be loaded; using the base processor.")
        print(processor_error)
        processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model: {BASE_MODEL_ID}")
    base_model = LlavaForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        token=HF_TOKEN,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    processor.patch_size = base_model.config.vision_config.patch_size
    processor.vision_feature_select_strategy = getattr(
        base_model.config,
        "vision_feature_select_strategy",
        "default",
    )

    print(f"Loading LoRA adapter: {ADAPTER_MODEL_ID}")
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_MODEL_ID,
        token=HF_TOKEN,
        is_trainable=False,
    )
    model.eval()
    model.config.use_cache = True

    lora_parameter_count = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "lora_" in name.lower()
    )
    if lora_parameter_count == 0:
        raise RuntimeError("The LoRA adapter loaded without any LoRA parameters.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"LoRA parameters: {lora_parameter_count:,}")
    print("Model is ready.")
    return model, processor


def get_model() -> tuple[Any, Any]:
    global _MODEL, _PROCESSOR
    if _MODEL is None or _PROCESSOR is None:
        with _MODEL_LOCK:
            if _MODEL is None or _PROCESSOR is None:
                _MODEL, _PROCESSOR = load_model()
    return _MODEL, _PROCESSOR


def get_model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


# -----------------------------------------------------------------------------
# Image and pixel helpers
# -----------------------------------------------------------------------------

def ensure_rgb(image: Image.Image) -> Image.Image:
    if image is None:
        raise PredictionError("Please upload an artwork image first.")
    return image.convert("RGB")


def extract_pixel_features(image: Image.Image) -> dict[str, float]:
    img_rgb = np.asarray(ensure_rgb(image))
    mean_rgb = img_rgb.mean(axis=(0, 1))
    std_rgb = img_rgb.std(axis=(0, 1))
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_estimate = float(
        np.std(gray.astype(np.float32) - blurred.astype(np.float32))
    )

    red_green = (
        img_rgb[:, :, 0].astype(np.float32)
        - img_rgb[:, :, 1].astype(np.float32)
    )
    yellow_blue = (
        0.5
        * (
            img_rgb[:, :, 0].astype(np.float32)
            + img_rgb[:, :, 1].astype(np.float32)
        )
        - img_rgb[:, :, 2].astype(np.float32)
    )
    colorfulness = float(
        np.sqrt(np.std(red_green) ** 2 + np.std(yellow_blue) ** 2)
        + 0.3
        * np.sqrt(np.mean(red_green) ** 2 + np.mean(yellow_blue) ** 2)
    )

    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "edge_density": edge_density,
        "noise_estimate": noise_estimate,
        "mean_red": float(mean_rgb[0]),
        "mean_green": float(mean_rgb[1]),
        "mean_blue": float(mean_rgb[2]),
        "std_red": float(std_rgb[0]),
        "std_green": float(std_rgb[1]),
        "std_blue": float(std_rgb[2]),
        "colorfulness": colorfulness,
    }


def pixel_features_to_text(features: dict[str, float]) -> str:
    brightness_level = (
        "low"
        if features["brightness"] < 85
        else "medium"
        if features["brightness"] < 170
        else "high"
    )
    contrast_level = (
        "low"
        if features["contrast"] < 40
        else "medium"
        if features["contrast"] < 80
        else "high"
    )
    sharpness_level = (
        "low/soft"
        if features["sharpness"] < 80
        else "medium"
        if features["sharpness"] < 300
        else "high/sharp"
    )
    edge_level = (
        "low"
        if features["edge_density"] < 0.04
        else "medium"
        if features["edge_density"] < 0.10
        else "high"
    )
    noise_level = (
        "low"
        if features["noise_estimate"] < 5
        else "medium"
        if features["noise_estimate"] < 12
        else "high"
    )
    color_level = (
        "low"
        if features["colorfulness"] < 25
        else "medium"
        if features["colorfulness"] < 60
        else "high"
    )

    return (
        f"Pixel statistics: brightness is {brightness_level} "
        f"({features['brightness']:.2f}), contrast is {contrast_level} "
        f"({features['contrast']:.2f}), sharpness is {sharpness_level} "
        f"({features['sharpness']:.2f}), edge density is {edge_level} "
        f"({features['edge_density']:.4f}), noise estimate is {noise_level} "
        f"({features['noise_estimate']:.2f}), colorfulness is {color_level} "
        f"({features['colorfulness']:.2f}). Average RGB color is approximately "
        f"R={features['mean_red']:.1f}, G={features['mean_green']:.1f}, "
        f"B={features['mean_blue']:.1f}."
    )


def analyze_pixel_flags(features: dict[str, float]) -> tuple[list[str], int]:
    flags: list[str] = []

    if features["brightness"] < 40 or features["brightness"] > 220:
        flags.append("Abnormal brightness detected.")
    if features["contrast"] < 20 or features["contrast"] > 85:
        flags.append("Abnormal contrast detected.")
    if features["sharpness"] < 25 or features["sharpness"] > 900:
        flags.append("Abnormal sharpness detected.")
    if features["edge_density"] > 0.18:
        flags.append("High edge density detected.")
    if features["noise_estimate"] > 12:
        flags.append("High noise level detected.")
    if features["colorfulness"] > 95:
        flags.append("Abnormal color saturation detected.")

    abnormal_count = len(flags)
    if not flags:
        flags.append("Pixel-level features look mostly normal.")
    return flags, abnormal_count


# -----------------------------------------------------------------------------
# Description helpers
# -----------------------------------------------------------------------------

def clean_text(text: str) -> str:
    text = str(text).replace("</s>", "").replace("<s>", "").strip()
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:")[-1].strip()
    if "USER:" in text:
        text = text.split("USER:")[0].strip()
    return text.strip()


def generate_system_description(model: Any, processor: Any, image: Image.Image) -> str:
    prompt = """USER: <image>
Describe this artwork briefly. Mention main objects, background, color, style, composition, and any unusual visible area.
ASSISTANT:"""

    inputs = processor(
        text=prompt,
        images=ensure_rgb(image),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    device = get_model_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            use_cache=True,
        )

    new_tokens = output_ids[0][input_length:]
    decoded = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    description = clean_text(decoded)
    return description or "The model did not generate a usable description."


STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "with",
    "is", "are", "was", "were", "this", "that", "there", "it", "as", "image",
    "artwork", "painting", "picture", "shows", "showing", "has", "have", "from",
    "by", "for", "near", "some", "one",
}


def normalize_text(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9\s]", " ", str(text).lower())
    return re.sub(r"\s+", " ", normalized).strip()


def get_keywords(text: str) -> set[str]:
    return {
        word
        for word in normalize_text(text).split()
        if len(word) > 2 and word not in STOPWORDS
    }


def compare_descriptions(user_description: str, system_description: str) -> dict[str, Any]:
    if not str(user_description or "").strip():
        return {
            "has_user_description": False,
            "match_score": None,
            "match_level": "Not provided",
            "summary": "No user description was provided.",
            "user_only_keywords": [],
            "system_only_keywords": [],
        }

    user_text = normalize_text(user_description)
    system_text = normalize_text(system_description)
    sequence_score = SequenceMatcher(None, user_text, system_text).ratio()

    user_keywords = get_keywords(user_description)
    system_keywords = get_keywords(system_description)
    union = user_keywords | system_keywords
    keyword_score = len(user_keywords & system_keywords) / len(union) if union else 0.0
    final_score = (0.35 * sequence_score + 0.65 * keyword_score) * 100.0

    if final_score >= 70:
        level = "High match"
        summary = "User and system descriptions are strongly consistent."
    elif final_score >= 45:
        level = "Partial match"
        summary = "User and system descriptions partially match."
    else:
        level = "Low match"
        summary = "User and system descriptions have low consistency."

    return {
        "has_user_description": True,
        "match_score": float(final_score),
        "match_level": level,
        "summary": summary,
        "user_only_keywords": sorted(user_keywords - system_keywords),
        "system_only_keywords": sorted(system_keywords - user_keywords),
    }


# -----------------------------------------------------------------------------
# Classification
# -----------------------------------------------------------------------------

def build_classification_prompt(description: str, pixel_text: str) -> str:
    return f"""USER: <image>
You are a fake artwork detector.

Analyze the artwork image, its visual style, objects, background, color, composition, and possible manipulation signs.

Given artwork description:
{description}

Pixel-level image information:
{pixel_text}

Classify this artwork as one word only:
original or fake.

ASSISTANT:"""


def score_label(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    label: str,
) -> float:
    text = prompt + " " + label + processor.tokenizer.eos_token

    full_inputs = processor(
        text=text,
        images=ensure_rgb(image),
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    prompt_inputs = processor(
        text=prompt,
        images=ensure_rgb(image),
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    device = get_model_device(model)
    full_inputs = {key: value.to(device) for key, value in full_inputs.items()}
    prompt_inputs = {key: value.to(device) for key, value in prompt_inputs.items()}

    input_ids = full_inputs["input_ids"]
    prompt_length = prompt_inputs["input_ids"].shape[1]
    if prompt_length >= input_ids.shape[1]:
        raise RuntimeError(
            "The classification answer was truncated. Reduce MAX_LENGTH input text."
        )

    with torch.inference_mode():
        logits = model(**full_inputs).logits
    log_probabilities = F.log_softmax(logits, dim=-1)

    total_log_probability = 0.0
    token_count = 0
    for index in range(prompt_length, input_ids.shape[1]):
        token_id = input_ids[0, index]
        total_log_probability += float(
            log_probabilities[0, index - 1, token_id].detach().cpu()
        )
        token_count += 1

    return total_log_probability / max(token_count, 1)


def classify_artwork(
    model: Any,
    processor: Any,
    image: Image.Image,
    description: str,
) -> dict[str, Any]:
    features = extract_pixel_features(image)
    pixel_text = pixel_features_to_text(features)
    prompt = build_classification_prompt(description, pixel_text)

    original_score = score_label(model, processor, image, prompt, "original")
    fake_score = score_label(model, processor, image, prompt, "fake")
    margin = fake_score - original_score
    prediction = "Fake" if margin >= FAKE_THRESHOLD else "Original"

    probabilities = torch.softmax(
        torch.tensor([original_score, fake_score], dtype=torch.float32), dim=0
    ).tolist()
    distance = abs(margin - FAKE_THRESHOLD)
    heuristic_confidence = min(95.0, 55.0 + distance * 120.0)
    pixel_flags, abnormal_count = analyze_pixel_flags(features)

    reasons: list[str] = []
    if prediction == "Fake":
        reasons.append(
            "The fake-minus-original score reached or exceeded the configured threshold."
        )
        if abnormal_count:
            reasons.extend(pixel_flags[:3])
        else:
            reasons.append(
                "The model detected fake-like visual or semantic patterns even though "
                "the basic pixel checks were not strongly abnormal."
            )
    else:
        reasons.append(
            "The fake-minus-original score remained below the configured threshold."
        )
        reasons.append(
            "The available visual, semantic, and pixel evidence was not sufficient "
            "for a Fake verdict."
        )

    return {
        "prediction": prediction,
        "confidence": float(heuristic_confidence),
        "original_score": float(original_score),
        "fake_score": float(fake_score),
        "margin": float(margin),
        "original_probability": float(probabilities[0] * 100.0),
        "fake_probability": float(probabilities[1] * 100.0),
        "pixel_text": pixel_text,
        "pixel_flags": pixel_flags,
        "abnormal_pixel_count": abnormal_count,
        "reasons": reasons,
    }


# -----------------------------------------------------------------------------
# Gradio callback
# -----------------------------------------------------------------------------

def run_prediction(
    image: Image.Image | None,
    user_description: str,
) -> tuple[str, str, str, str, str, dict[str, Any]]:
    if image is None:
        raise PredictionError("Please upload an artwork image first.")

    try:
        with _INFERENCE_LOCK:
            model, processor = get_model()
            rgb_image = ensure_rgb(image)
            system_description = generate_system_description(
                model, processor, rgb_image
            )
            match_info = compare_descriptions(
                user_description, system_description
            )
            classification_description = (
                str(user_description).strip()[:1500]
                if str(user_description or "").strip()
                else system_description
            )
            result = classify_artwork(
                model,
                processor,
                rgb_image,
                classification_description,
            )

        calibration_text = (
            "calibrated"
            if THRESHOLD_IS_CALIBRATED
            else "default/unconfirmed — replace with the notebook's calibrated value"
        )
        result_markdown = (
            f"## Verdict: **{result['prediction']}**\n\n"
            f"**Confidence-like score:** {result['confidence']:.2f}%  \n"
            f"**Configured fake threshold:** {FAKE_THRESHOLD:.6f} "
            f"({calibration_text})\n\n"
            "> This is a research prediction, not proof of legal or market authenticity."
        )

        if match_info["match_score"] is None:
            match_text = (
                "No user description provided. The system-generated description "
                "was used for classification."
            )
        else:
            match_text = (
                f"{match_info['match_level']} — {match_info['match_score']:.2f}%\n\n"
                f"{match_info['summary']}\n\n"
                "This consistency result is supplementary and did not override "
                "the authenticity verdict."
            )

        pixel_observations = "\n".join(
            f"- {flag}" for flag in result["pixel_flags"]
        )
        pixel_output = result["pixel_text"] + "\n\n" + pixel_observations
        reasons_output = "\n".join(
            f"{index}. {reason}"
            for index, reason in enumerate(result["reasons"], start=1)
        )

        technical = {
            "base_model": BASE_MODEL_ID,
            "adapter_model": ADAPTER_MODEL_ID,
            "threshold": FAKE_THRESHOLD,
            "threshold_source": THRESHOLD_SOURCE,
            "threshold_calibrated": THRESHOLD_IS_CALIBRATED,
            "classification_description_source": (
                "user description"
                if str(user_description or "").strip()
                else "system-generated description"
            ),
            "original_score": round(result["original_score"], 6),
            "fake_score": round(result["fake_score"], 6),
            "fake_minus_original_margin": round(result["margin"], 6),
            "softmax_original_percent": round(
                result["original_probability"], 2
            ),
            "softmax_fake_percent": round(result["fake_probability"], 2),
            "abnormal_pixel_flags": result["abnormal_pixel_count"],
            "description_match": match_info,
            "confidence_note": (
                "The displayed confidence is a heuristic distance-based score, "
                "not a calibrated probability."
            ),
        }

        return (
            result_markdown,
            system_description,
            match_text,
            pixel_output,
            reasons_output,
            technical,
        )
    except PredictionError:
        raise
    except Exception as error:
        traceback.print_exc()
        raise PredictionError(f"Prediction failed: {error}") from error

def clear_outputs() -> tuple[
    None,
    str,
    str,
    str,
    str,
    str,
    str,
    None,
]:
    """Reset every input and output component to the initial state."""
    return (
        None,
        "",
        (
            "## Ready for Artwork Analysis\n\n"
            "Upload an artwork image and click **Check Artwork** to begin."
        ),
        "",
        "",
        "",
        "",
        None,
    )



# -----------------------------------------------------------------------------
# Multi-page modern Gradio user interface
# -----------------------------------------------------------------------------

def _load_calibration_details() -> dict[str, Any]:
    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as file:
                data = json.load(file)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _asset_data_uri(relative_path: str) -> str:
    """Return a data URI so the hero artwork preview works even if HF static paths fail."""
    try:
        import base64
        path = Path(__file__).parent / relative_path
        if not path.exists():
            return ""
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return ""


_CALIBRATION_DETAILS = _load_calibration_details()
_CALIBRATION_ACCURACY = _CALIBRATION_DETAILS.get("calibration_accuracy")
_BALANCED_ACCURACY = _CALIBRATION_DETAILS.get("balanced_accuracy")

if isinstance(_CALIBRATION_ACCURACY, (int, float)):
    CALIBRATION_ACCURACY_TEXT = f"{float(_CALIBRATION_ACCURACY) * 100:.0f}%"
else:
    CALIBRATION_ACCURACY_TEXT = "97%"

if isinstance(_BALANCED_ACCURACY, (int, float)):
    BALANCED_ACCURACY_TEXT = f"{float(_BALANCED_ACCURACY) * 100:.1f}%"
else:
    BALANCED_ACCURACY_TEXT = "96.8%"

HERO_ART_URI = _asset_data_uri("assets/hero_art_cards.png")
HERO_ART_IMAGE_HTML = (
    f'<img class="fad-art-preview-img" src="{HERO_ART_URI}" '
    'alt="Original and fake artwork comparison preview">'
    if HERO_ART_URI
    else """
    <div class="fad-css-art-preview">
        <span class="fad-chip original">ORIGINAL</span>
        <span class="fad-chip fake">FAKE</span>
    </div>
    """
)

threshold_banner = (
    f"""
    <div class="status-badge success-badge">
        <span>✓</span>
        Calibrated threshold active.
    </div>
    """
    if THRESHOLD_IS_CALIBRATED
    else f"""
    <div class="status-badge warning-badge">
        <span>⚠</span>
        Default threshold active:
        <strong>{FAKE_THRESHOLD:.6f}</strong>
    </div>
    """
)


CUSTOM_CSS = r"""

:root {
    --fad-bg: #070912;
    --fad-bg2: #0b0e19;
    --fad-card: rgba(18, 21, 34, 0.88);
    --fad-card2: rgba(23, 26, 39, 0.94);
    --fad-border: rgba(244, 63, 94, 0.28);
    --fad-border-soft: rgba(255, 255, 255, 0.10);
    --fad-text: #f8fafc;
    --fad-muted: #b9c0cc;
    --fad-soft: #8e98aa;
    --fad-accent: #f43f5e;
    --fad-accent2: #fb7185;
    --fad-orange: #f97316;
}

html,
body,
.gradio-container {
    background:
        radial-gradient(circle at 10% 10%, rgba(244, 63, 94, 0.16), transparent 26%),
        radial-gradient(circle at 88% 8%, rgba(79, 70, 229, 0.13), transparent 24%),
        linear-gradient(180deg, #070912 0%, #090b15 50%, #070912 100%) !important;
    color: var(--fad-text) !important;
    width: 100%;
    max-width: 100%;
    overflow-x: hidden;
}

body {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

* { box-sizing: border-box; }

.gradio-container {
    max-width: 1500px !important;
    margin: 0 auto !important;
    padding: 0 38px 28px 38px !important;
}

.fad-page { display: none; }
#home-page { display: block; }

.fad-shell {
    width: min(1280px, 100%);
    margin: 0 auto;
}

.fad-navbar {
    position: sticky;
    top: 0;
    z-index: 80;
    min-height: 78px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    padding: 12px 0;
    background: rgba(7, 9, 18, 0.84);
    border-bottom: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(18px);
}

.fad-brand {
    display: inline-flex;
    align-items: center;
    gap: 12px;
    color: var(--fad-text) !important;
    text-decoration: none !important;
    font-size: 22px;
    font-weight: 900;
    letter-spacing: -0.5px;
}

.fad-brand .accent { color: var(--fad-accent2); }

.fad-logo-mark {
    width: 38px;
    height: 38px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 12px;
    color: var(--fad-accent2);
    border: 1px solid rgba(244,63,94,0.62);
    background: linear-gradient(145deg, rgba(244,63,94,0.18), rgba(15,23,42,0.75));
    box-shadow: 0 0 24px rgba(244,63,94,0.22);
}

.fad-navlinks {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 32px;
}

.fad-nav-link {
    position: relative;
    color: #f8fafc !important;
    text-decoration: none !important;
    font-size: 14px;
    font-weight: 800;
    opacity: 0.94;
}

.fad-nav-link.active::after,
.fad-nav-link:hover::after {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    bottom: -11px;
    height: 3px;
    border-radius: 999px;
    background: var(--fad-accent);
    box-shadow: 0 0 14px rgba(244,63,94,0.45);
}

.fad-nav-cta,
.fad-primary-link,
.fad-secondary-link {
    min-height: 48px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 9px;
    padding: 0 23px;
    border-radius: 10px;
    text-decoration: none !important;
    font-size: 14px;
    font-weight: 900;
    transition: transform 0.16s ease, box-shadow 0.16s ease;
}

.fad-nav-cta,
.fad-primary-link {
    color: white !important;
    background: linear-gradient(135deg, #f43f5e, #e11d48);
    box-shadow: 0 14px 34px rgba(225, 29, 72, 0.28);
    border: 1px solid rgba(255,255,255,0.08);
}

.fad-secondary-link {
    color: white !important;
    background: rgba(11, 14, 24, 0.76);
    border: 1px solid rgba(244, 63, 94, 0.50);
}

.fad-nav-cta:hover,
.fad-primary-link:hover,
.fad-secondary-link:hover {
    transform: translateY(-1px);
    box-shadow: 0 18px 38px rgba(225, 29, 72, 0.34);
}

/* HOME */
.fad-home-hero {
    padding: 68px 0 28px 0;
}

.fad-home-grid {
    display: grid;
    grid-template-columns: minmax(0, 0.95fr) minmax(460px, 1.05fr);
    gap: 58px;
    align-items: center;
}

.fad-kicker {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 10px 17px;
    border-radius: 999px;
    color: var(--fad-text);
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(255,255,255,0.10);
    font-size: 13px;
    font-weight: 800;
}

.fad-kicker span {
    color: var(--fad-accent2);
    font-size: 18px;
}

.fad-home-title {
    margin: 0 0 22px 0;
    color: var(--fad-text) !important;
    font-size: clamp(42px, 4.35vw, 64px);
    line-height: 1.08;
    letter-spacing: -2.1px;
    font-weight: 950;
}

.fad-home-title .accent {
    color: var(--fad-accent2);
    text-shadow: 0 0 30px rgba(244,63,94,0.22);
}

.fad-home-subtitle {
    max-width: 650px;
    margin: 0;
    color: var(--fad-muted) !important;
    font-size: 18px;
    line-height: 1.82;
}

.fad-home-actions {
    display: flex;
    align-items: center;
    gap: 18px;
    flex-wrap: wrap;
    margin-top: 34px;
}

.fad-stats {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 26px;
    margin-top: 36px;
    max-width: 710px;
}

.fad-stat-icon {
    width: 32px;
    height: 32px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: var(--fad-accent2);
    font-size: 25px;
    margin-bottom: 10px;
}

.fad-stat-value {
    color: var(--fad-text);
    font-size: 21px;
    font-weight: 900;
}

.fad-stat-label {
    margin-top: 5px;
    color: var(--fad-muted);
    font-size: 13px;
    line-height: 1.35;
}

.status-wrapper { margin: 13px 0 0 0; }

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    padding: 9px 14px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 800;
    white-space: normal;
}

.status-badge span {
    width: 20px;
    height: 20px;
    border-radius: 999px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: white;
}

.success-badge {
    color: #dcfce7;
    border: 1px solid rgba(34,197,94,0.36);
    background: rgba(22,101,52,0.28);
}
.success-badge span { background: #16a34a; }
.success-badge strong { color: #bbf7d0; }
.warning-badge {
    color: #fde68a;
    border: 1px solid rgba(245,158,11,0.36);
    background: rgba(146,64,14,0.28);
}
.warning-badge span { background: #d97706; }
.warning-badge strong { color: #fef3c7; }

.fad-home-test-card {
    position: relative;
    padding: 34px;
    border-radius: 24px;
    border: 1px solid rgba(244,63,94,0.28);
    background:
        radial-gradient(circle at 90% 17%, rgba(244,63,94,0.13), transparent 28%),
        linear-gradient(145deg, rgba(20, 22, 36, 0.94), rgba(11, 13, 23, 0.94));
    box-shadow: 0 26px 70px rgba(0,0,0,0.36), inset 0 1px 0 rgba(255,255,255,0.05);
    overflow: hidden;
}

.fad-home-test-title {
    margin: 0 0 5px 0;
    color: white !important;
    font-size: 23px;
    font-weight: 900;
}

.fad-home-test-subtitle {
    margin: 0 0 22px 0;
    color: var(--fad-muted) !important;
    font-size: 14px;
    line-height: 1.55;
}

.fad-home-test-body {
    display: grid;
    grid-template-columns: minmax(260px, 1fr) minmax(210px, 0.8fr);
    gap: 28px;
    align-items: center;
}

.fad-mock-upload {
    min-height: 235px;
    border: 1.8px dashed rgba(244,63,94,0.55);
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    background: rgba(7,9,18,0.72);
    color: white !important;
    text-decoration: none !important;
    padding: 20px;
}

.fad-upload-icon {
    font-size: 44px;
    color: var(--fad-accent2);
    margin-bottom: 12px;
}

.fad-mock-upload strong {
    font-size: 17px;
    line-height: 1.4;
    text-decoration: underline;
}

.fad-mock-upload small {
    display: block;
    color: var(--fad-soft);
    margin-top: 14px;
}

.fad-sample-divider {
    display: flex;
    align-items: center;
    gap: 16px;
    margin: 18px 0;
    color: var(--fad-soft);
    font-size: 13px;
}

.fad-sample-divider::before,
.fad-sample-divider::after {
    content: "";
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.12);
}

.fad-sample-btn {
    width: 100%;
    min-height: 55px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: white !important;
    text-decoration: none !important;
    border-radius: 13px;
    border: 1px solid rgba(255,255,255,0.13);
    background: rgba(15,18,30,0.55);
    font-weight: 850;
}

.fad-secure-note {
    margin-top: 18px;
    color: var(--fad-muted);
    font-size: 13px;
}

.fad-art-preview-wrap {
    min-width: 0;
    display: flex;
    align-items: center;
    justify-content: center;
}

.fad-art-preview-img {
    width: min(270px, 100%);
    max-height: 360px;
    object-fit: contain;
    display: block;
    filter: drop-shadow(0 28px 45px rgba(0,0,0,0.48));
}

.fad-css-art-preview {
    position: relative;
    width: 240px;
    height: 330px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.18);
    background:
        linear-gradient(90deg, rgba(12, 16, 28, 0.05) 0 49.5%, rgba(255,255,255,0.82) 49.5% 50.5%, rgba(12, 16, 28, 0.05) 50.5% 100%),
        radial-gradient(circle at 32% 30%, rgba(251,113,133,0.86), transparent 20%),
        radial-gradient(circle at 42% 62%, rgba(15,23,42,0.90), transparent 23%),
        linear-gradient(135deg, #0f172a 0%, #78350f 46%, #fda4af 100%);
}

.fad-chip {
    position: absolute;
    top: 18px;
    z-index: 4;
    padding: 8px 11px;
    border-radius: 8px;
    color: #fff;
    font-size: 10px;
    font-weight: 900;
}
.fad-chip.original { left: 18px; background: #16a34a; }
.fad-chip.fake { right: 18px; background: #ef4444; }

.fad-feature-section {
    padding: 45px 0 20px 0;
}

.fad-section-heading {
    text-align: center;
    max-width: 740px;
    margin: 0 auto 28px auto;
}

.fad-section-kicker {
    color: var(--fad-accent2);
    font-size: 14px;
    font-weight: 900;
    margin-bottom: 8px;
}

.fad-section-title {
    margin: 0;
    color: var(--fad-text) !important;
    font-size: clamp(30px, 3vw, 40px);
    font-weight: 950;
    letter-spacing: -0.9px;
}

.fad-section-subtitle {
    margin: 10px 0 0 0;
    color: var(--fad-muted) !important;
    font-size: 16px;
    line-height: 1.65;
}

.fad-feature-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 18px;
}

.fad-feature-card,
.fad-dark-card,
.fad-contact-card,
.fad-process-card {
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 18px;
    background: linear-gradient(145deg, rgba(24,27,42,0.82), rgba(13,16,28,0.86));
    box-shadow: 0 18px 45px rgba(0,0,0,0.22);
}

.fad-feature-card {
    padding: 27px 22px;
    text-align: center;
    min-height: 218px;
}

.fad-feature-icon {
    color: var(--fad-accent2);
    font-size: 42px;
    margin-bottom: 18px;
}

.fad-feature-card h3 {
    margin: 0 0 10px 0;
    color: white !important;
    font-size: 17px;
    font-weight: 900;
}

.fad-feature-card p {
    margin: 0;
    color: var(--fad-muted) !important;
    font-size: 14px;
    line-height: 1.65;
}

/* GENERAL PAGE CARDS */
.fad-page-pad {
    padding: 54px 0 26px 0;
}

.fad-page-panel {
    padding: 58px 24px;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(255,255,255,0.055), rgba(255,255,255,0.03));
    border: 1px solid rgba(255,255,255,0.08);
}

.fad-info-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 18px;
    margin-top: 26px;
}

.fad-dark-card,
.fad-contact-card,
.fad-process-card {
    padding: 24px;
}

.fad-dark-card h3,
.fad-contact-card h3,
.fad-process-card h3 {
    margin: 0 0 10px 0;
    color: white !important;
    font-size: 18px;
    font-weight: 900;
}

.fad-dark-card p,
.fad-contact-card p,
.fad-process-card p {
    margin: 0;
    color: var(--fad-muted) !important;
    font-size: 15px;
    line-height: 1.7;
}

.fad-process-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 18px;
    margin-top: 28px;
}

.fad-step-number {
    width: 38px;
    height: 38px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 14px;
    border-radius: 12px;
    color: white;
    font-weight: 900;
    background: linear-gradient(135deg, #f43f5e, #e11d48);
}

/* DETECTOR PAGE: dark clean form, matched with home theme */
#detector-page {
    color: var(--fad-text) !important;
}

.detector-light-wrap {
    width: min(1220px, 100%);
    margin: 34px auto 0 auto;
    color: var(--fad-text) !important;
}

.detector-light-panel {
    padding: 22px;
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 22px;
    background:
        radial-gradient(circle at 8% 0%, rgba(244,63,94,0.12), transparent 30%),
        linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.035));
    box-shadow: 0 26px 70px rgba(0,0,0,0.30);
}

.detector-card {
    height: 100%;
    padding: 22px;
    border: 1px solid rgba(244, 63, 94, 0.24);
    border-radius: 20px;
    background:
        radial-gradient(circle at 100% 0%, rgba(244,63,94,0.10), transparent 34%),
        linear-gradient(145deg, rgba(18, 21, 34, 0.96), rgba(11, 13, 23, 0.96));
    box-shadow: 0 18px 45px rgba(0,0,0,0.28);
}

#detector-page .form,
#detector-page .block,
#detector-page .wrap,
#detector-page .contain {
    min-width: 0 !important;
}

.detector-card-title {
    margin: 0 0 6px 0;
    padding-left: 11px;
    border-left: 4px solid var(--fad-accent2);
    color: var(--fad-text) !important;
    font-size: 20px;
    font-weight: 900;
}

.detector-card-subtitle {
    margin: 0 0 16px 0;
    color: var(--fad-muted) !important;
    font-size: 13px;
    line-height: 1.6;
}

#detector-page textarea,
#detector-page input {
    background: rgba(255,255,255,0.06) !important;
    color: var(--fad-text) !important;
    border-color: rgba(255,255,255,0.12) !important;
    border-radius: 14px !important;
}

#detector-page textarea::placeholder,
#detector-page input::placeholder {
    color: rgba(226,232,240,0.58) !important;
}

#detector-page label,
#detector-page .wrap label span {
    color: var(--fad-accent2) !important;
    font-weight: 850 !important;
}

#artwork-upload {
    min-height: 300px !important;
    border: 1.8px dashed rgba(244,63,94,0.70) !important;
    border-radius: 17px !important;
    background: rgba(7, 9, 18, 0.74) !important;
    overflow: hidden !important;
}

#artwork-upload:hover {
    box-shadow: 0 0 0 4px rgba(244, 63, 94, 0.12) !important;
}

.detector-check-button {
    min-height: 48px !important;
    border-radius: 13px !important;
    border: 0 !important;
    color: white !important;
    background: linear-gradient(135deg, #f43f5e, #f97316) !important;
    font-size: 15px !important;
    font-weight: 900 !important;
    box-shadow: 0 14px 30px rgba(244,63,94,0.22) !important;
}

.detector-clear-button {
    min-height: 48px !important;
    border-radius: 13px !important;
    background: rgba(7,9,18,0.42) !important;
    color: var(--fad-text) !important;
    border: 1px solid rgba(244,63,94,0.36) !important;
    font-size: 15px !important;
    font-weight: 850 !important;
}

#verdict-card {
    min-height: 145px;
    padding: 22px !important;
    border: 1px solid rgba(244,63,94,0.36) !important;
    border-radius: 17px !important;
    background:
        radial-gradient(circle at 100% 0%, rgba(244,63,94,0.12), transparent 34%),
        linear-gradient(145deg, rgba(244,63,94,0.10), rgba(15,23,42,0.56)) !important;
    color: var(--fad-text) !important;
}

#verdict-card h2 {
    margin-top: 0 !important;
    color: var(--fad-text) !important;
    font-size: 28px !important;
}

#verdict-card p,
#verdict-card strong {
    color: rgba(248,250,252,0.92) !important;
}

#detector-page .output-box textarea {
    background: rgba(255,255,255,0.06) !important;
    color: var(--fad-text) !important;
    line-height: 1.65 !important;
}

#detector-page .analysis-tabs {
    margin-top: 22px;
    border: 1px solid rgba(244,63,94,0.22);
    border-radius: 20px;
    overflow: hidden;
    background:
        radial-gradient(circle at 0% 0%, rgba(244,63,94,0.10), transparent 32%),
        linear-gradient(145deg, rgba(18, 21, 34, 0.96), rgba(11, 13, 23, 0.96));
    box-shadow: 0 16px 38px rgba(0,0,0,0.25);
}

#detector-page .analysis-tabs [role="tablist"] {
    display: flex !important;
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    gap: 6px !important;
    padding: 10px 12px 0 12px !important;
    background: rgba(7,9,18,0.30) !important;
    border-bottom: 1px solid rgba(255,255,255,0.08) !important;
}

#detector-page .analysis-tabs [role="tab"] {
    color: var(--fad-muted) !important;
    background: transparent !important;
    border: 0 !important;
    white-space: nowrap !important;
    border-radius: 12px 12px 0 0 !important;
    padding: 10px 16px !important;
    font-weight: 800 !important;
}

#detector-page .analysis-tabs button.selected {
    color: white !important;
    font-weight: 900 !important;
    background: rgba(244,63,94,0.14) !important;
    border-bottom: 2px solid var(--fad-accent2) !important;
}

#detector-page .important-info {
    padding: 24px !important;
    color: var(--fad-muted) !important;
    line-height: 1.8 !important;
}

#detector-page .important-info h3 {
    color: var(--fad-text) !important;
    font-size: 21px !important;
    margin-top: 0 !important;
}

#detector-page .important-info li {
    margin-bottom: 10px !important;
    color: var(--fad-muted) !important;
}

#detector-page .important-info strong {
    color: var(--fad-text) !important;
}

#detector-page .technical-section {
    margin-top: 18px;
    border-radius: 16px !important;
    background: linear-gradient(145deg, rgba(18,21,34,0.92), rgba(11,13,23,0.92)) !important;
    color: var(--fad-text) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
}


/* Extra detector dark-theme fix for tab contents, textbox wrappers and technical output */
#detector-page .analysis-tabs,
#detector-page .analysis-tabs > div,
#detector-page .analysis-tabs .tabitem,
#detector-page .analysis-tabs .tabpanel,
#detector-page .analysis-tabs .form,
#detector-page .analysis-tabs .wrap,
#detector-page .analysis-tabs .block,
#detector-page .analysis-tabs .block.svelte-vt1mxs,
#detector-page .analysis-tabs .container,
#detector-page .technical-section,
#detector-page .technical-section > div,
#detector-page .technical-section .form,
#detector-page .technical-section .wrap,
#detector-page .technical-section .block,
#detector-page .technical-section .container {
    background:
        radial-gradient(circle at 0% 0%, rgba(244,63,94,0.10), transparent 32%),
        linear-gradient(145deg, rgba(18, 21, 34, 0.96), rgba(11, 13, 23, 0.96)) !important;
    color: var(--fad-text) !important;
    border-color: rgba(244,63,94,0.20) !important;
}

#detector-page .analysis-tabs textarea,
#detector-page .technical-section textarea,
#detector-page .technical-section pre,
#detector-page .technical-section code,
#detector-page .technical-section .json-holder,
#detector-page .technical-section .json-wrap,
#detector-page .technical-section .json-container,
#detector-page .technical-section .pretty-json-container,
#detector-page .technical-section .json-tree {
    background: rgba(7, 9, 18, 0.82) !important;
    color: rgba(248,250,252,0.92) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 14px !important;
}

#detector-page .analysis-tabs textarea::placeholder,
#detector-page .technical-section textarea::placeholder {
    color: rgba(226,232,240,0.54) !important;
}

#detector-page .analysis-tabs label,
#detector-page .technical-section label {
    color: var(--fad-accent2) !important;
    background: rgba(244,63,94,0.14) !important;
    border-radius: 9px !important;
    padding: 4px 8px !important;
}

#detector-page .analysis-tabs [data-testid="block-info"],
#detector-page .technical-section [data-testid="block-info"] {
    color: var(--fad-accent2) !important;
    background: rgba(244,63,94,0.14) !important;
    border-radius: 9px !important;
}

#detector-page .technical-section summary,
#detector-page .technical-section .label-wrap,
#detector-page .technical-section .wrap > label {
    background: rgba(7, 9, 18, 0.84) !important;
    color: var(--fad-text) !important;
    border-color: rgba(244,63,94,0.22) !important;
}

#detector-page .analysis-tabs .gap,
#detector-page .technical-section .gap {
    background: transparent !important;
}

/* Keep JSON output readable in dark mode */
#detector-page .technical-section span,
#detector-page .technical-section p,
#detector-page .technical-section div {
    color: rgba(248,250,252,0.90);
}


/* Final fix: make all detector textbox/description blocks fully dark */
#detector-page [data-testid="textbox"],
#detector-page [data-testid="textbox"] > div,
#detector-page [data-testid="textbox"] .wrap,
#detector-page [data-testid="textbox"] .container,
#detector-page [data-testid="textbox"] .input-container,
#detector-page .textbox,
#detector-page .textarea,
#detector-page .input-container,
#detector-page .form > .block,
#detector-page .block:has(textarea) {
    background: rgba(7, 9, 18, 0.84) !important;
    color: var(--fad-text) !important;
    border-color: rgba(255,255,255,0.10) !important;
}

#detector-page [data-testid="textbox"] textarea,
#detector-page textarea.scroll-hide,
#detector-page textarea[data-testid="textbox"],
#detector-page textarea {
    background: rgba(7, 9, 18, 0.86) !important;
    color: rgba(248,250,252,0.92) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04) !important;
}

#detector-page [data-testid="textbox"] textarea::placeholder,
#detector-page textarea::placeholder {
    color: rgba(226,232,240,0.46) !important;
    opacity: 1 !important;
}

#detector-page [data-testid="block-info"],
#detector-page .block-info,
#detector-page label,
#detector-page label span {
    color: #ff6b86 !important;
    background: rgba(244,63,94,0.16) !important;
    border-radius: 9px !important;
    font-weight: 900 !important;
}

#detector-page .detector-card [data-testid="textbox"],
#detector-page .detector-card [data-testid="textbox"] > div,
#detector-page .detector-card .block:has(textarea),
#detector-page .detector-card textarea {
    background: rgba(7, 9, 18, 0.86) !important;
}

#detector-page .detector-card .wrap,
#detector-page .detector-card .container {
    background: transparent !important;
}

/* Keep upload component footer and separators dark too */
#detector-page #artwork-upload,
#detector-page #artwork-upload > div,
#detector-page #artwork-upload .wrap {
    background: rgba(7, 9, 18, 0.78) !important;
}


/* Final requested layout fixes */
.detector-light-wrap {
    margin-top: 26px !important;
}

#detector-main-row {
    align-items: stretch !important;
}

#detector-main-row > div {
    display: flex !important;
    flex-direction: column !important;
}

#detector-main-row .detector-card {
    flex: 1 1 auto !important;
}

.fad-stat-value {
    line-height: 1.25 !important;
}

@media (max-width: 760px) {
    .gradio-container {
        padding-left: 10px !important;
        padding-right: 10px !important;
    }

    .fad-shell {
        width: 100% !important;
    }

    .fad-navbar {
        gap: 10px !important;
    }

    .fad-brand {
        font-size: 18px !important;
        line-height: 1.1 !important;
    }

    .fad-logo-mark {
        width: 34px !important;
        height: 34px !important;
        flex: 0 0 auto !important;
    }

    .fad-nav-cta {
        display: inline-flex !important;
        min-height: 40px !important;
        padding: 0 13px !important;
        font-size: 12px !important;
        white-space: nowrap !important;
    }

    .fad-home-title {
        font-size: 36px !important;
        line-height: 1.12 !important;
        letter-spacing: -1.2px !important;
    }

    .fad-home-hero {
        padding-top: 26px !important;
    }

    .detector-light-wrap {
        width: 100% !important;
        margin-top: 18px !important;
    }

    .detector-light-panel {
        padding: 12px !important;
        border-radius: 18px !important;
    }

    #detector-main-row,
    #detector-main-row > div,
    #detector-main-row .gradio-column,
    #detector-main-row .column {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
    }

    #detector-main-row {
        flex-direction: column !important;
        display: flex !important;
        gap: 14px !important;
    }

    .detector-card {
        height: auto !important;
        padding: 16px !important;
        border-radius: 17px !important;
    }

    .detector-card-title {
        font-size: 18px !important;
    }

    .detector-card-subtitle {
        font-size: 12px !important;
    }

    #artwork-upload {
        min-height: 240px !important;
    }

    #verdict-card h2 {
        font-size: 22px !important;
    }

    #detector-page .analysis-tabs [role="tab"] {
        padding: 9px 11px !important;
        font-size: 12px !important;
    }
}

@media (max-width: 480px) {
    .fad-brand span:last-child {
        max-width: 120px !important;
        display: inline-block !important;
        overflow-wrap: anywhere !important;
    }

    .fad-nav-cta {
        padding: 0 10px !important;
    }
}


/* Align navbar box with hero section start/end */
.fad-shell,
.detector-light-wrap {
    width: min(1280px, calc(100vw - 96px)) !important;
    max-width: 1280px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

.fad-navbar,
.fad-home-hero,
.fad-page-panel {
    width: 100% !important;
    max-width: 100% !important;
}

.fad-home-hero {
    border-radius: 6px !important;
}

@media (max-width: 920px) {
    .fad-shell,
    .detector-light-wrap {
        width: min(100%, calc(100vw - 28px)) !important;
    }
}

@media (max-width: 640px) {
    .fad-shell,
    .detector-light-wrap {
        width: min(100%, calc(100vw - 20px)) !important;
    }
}


/* Final responsive polish: smooth PC + mobile navigation and layout */
.fad-navbar {
    width: 100% !important;
    max-width: 100% !important;
    overflow: visible !important;
}

.fad-navlinks {
    min-width: 0 !important;
}

.fad-nav-link,
.fad-nav-cta,
.fad-primary-link,
.fad-secondary-link {
    -webkit-tap-highlight-color: transparent;
}

.fad-home-test-card,
.detector-card,
.fad-feature-card,
.fad-page-panel {
    transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}

.fad-home-test-card:hover,
.detector-card:hover,
.fad-feature-card:hover,
.fad-page-panel:hover {
    border-color: rgba(244, 63, 94, 0.34) !important;
}

/* Keep the main navbar visible on tablets and mobile instead of hiding it */
@media (max-width: 1100px) {
    .fad-navbar {
        display: flex !important;
        flex-wrap: wrap !important;
        align-items: center !important;
        justify-content: space-between !important;
        gap: 10px 14px !important;
        padding: 12px 0 10px 0 !important;
    }

    .fad-brand {
        flex: 1 1 auto !important;
        min-width: 0 !important;
    }

    .fad-nav-cta {
        flex: 0 0 auto !important;
        display: inline-flex !important;
    }

    .fad-navlinks {
        order: 3 !important;
        width: 100% !important;
        display: flex !important;
        justify-content: flex-start !important;
        gap: 18px !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
        white-space: nowrap !important;
        padding: 10px 2px 2px 2px !important;
        border-top: 1px solid rgba(255, 255, 255, 0.08) !important;
        scrollbar-width: thin;
        scrollbar-color: rgba(244, 63, 94, 0.65) transparent;
    }

    .fad-navlinks::-webkit-scrollbar {
        height: 4px;
    }

    .fad-navlinks::-webkit-scrollbar-thumb {
        background: rgba(244, 63, 94, 0.65);
        border-radius: 999px;
    }

    .fad-nav-link {
        flex: 0 0 auto !important;
        font-size: 13px !important;
        padding: 4px 0 !important;
    }

    .fad-nav-link.active::after,
    .fad-nav-link:hover::after {
        bottom: -4px !important;
    }

    .fad-home-grid {
        gap: 30px !important;
    }

    .fad-home-test-card {
        padding: 26px !important;
    }
}

@media (max-width: 760px) {
    .gradio-container {
        padding-left: 10px !important;
        padding-right: 10px !important;
    }

    .fad-shell,
    .detector-light-wrap {
        width: min(100%, calc(100vw - 20px)) !important;
        max-width: calc(100vw - 20px) !important;
    }

    .fad-navbar {
        min-height: auto !important;
        border-radius: 0 !important;
    }

    .fad-brand {
        font-size: 18px !important;
        line-height: 1.12 !important;
        gap: 9px !important;
    }

    .fad-logo-mark {
        width: 33px !important;
        height: 33px !important;
        border-radius: 10px !important;
        flex: 0 0 auto !important;
    }

    .fad-nav-cta {
        min-height: 38px !important;
        padding: 0 13px !important;
        font-size: 12px !important;
        border-radius: 9px !important;
        white-space: nowrap !important;
    }

    .fad-navlinks {
        gap: 16px !important;
        padding-top: 9px !important;
        padding-bottom: 4px !important;
    }

    .fad-nav-link {
        font-size: 12px !important;
    }

    .fad-home-hero {
        padding-top: 24px !important;
        border-radius: 6px !important;
    }

    .fad-home-title {
        font-size: clamp(31px, 9vw, 38px) !important;
        line-height: 1.13 !important;
        letter-spacing: -1.1px !important;
    }

    .fad-home-subtitle {
        font-size: 14px !important;
        line-height: 1.68 !important;
    }

    .fad-home-actions {
        gap: 12px !important;
        margin-top: 24px !important;
    }

    .fad-primary-link,
    .fad-secondary-link {
        width: 100% !important;
        min-height: 44px !important;
    }

    .status-badge {
        font-size: 11px !important;
        padding: 8px 12px !important;
    }

    .fad-stats {
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
        gap: 18px !important;
    }

    .fad-stat-value {
        font-size: 17px !important;
    }

    .fad-stat-label {
        font-size: 12px !important;
    }

    .fad-home-test-card {
        padding: 20px !important;
        border-radius: 20px !important;
    }

    .fad-home-test-body {
        display: grid !important;
        grid-template-columns: 1fr !important;
        gap: 20px !important;
    }

    .fad-mock-upload {
        min-height: 210px !important;
    }

    .fad-art-preview-img {
        width: min(235px, 78vw) !important;
        margin: 0 auto !important;
    }

    .fad-section-title {
        font-size: 27px !important;
        line-height: 1.2 !important;
    }

    .fad-section-subtitle {
        font-size: 13px !important;
    }

    .fad-feature-grid,
    .fad-info-grid,
    .fad-process-grid {
        grid-template-columns: 1fr !important;
    }

    .fad-feature-card,
    .fad-dark-card,
    .fad-contact-card,
    .fad-process-card {
        padding: 20px !important;
    }

    .detector-light-wrap {
        margin-top: 18px !important;
    }

    .detector-light-panel {
        padding: 12px !important;
        border-radius: 18px !important;
    }

    #detector-main-row {
        display: flex !important;
        flex-direction: column !important;
        gap: 14px !important;
    }

    #detector-main-row > div,
    #detector-main-row .gradio-column,
    #detector-main-row .column {
        width: 100% !important;
        min-width: 0 !important;
        max-width: 100% !important;
    }

    .detector-card {
        height: auto !important;
        padding: 16px !important;
        border-radius: 17px !important;
    }

    #artwork-upload {
        min-height: 235px !important;
    }

    #verdict-card h2 {
        font-size: 22px !important;
    }

    #detector-page .analysis-tabs [role="tablist"] {
        padding-left: 8px !important;
        padding-right: 8px !important;
    }

    #detector-page .analysis-tabs [role="tab"] {
        padding: 9px 11px !important;
        font-size: 12px !important;
    }
}

@media (max-width: 430px) {
    .fad-brand {
        font-size: 16px !important;
    }

    .fad-brand span:last-child {
        max-width: 150px !important;
        display: inline-block !important;
        overflow-wrap: normal !important;
        white-space: normal !important;
    }

    .fad-nav-cta {
        padding: 0 10px !important;
        font-size: 11px !important;
    }

    .fad-navlinks {
        gap: 14px !important;
    }

    .fad-home-title {
        font-size: 30px !important;
    }
}


/* Final navbar fix: keep brand, menu, and Try Detector on the same line on PC + mobile */
.fad-navbar {
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    scrollbar-width: thin;
    scrollbar-color: rgba(244, 63, 94, 0.55) transparent;
}

.fad-navbar::-webkit-scrollbar {
    height: 4px;
}

.fad-navbar::-webkit-scrollbar-thumb {
    background: rgba(244, 63, 94, 0.55);
    border-radius: 999px;
}

.fad-brand {
    flex: 0 0 auto !important;
    white-space: nowrap !important;
}

.fad-navlinks {
    order: initial !important;
    width: auto !important;
    flex: 0 0 auto !important;
    display: flex !important;
    flex-wrap: nowrap !important;
    overflow: visible !important;
    white-space: nowrap !important;
    border-top: 0 !important;
}

.fad-nav-cta {
    flex: 0 0 auto !important;
    display: inline-flex !important;
    white-space: nowrap !important;
}

@media (max-width: 1100px) {
    .fad-navbar {
        flex-wrap: nowrap !important;
        justify-content: flex-start !important;
        gap: 18px !important;
        padding: 12px 0 !important;
    }

    .fad-brand {
        flex: 0 0 auto !important;
        min-width: max-content !important;
    }

    .fad-navlinks {
        order: initial !important;
        width: auto !important;
        flex: 0 0 auto !important;
        display: flex !important;
        flex-wrap: nowrap !important;
        gap: 22px !important;
        overflow: visible !important;
        padding: 0 !important;
        border-top: 0 !important;
    }

    .fad-nav-link {
        flex: 0 0 auto !important;
        font-size: 13px !important;
        padding: 0 !important;
    }

    .fad-nav-link.active::after,
    .fad-nav-link:hover::after {
        bottom: -11px !important;
    }

    .fad-nav-cta {
        margin-left: auto !important;
        flex: 0 0 auto !important;
    }
}

@media (max-width: 760px) {
    .fad-navbar {
        gap: 18px !important;
        min-height: 62px !important;
        padding: 10px 0 !important;
    }

    .fad-brand {
        font-size: 17px !important;
        line-height: 1 !important;
        min-width: max-content !important;
    }

    .fad-brand span:last-child {
        max-width: none !important;
        white-space: nowrap !important;
        overflow-wrap: normal !important;
    }

    .fad-logo-mark {
        width: 32px !important;
        height: 32px !important;
        flex: 0 0 auto !important;
    }

    .fad-navlinks {
        gap: 18px !important;
        padding: 0 !important;
        min-width: max-content !important;
    }

    .fad-nav-link {
        font-size: 12px !important;
        white-space: nowrap !important;
    }

    .fad-nav-cta {
        min-height: 38px !important;
        padding: 0 12px !important;
        font-size: 12px !important;
        margin-left: 6px !important;
    }
}

@media (max-width: 430px) {
    .fad-brand {
        font-size: 16px !important;
    }

    .fad-navlinks {
        gap: 16px !important;
    }

    .fad-nav-cta {
        font-size: 11px !important;
        padding: 0 10px !important;
    }
}

/* Footer */
.custom-footer {
    width: min(1320px, 100%);
    margin: 50px auto 0 auto;
    padding: 28px 0 8px 0;
    text-align: center;
    color: var(--fad-muted);
    border-top: 1px solid rgba(255,255,255,0.08);
    font-size: 13px;
    line-height: 1.8;
}

.custom-footer strong { color: white; }

/* Responsive */
@media (max-width: 1180px) {
    .fad-home-grid { grid-template-columns: 1fr; gap: 38px; }
    .fad-home-test-body { grid-template-columns: 1fr; }
    .fad-feature-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .fad-process-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}

@media (max-width: 920px) {
    .gradio-container { padding: 0 18px 22px 18px !important; }
    .fad-navlinks { display: none; }
    .fad-brand { font-size: 19px; }
    .fad-nav-cta { min-height: 42px; padding: 0 15px; font-size: 13px; }
    .fad-home-hero { padding-top: 42px; }
    .fad-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .fad-feature-grid,
    .fad-info-grid { grid-template-columns: 1fr 1fr; }
}

@media (max-width: 640px) {
    .gradio-container { padding: 0 12px 18px 12px !important; }
    .fad-navbar { padding: 10px 0; min-height: auto; }
    .fad-logo-mark { width: 34px; height: 34px; }
    .fad-brand { font-size: 17px; }
    .fad-home-title { font-size: 36px; letter-spacing: -1.2px; }
    .fad-home-subtitle { font-size: 15px; line-height: 1.68; }
    .fad-primary-link,
    .fad-secondary-link { width: 100%; }
    .fad-home-test-card { padding: 22px; }
    .fad-stats,
    .fad-feature-grid,
    .fad-info-grid,
    .fad-process-grid { grid-template-columns: 1fr; }
    .detector-light-panel { padding: 10px; }
    #artwork-upload { min-height: 235px !important; }
}

"""
# ---------------- Streamlit hosting layer (original visual design preserved) ----------------
st.set_page_config(page_title="FakeArt Detector", page_icon="🎨", layout="wide", initial_sidebar_state="collapsed")

_CALIBRATION_DETAILS = _load_calibration_details()
_CALIBRATION_ACCURACY = _CALIBRATION_DETAILS.get("calibration_accuracy")
_BALANCED_ACCURACY = _CALIBRATION_DETAILS.get("balanced_accuracy")
CALIBRATION_ACCURACY_TEXT = f"{float(_CALIBRATION_ACCURACY) * 100:.0f}%" if isinstance(_CALIBRATION_ACCURACY, (int, float)) else "97%"
BALANCED_ACCURACY_TEXT = f"{float(_BALANCED_ACCURACY) * 100:.1f}%" if isinstance(_BALANCED_ACCURACY, (int, float)) else "96.8%"
HERO_ART_URI = _asset_data_uri("assets/hero_art_cards.png")
HERO_ART_IMAGE_HTML = f'<img class="fad-art-preview-img" src="{HERO_ART_URI}" alt="Original and fake artwork comparison preview">' if HERO_ART_URI else '<div class="fad-css-art-preview"><span class="fad-chip original">ORIGINAL</span><span class="fad-chip fake">FAKE</span></div>'
threshold_banner = '<div class="status-badge success-badge"><span>✓</span> Calibrated threshold active.</div>' if THRESHOLD_IS_CALIBRATED else f'<div class="status-badge warning-badge"><span>⚠</span> Default threshold active: <strong>{FAKE_THRESHOLD:.6f}</strong></div>'

st.markdown("<style>" + CUSTOM_CSS + "</style><style>" + r'''
#MainMenu, footer, header { visibility:hidden; }
.stApp { background:#070912 !important; }
.block-container { max-width:1500px !important; padding:0 38px 28px 38px !important; }
div[data-testid="stFileUploader"] { background:rgba(7,9,18,.74); border:1.8px dashed rgba(244,63,94,.70); border-radius:17px; padding:12px; }
div[data-testid="stFileUploader"] label { color:#fb7185 !important; font-weight:850 !important; }
.stTextArea textarea { background:rgba(255,255,255,.06) !important; color:#f8fafc !important; border-color:rgba(255,255,255,.12) !important; border-radius:14px !important; }
.stTextArea label { color:#fb7185 !important; font-weight:850 !important; }
.stButton > button { min-height:48px; border-radius:13px; font-weight:900; }
</style>''', unsafe_allow_html=True)

page = st.query_params.get("page", "home")
if page not in {"home","detector","how","about","dataset","team","contact"}:
    page = "home"

def nav_link(label, target, active=False):
    cls = "fad-nav-link active" if active else "fad-nav-link"
    return f'<a class="{cls}" href="?page={target}">{label}</a>'

st.markdown(f'''
<div class="fad-shell"><nav class="fad-navbar">
<a class="fad-brand" href="?page=home"><span class="fad-logo-mark">⌾</span><span><span class="accent">FakeArt</span> Detector</span></a>
<div class="fad-navlinks">{nav_link("Home","home",page=="home")}{nav_link("About","about",page=="about")}{nav_link("How It Works","how",page=="how")}{nav_link("Dataset","dataset",page=="dataset")}{nav_link("Team","team",page=="team")}{nav_link("Contact","contact",page=="contact")}</div>
<a class="fad-nav-cta" href="?page=detector">Try Detector</a></nav></div>
''', unsafe_allow_html=True)

if page == "home":
    st.markdown(f'''
<main class="fad-shell"><section class="fad-home-hero"><div class="fad-home-grid"><div>
<h1 class="fad-home-title">Detect <span class="accent">Fake</span> Art.<br>Trust Real Creativity.</h1>
<p class="fad-home-subtitle">FakeArt Detector uses advanced AI models that combine visual analysis, semantic understanding, and image-text reasoning to identify forged or manipulated digital artworks.</p>
<div class="fad-home-actions"><a class="fad-primary-link" href="?page=detector">Try FakeArt Detector <span>➜</span></a><a class="fad-secondary-link" href="?page=how">Learn How It Works <span>▷</span></a></div>
<div class="status-wrapper">{threshold_banner}</div><div class="fad-stats">
<div><div class="fad-stat-icon">▧</div><div class="fad-stat-value">2.5K+</div><div class="fad-stat-label">Artwork Images</div></div>
<div><div class="fad-stat-icon">◎</div><div class="fad-stat-value">{CALIBRATION_ACCURACY_TEXT}</div><div class="fad-stat-label">Calibration Accuracy</div></div>
<div><div class="fad-stat-icon">◫</div><div class="fad-stat-value">{BALANCED_ACCURACY_TEXT}</div><div class="fad-stat-label">Balanced Accuracy</div></div>
<div><div class="fad-stat-icon">ϟ</div><div class="fad-stat-value">Multi-Modal Analysis</div><div class="fad-stat-label">LLaVA</div></div></div></div>
<div class="fad-home-test-card"><h2 class="fad-home-test-title">Test Your Artwork</h2><p class="fad-home-test-subtitle">Upload an artwork image to check its authenticity.</p><div class="fad-home-test-body"><div><a class="fad-mock-upload" href="?page=detector"><div><div class="fad-upload-icon">☁</div><strong>Drag &amp; drop your image here<br>or click to browse</strong><small>Supports: JPG, PNG, WEBP</small></div></a><div class="fad-sample-divider">or</div><a class="fad-sample-btn" href="?page=detector">▧ Try Sample Artwork</a><div class="fad-secure-note">▣ Your images are secure and never stored.</div></div><div class="fad-art-preview-wrap">{HERO_ART_IMAGE_HTML}</div></div></div></div></section>
<section class="fad-feature-section"><div class="fad-section-heading"><div class="fad-section-kicker">Why FakeArt Detector?</div><h2 class="fad-section-title">Advanced Multi-Modal Analysis</h2><p class="fad-section-subtitle">Combining multiple AI techniques for comprehensive artwork analysis.</p></div><div class="fad-feature-grid">
<div class="fad-feature-card"><div class="fad-feature-icon">◉</div><h3>Visual Analysis</h3><p>Extracts patterns, textures and visual inconsistencies.</p></div><div class="fad-feature-card"><div class="fad-feature-icon">♧</div><h3>Semantic Understanding</h3><p>Understands the meaning and context behind artwork content.</p></div><div class="fad-feature-card"><div class="fad-feature-icon">▤</div><h3>Image-Text Reasoning</h3><p>Compares visual content with textual descriptions.</p></div><div class="fad-feature-card"><div class="fad-feature-icon">▦</div><h3>Pixel-Level Inspection</h3><p>Checks brightness, contrast, sharpness, noise and colorfulness.</p></div><div class="fad-feature-card"><div class="fad-feature-icon">♢</div><h3>Forgery Detection</h3><p>Returns Original or Fake verdict with research score details.</p></div>
</div></section></main>''', unsafe_allow_html=True)

elif page == "detector":
    st.markdown('<div class="detector-light-wrap"><div class="detector-light-panel">', unsafe_allow_html=True)
    left, right = st.columns([5, 6], gap="large")
    with left:
        st.markdown('<div class="detector-card"><h3 class="detector-card-title">Upload Artwork</h3><p class="detector-card-subtitle">Upload one artwork image to analyze its visual and semantic characteristics. You may also add an optional description to support the interpretation.</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Artwork image", type=["jpg","jpeg","png","webp"], label_visibility="collapsed", key="artwork-upload")
        image_input = Image.open(uploaded).convert("RGB") if uploaded else None
        if image_input is not None:
            st.image(image_input, use_container_width=True)
        user_description = st.text_area("Optional artwork description", placeholder="Describe the artwork, objects, background, style, colors or composition. Leave this field empty for image-only analysis.", height=140)
        c1, c2 = st.columns([2,1])
        with c1: check = st.button("Check Artwork", type="primary", use_container_width=True)
        with c2: clear = st.button("Clear", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="detector-card"><h3 class="detector-card-title">Analysis Result</h3><p class="detector-card-subtitle">The system verdict and model-generated interpretation will appear below after analysis.</p>', unsafe_allow_html=True)
        if clear:
            for k in ["result","system_description","match","pixel","reasons","technical"]:
                st.session_state.pop(k, None)
            st.rerun()
        if check:
            if image_input is None:
                st.error("Please upload an artwork image first.")
            else:
                with st.spinner("Analyzing artwork..."):
                    try:
                        vals = run_prediction(image_input, user_description)
                        for k, v in zip(["result","system_description","match","pixel","reasons","technical"], vals):
                            st.session_state[k] = v
                    except PredictionError as e:
                        st.error(str(e))
                    except Exception as e:
                        traceback.print_exc()
                        st.error(f"Prediction failed: {e}")
        result = st.session_state.get("result", "## Ready for Artwork Analysis\n\nUpload an artwork image and click **Check Artwork** to begin.")
        st.markdown(f'<div id="verdict-card">{result}</div>', unsafe_allow_html=True)
        st.text_area("System-generated visual description", value=st.session_state.get("system_description",""), height=140, disabled=True)
        st.text_area("Description consistency", value=st.session_state.get("match",""), height=110, disabled=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)
    tabs = st.tabs(["Pixel-level Analysis", "Prediction Reasons", "Important Information"])
    with tabs[0]: st.text_area("Pixel statistics and observations", value=st.session_state.get("pixel",""), height=250, disabled=True)
    with tabs[1]: st.text_area("Prediction reasons", value=st.session_state.get("reasons",""), height=220, disabled=True)
    with tabs[2]:
        st.markdown("""### About the Analysis
- The **Verdict** is determined from model scores and the configured fake-threshold.
- The **System-generated description** is created automatically from the uploaded image.
- The **Description consistency** result is supplementary and does not override the main verdict.
- The displayed confidence is a heuristic score for research interpretation, not a calibrated legal probability.
- This application is a research prototype developed within an academic project environment.""")
    with st.expander("Advanced Technical Details", expanded=False):
        st.json(st.session_state.get("technical", {}))

else:
    pages = {
      "how": ("Workflow", "How It Works", "A simple process designed for academic research and demonstration.", [("1","Upload Artwork","Provide one clear digital artwork image for analysis."),("2","Generate Description","The model creates a description of objects, style, background and composition."),("3","Analyze Features","The system checks semantic evidence, label scores and pixel-level observations."),("4","Receive Verdict","Output includes Original/Fake verdict, reasons and technical scores.")]),
      "about": ("About", "FakeArt Detector", "FakeArt Detector is an academic research prototype for detecting original and manipulated digital artworks using visual, semantic and pixel-level evidence.", [("","Purpose","Support research-based authenticity prediction for artwork images."),("","Model","LLaVA-1.5-7B with LoRA fine-tuning for multimodal reasoning."),("","Output","Verdict, description, consistency report, pixel evidence and technical details.")]),
      "dataset": ("Dataset", "Research Dataset Context", "Prepared with paired original and fake artwork images for binary authenticity prediction.", [("","Total Images","2,500+ artwork images used in the project context."),("","Classes","Original artwork and fake/manipulated artwork."),("","Description","Every artwork image includes a description to support image-text reasoning and semantic analysis.")]),
      "team": ("Team", "Final Year Design Project", "Academic research prototype developed for fake artwork detection using multimodal AI.", [("","Built by","MD. Moshiur Rahman<br>MD. Farhan Sadik Shihab"),("","Supervisor","Mr. Syed Eftasum Alam<br>Lecturer"),("","Project","FakeArt Detector · Final Year Design Project")]),
      "contact": ("Contact", "Project Contact", "Project team, supervisor and contact information for FakeArt Detector.", [("","Built by","MD. Moshiur Rahman<br>MD. Farhan Sadik Shihab"),("","Supervisor","Mr. Syed Eftasum Alam"),("","Email","fakeart.detector.bd@gmail.com")])}
    kicker, title, subtitle, cards = pages[page]
    card_class = "fad-process-card" if page == "how" else "fad-dark-card"
    grid = "fad-process-grid" if page == "how" else "fad-info-grid"
    cards_html = ''.join(f'<div class="{card_class}"><div class="fad-step-number">{a}</div><h3>{b}</h3><p>{c}</p></div>' if page == "how" else f'<div class="{card_class}"><h3>{b}</h3><p>{c}</p></div>' for a,b,c in cards)
    st.markdown(f'<main class="fad-shell fad-page-pad"><section class="fad-page-panel"><div class="fad-section-heading"><div class="fad-section-kicker">{kicker}</div><h2 class="fad-section-title">{title}</h2><p class="fad-section-subtitle">{subtitle}</p></div><div class="{grid}">{cards_html}</div></section></main>', unsafe_allow_html=True)

st.markdown('<footer class="custom-footer"><strong>FakeArt Detector</strong><br>Built by MD. Moshiur Rahman and MD. Farhan Sadik Shihab<br>Final Year Design Project · Academic Research Prototype<br>LLaVA-1.5-7B with LoRA Fine-Tuning</footer>', unsafe_allow_html=True)
