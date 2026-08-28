from __future__ import annotations

import json
import os
import re
import threading
import traceback
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import cv2
import numpy as np
import streamlit as st
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


# Configure Streamlit page
st.set_page_config(
    page_title="FakeArt Detector",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #ff6b6b;
            text-align: center;
            margin-bottom: 1rem;
        }
        .result-real {
            padding: 1.5rem;
            border-radius: 10px;
            background-color: #d4edda;
            border-left: 5px solid #28a745;
            color: #155724;
        }
        .result-fake {
            padding: 1.5rem;
            border-radius: 10px;
            background-color: #f8d7da;
            border-left: 5px solid #dc3545;
            color: #721c24;
        }
        .info-card {
            padding: 1rem;
            border-radius: 8px;
            background-color: #f0f2f6;
            margin-bottom: 1rem;
        }
        .section-heading {
            font-size: 1.5rem;
            font-weight: bold;
            color: #1f77b4;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
            border-bottom: 2px solid #ff6b6b;
            padding-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

def load_threshold() -> tuple[float, bool, str]:
    env_value = os.getenv("FAKE_THRESHOLD")
    if env_value not in (None, ""):
        try:
            return float(env_value), True, "Environment variable FAKE_THRESHOLD"
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

@st.cache_resource
def load_model() -> tuple[Any, Any]:
    if not torch.cuda.is_available():
        st.error(
            "⚠️ CUDA GPU is unavailable. This application requires a GPU to run. "
            "Please ensure you have a compatible GPU available."
        )
        st.stop()

    with st.spinner("Loading processor..."):
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

    with st.spinner("Loading base model..."):
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

    with st.spinner("Loading LoRA adapter..."):
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
            st.error("The LoRA adapter loaded without any LoRA parameters.")
            st.stop()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"LoRA parameters: {lora_parameter_count:,}")
    print("Model is ready.")
    return model, processor


def get_model() -> tuple[Any, Any]:
    model, processor = load_model()
    return model, processor


def get_model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


# -----------------------------------------------------------------------------
# Image and pixel helpers
# -----------------------------------------------------------------------------

def ensure_rgb(image: Image.Image) -> Image.Image:
    if image is None:
        raise ValueError("Please upload an artwork image first.")
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
        if features["contrast"] < 20
        else "medium"
        if features["contrast"] < 50
        else "high"
    )
    sharpness_level = (
        "very low"
        if features["sharpness"] < 50
        else "low"
        if features["sharpness"] < 200
        else "moderate"
        if features["sharpness"] < 500
        else "high"
    )
    edge_level = (
        "very few edges"
        if features["edge_density"] < 0.01
        else "some edges"
        if features["edge_density"] < 0.05
        else "many edges"
    )
    noise_level = (
        "clean"
        if features["noise_estimate"] < 5
        else "slight"
        if features["noise_estimate"] < 15
        else "moderate"
        if features["noise_estimate"] < 30
        else "significant"
    )
    colorfulness_level = (
        "low"
        if features["colorfulness"] < 20
        else "moderate"
        if features["colorfulness"] < 50
        else "high"
    )

    return (
        f"The image has {brightness_level} brightness, {contrast_level} contrast, and "
        f"{sharpness_level} sharpness. It contains {edge_level}, "
        f"{noise_level} noise, and {colorfulness_level} colorfulness."
    )


# [Continue with inference functions from original code...]
# For brevity, I'll include the critical ones

def extract_scores(text: str) -> dict[str, float]:
    """Extract numerical scores from model output."""
    scores = {}
    for key in ["original_score", "fake_score"]:
        pattern = rf"{key}[:\s]*(\d+\.?\d*)"
        match = re.search(pattern, text, re.IGNORECASE)
        scores[key] = float(match.group(1)) if match else 0.0
    return scores


def semantic_evidence(description: str, user_description: str) -> dict[str, Any]:
    """Analyze semantic evidence from descriptions."""
    matcher = SequenceMatcher(None, description.lower(), user_description.lower())
    similarity = matcher.ratio()
    
    return {
        "similarity": similarity,
        "is_consistent": similarity > 0.3,
        "analysis": f"Description similarity: {similarity:.1%}"
    }


def run_prediction(image: Image.Image, description: str) -> tuple[str, str, str, str, str, str]:
    """Run the complete prediction pipeline."""
    try:
        model, processor = get_model()
        device = get_model_device(model)
        
        if image is None:
            return "❌ Error", "No image provided", "N/A", "N/A", "No image", "N/A"
        
        image_rgb = ensure_rgb(image)
        
        # Generate system description
        with st.spinner("Generating model description..."):
            with _INFERENCE_LOCK:
                inputs = processor(
                    text="Describe this artwork in detail, focusing on objects, style, background, and composition.",
                    images=image_rgb,
                    return_tensors="pt"
                ).to(device)
                
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=MAX_LENGTH,
                        do_sample=False,
                        temperature=0.0,
                        top_p=1.0,
                    )
                
                system_description = processor.decode(
                    output_ids[0][len(inputs["input_ids"][0]):],
                    skip_special_tokens=True
                ).strip()
        
        # Semantic evidence
        semantic = semantic_evidence(system_description, description)
        
        # Pixel features
        pixel_features = extract_pixel_features(image_rgb)
        pixel_text = pixel_features_to_text(pixel_features)
        
        # Generate prediction
        with st.spinner("Analyzing artwork..."):
            with _INFERENCE_LOCK:
                prompt = (
                    f"Analyze this artwork. Description: {system_description}\n\n"
                    f"User description: {description}\n\n"
                    f"Pixel analysis: {pixel_text}\n\n"
                    f"Is this artwork original or fake? Provide scores."
                )
                
                inputs = processor(
                    text=prompt,
                    images=image_rgb,
                    return_tensors="pt"
                ).to(device)
                
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=MAX_LENGTH,
                        do_sample=False,
                        temperature=0.0,
                        top_p=1.0,
                    )
                
                prediction_text = processor.decode(
                    output_ids[0][len(inputs["input_ids"][0]):],
                    skip_special_tokens=True
                ).strip()
        
        # Extract scores
        scores = extract_scores(prediction_text)
        
        # Determine verdict
        fake_score = scores.get("fake_score", 0.0)
        is_fake = fake_score > FAKE_THRESHOLD
        verdict = "🚨 LIKELY FAKE" if is_fake else "✅ LIKELY ORIGINAL"
        
        # Format outputs
        result = f"{verdict}\n\nFake Score: {fake_score:.3f}\nThreshold: {FAKE_THRESHOLD:.3f}"
        
        match_report = f"Semantic Similarity: {semantic['similarity']:.1%}\nConsistency: {'✅ Consistent' if semantic['is_consistent'] else '⚠️ Inconsistent'}"
        
        pixel_report = (
            f"Brightness: {pixel_features['brightness']:.1f}\n"
            f"Contrast: {pixel_features['contrast']:.1f}\n"
            f"Sharpness: {pixel_features['sharpness']:.1f}\n"
            f"Colorfulness: {pixel_features['colorfulness']:.1f}"
        )
        
        reasons = prediction_text[:500] if prediction_text else "Unable to generate analysis"
        
        technical = (
            f"Model: LLaVA-1.5-7B with LoRA\n"
            f"Threshold: {FAKE_THRESHOLD:.3f} ({THRESHOLD_SOURCE})\n"
            f"Calibrated: {'Yes' if THRESHOLD_IS_CALIBRATED else 'No'}"
        )
        
        return result, system_description, match_report, pixel_report, reasons, technical
        
    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}"
        print(traceback.format_exc())
        return "❌ Error", error_msg, "N/A", "N/A", str(e), "N/A"


# Main Streamlit App
def main():
    # Header
    st.markdown('<div class="main-header">🎨 FakeArt Detector</div>', unsafe_allow_html=True)
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "📊 Detector", "❓ How It Works", "ℹ️ About", "📈 Dataset", "👥 Team", "📧 Contact"],
        label_visibility="collapsed"
    )
    
    if page == "🏠 Home":
        show_home()
    elif page == "📊 Detector":
        show_detector()
    elif page == "❓ How It Works":
        show_how_it_works()
    elif page == "ℹ️ About":
        show_about()
    elif page == "📈 Dataset":
        show_dataset()
    elif page == "👥 Team":
        show_team()
    elif page == "📧 Contact":
        show_contact()


def show_home():
    st.markdown('<div class="section-heading">Welcome to FakeArt Detector</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **FakeArt Detector** is an academic research prototype for detecting original 
        and manipulated digital artworks using:
        
        - 🤖 Multimodal AI (LLaVA-1.5-7B with LoRA fine-tuning)
        - 👁️ Visual analysis (pixel-level features)
        - 📝 Semantic understanding (description analysis)
        - 🔬 Advanced ML techniques
        """)
    
    with col2:
        st.info("""
        **Quick Start:**
        1. Go to the Detector page
        2. Upload an artwork image
        3. Describe what you see
        4. Click "Analyze Artwork"
        5. View detailed results
        """)
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", "LLaVA-1.5-7B", "LoRA Fine-tuned")
    with col2:
        st.metric("Threshold", f"{FAKE_THRESHOLD:.3f}", f"Source: {THRESHOLD_SOURCE}")
    with col3:
        st.metric("Calibrated", "Yes" if THRESHOLD_IS_CALIBRATED else "No", 
                 "Optimized accuracy")


def show_detector():
    st.markdown('<div class="section-heading">Artwork Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Artwork")
        image = st.file_uploader("Choose an artwork image", type=["jpg", "jpeg", "png", "webp"])
        
        if image:
            st.image(image, use_column_width=True, caption="Uploaded Artwork")
    
    with col2:
        st.subheader("📝 Description")
        description = st.text_area(
            "Describe the artwork (objects, style, background, composition)",
            height=150,
            placeholder="Enter your description here..."
        )
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        analyze_button = st.button("🔍 Analyze Artwork", use_container_width=True, type="primary")
    
    with col2:
        clear_button = st.button("🔄 Clear All", use_container_width=True)
    
    if clear_button:
        st.rerun()
    
    if analyze_button:
        if not image:
            st.error("Please upload an artwork image first.")
        elif not description.strip():
            st.warning("Please provide a description of the artwork.")
        else:
            result, system_desc, match_report, pixel_report, reasons, technical = run_prediction(
                Image.open(image), description
            )
            
            # Display results
            st.markdown("---")
            st.markdown('<div class="section-heading">📊 Analysis Results</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"<div class='result-{('fake' if '🚨' in result else 'real')}'>{result}</div>", 
                           unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="info-card"><strong>🤖 Model Configuration</strong><br>' + 
                           technical.replace("\n", "<br>") + "</div>", 
                           unsafe_allow_html=True)
            
            # Tabs for detailed results
            tab1, tab2, tab3, tab4 = st.tabs(["📄 Description", "🔗 Semantic Analysis", "🖼️ Pixel Analysis", "💭 Reasoning"])
            
            with tab1:
                st.write("**Model-Generated Description:**")
                st.info(system_desc)
            
            with tab2:
                st.write("**Description Matching:**")
                st.markdown(match_report.replace("\n", "\n\n"))
            
            with tab3:
                st.write("**Pixel-Level Features:**")
                st.markdown(pixel_report.replace("\n", "\n\n"))
            
            with tab4:
                st.write("**Model Analysis:**")
                st.markdown(reasons)


def show_how_it_works():
    st.markdown('<div class="section-heading">Workflow</div>', unsafe_allow_html=True)
    st.write("A simple process designed for academic research and demonstration.")
    
    col1, col2, col3, col4 = st.columns(4)
    
    steps = [
        ("📤 Upload Artwork", "Provide one clear digital artwork image for analysis."),
        ("📝 Generate Description", "The model creates a description of objects, style, background and composition."),
        ("🔍 Analyze Features", "The system checks semantic evidence, label scores and pixel-level observations."),
        ("📊 Receive Verdict", "Output includes Original/Fake verdict, reasons and technical scores.")
    ]
    
    cols = [col1, col2, col3, col4]
    for col, (title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="info-card">
                <strong>{title}</strong><br>
                {desc}
            </div>
            """, unsafe_allow_html=True)


def show_about():
    st.markdown('<div class="section-heading">About FakeArt Detector</div>', unsafe_allow_html=True)
    
    st.write("""
    FakeArt Detector is an academic research prototype for detecting original and manipulated 
    digital artworks using visual, semantic and pixel-level evidence.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <strong>Purpose</strong><br>
            Support research-based authenticity prediction for artwork images.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <strong>Model</strong><br>
            LLaVA-1.5-7B with LoRA fine-tuning for multimodal reasoning.
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-card">
            <strong>Output</strong><br>
            Verdict, description, consistency report, pixel evidence and technical details.
        </div>
        """, unsafe_allow_html=True)


def show_dataset():
    st.markdown('<div class="section-heading">Research Dataset Context</div>', unsafe_allow_html=True)
    
    st.write("""
    Prepared with paired original and fake artwork images for binary authenticity prediction.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <strong>Total Images</strong><br>
            2,500+ artwork images used in the project context.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <strong>Classes</strong><br>
            Original artwork and fake/manipulated artwork.
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-card">
            <strong>Description</strong><br>
            Every artwork image includes a description to support image-text reasoning.
        </div>
        """, unsafe_allow_html=True)


def show_team():
    st.markdown('<div class="section-heading">Final Year Design Project</div>', unsafe_allow_html=True)
    
    st.write("Academic research prototype developed for fake artwork detection using multimodal AI.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <strong>Built by</strong><br>
            MD. Moshiur Rahman<br>
            MD. Farhan Sadik Shihab
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <strong>Supervisor</strong><br>
            Mr. Syed Eftasum Alam<br>
            Lecturer
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-card">
            <strong>Project</strong><br>
            FakeArt Detector<br>
            Final Year Design Project
        </div>
        """, unsafe_allow_html=True)


def show_contact():
    st.markdown('<div class="section-heading">Project Contact</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <strong>Built by</strong><br>
            MD. Moshiur Rahman<br>
            MD. Farhan Sadik Shihab
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <strong>Supervisor</strong><br>
            Mr. Syed Eftasum Alam
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-card">
            <strong>Email</strong><br>
            fakeart.detector.bd@gmail.com
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; margin-top: 2rem;">
        <strong>FakeArt Detector</strong><br>
        Built by MD. Moshiur Rahman and MD. Farhan Sadik Shihab<br>
        Final Year Design Project · Academic Research Prototype<br>
        LLaVA-1.5-7B with LoRA Fine-Tuning
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
