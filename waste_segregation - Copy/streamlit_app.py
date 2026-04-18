"""
Smart Waste Segregation System - Streamlit Web Interface
Run with: streamlit run streamlit_app.py
"""

import torch
import timm
import numpy as np
from PIL import Image
from io import BytesIO

import streamlit as st
import plotly.graph_objects as go

from src.transforms import get_transforms

# ── Configuration ──────────────────────────────────────────────
MODEL_NAME = "efficientnet_b2"
MODEL_PATH = "models/best_1efficientnet_b2.pth"
IMG_SIZE = 224
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Waste category mapping
CATEGORY_MAP = {
    "cardboard": {"category": "Recyclable",      "bin": "Blue Bin",   "color": "#2196F3", "indicator": "🔵"},
    "glass":     {"category": "Recyclable",      "bin": "Blue Bin",   "color": "#2196F3", "indicator": "🔵"},
    "metal":     {"category": "Metal/Plastic",   "bin": "Yellow Bin", "color": "#FFC107", "indicator": "🟡"},
    "paper":     {"category": "Recyclable",      "bin": "Blue Bin",   "color": "#2196F3", "indicator": "🔵"},
    "plastic":   {"category": "Metal/Plastic",   "bin": "Yellow Bin", "color": "#FFC107", "indicator": "🟡"},
    "trash":     {"category": "Non-recyclable",  "bin": "Black Bin",  "color": "#424242", "indicator": "⚫"},
}

RECYCLING_TIPS = {
    "cardboard": "Flatten boxes and remove tape or staples before recycling",
    "glass":     "Rinse containers and remove lids; do not mix with ceramics",
    "metal":     "Rinse cans, crush if possible; remove paper labels",
    "paper":     "Keep dry and clean; do not recycle greasy or wet paper",
    "plastic":   "Rinse and crush bottles; check the resin code on the bottom",
    "trash":     "Bag securely before disposal; check for any recyclable items mixed in",
}


# ── Model Loading (cached) ────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASSES))
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    transform = get_transforms("torchvision", "val", IMG_SIZE)
    return model, device, transform


def predict_image(image: Image.Image) -> dict:
    model, device, transform = load_model()
    image = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0]
    top_prob, top_idx = probs.max(0)
    predicted_class = CLASSES[top_idx.item()]
    confidence = top_prob.item()
    all_probs = {cls: float(probs[i]) for i, cls in enumerate(CLASSES)}
    return {
        "object_name": predicted_class,
        "confidence": confidence,
        "probabilities": all_probs,
    }


# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Waste Segregation",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .main-title {
        text-align: center;
        padding: 1rem 0 0.2rem;
        font-size: 2.2rem;
        font-weight: 800;
        color: #1a5e3c;
        letter-spacing: -0.5px;
    }
    .main-subtitle {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    /* Detection panel */
    .panel {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 2px solid #dee2e6;
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin: 1rem 0;
        font-family: 'Consolas', 'Courier New', monospace;
    }
    .panel-header {
        text-align: center;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 2px;
        color: #1a5e3c;
        border-bottom: 2px solid #1a5e3c;
        padding-bottom: 0.6rem;
        margin-bottom: 1rem;
    }
    .panel-row {
        display: flex;
        justify-content: space-between;
        padding: 0.45rem 0;
        font-size: 0.95rem;
        border-bottom: 1px dashed #ced4da;
    }
    .panel-row:last-child { border-bottom: none; }
    .panel-label { color: #495057; font-weight: 600; }
    .panel-value { color: #212529; font-weight: 700; text-align: right; }

    /* Status box */
    .status-box {
        text-align: center;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        font-weight: 700;
        font-size: 1.5rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* Tip box */
    .tip-box {
        background: #e8f5e9;
        border-left: 5px solid #2e7d32;
        padding: 1rem 1.2rem;
        border-radius: 0 10px 10px 0;
        margin-top: 0.8rem;
        font-size: 0.92rem;
        color: #1b5e20;
    }
    .tip-label {
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    /* Upload area tweaks */
    [data-testid="stFileUploader"] {
        border: 2px dashed #4CAF50;
        border-radius: 14px;
        padding: 0.5rem;
    }

    .divider {
        border: none;
        border-top: 2px solid #e0e0e0;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────
st.markdown('<div class="main-title">Smart Waste Segregation System</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">AI-Powered Waste Classification using EfficientNet</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── Input Mode Selection ───────────────────────────────────────
input_mode = st.radio(
    "Select Input Mode",
    ["Upload Image", "Live Webcam"],
    horizontal=True,
    label_visibility="collapsed",
)

captured_image = None

if input_mode == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload a waste image",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        label_visibility="collapsed",
    )
    if uploaded_file:
        captured_image = Image.open(uploaded_file)

elif input_mode == "Live Webcam":
    cam_photo = st.camera_input("Capture waste image", label_visibility="collapsed")
    if cam_photo:
        captured_image = Image.open(cam_photo)


# ── Run Detection ──────────────────────────────────────────────
if captured_image is not None:
    # Show uploaded image centered
    col_img = st.columns([1, 2, 1])
    with col_img[1]:
        st.image(captured_image, caption="Input Image", use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    with st.spinner("Classifying waste..."):
        result = predict_image(captured_image)

    object_name = result["object_name"]
    confidence = result["confidence"]
    probabilities = result["probabilities"]
    info = CATEGORY_MAP[object_name]
    tip = RECYCLING_TIPS[object_name]

    # ── Detection Panel ────────────────────────────────────────
    panel_html = f"""
    <div class="panel">
        <div class="panel-header">DETECTION RESULT</div>
        <div class="panel-row">
            <span class="panel-label">Detected</span>
            <span class="panel-value">{object_name.title()}</span>
        </div>
        <div class="panel-row">
            <span class="panel-label">Category</span>
            <span class="panel-value">{info["category"]}</span>
        </div>
        <div class="panel-row">
            <span class="panel-label">Bin Type</span>
            <span class="panel-value">{info["indicator"]}  {info["bin"]}</span>
        </div>
        <div class="panel-row">
            <span class="panel-label">Confidence</span>
            <span class="panel-value">{confidence * 100:.1f}%</span>
        </div>
    </div>
    """
    st.markdown(panel_html, unsafe_allow_html=True)

    # ── Color-coded Status Box ─────────────────────────────────
    st.markdown(
        f'<div class="status-box" style="background:{info["color"]}20; '
        f'color:{info["color"]}; border: 2px solid {info["color"]};">'
        f'{info["indicator"]}  {info["category"]}  —  {info["bin"]}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Confidence Progress Bar ────────────────────────────────
    st.markdown("**Confidence Score**")
    st.progress(confidence, text=f"{confidence * 100:.1f}%")

    # ── Recycling Tip ──────────────────────────────────────────
    st.markdown(
        f'<div class="tip-box">'
        f'<div class="tip-label">Recycling Tip</div>'
        f'{tip}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Probability Chart ──────────────────────────────────────
    st.markdown("**Detection Confidence by Category**")

    sorted_probs = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))
    class_names = [c.title() for c in sorted_probs.keys()]
    class_vals = [v * 100 for v in sorted_probs.values()]
    bar_colors = [CATEGORY_MAP[c]["color"] for c in sorted_probs.keys()]

    fig = go.Figure(
        go.Bar(
            x=class_vals,
            y=class_names,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.1f}%" for v in class_vals],
            textposition="auto",
            textfont=dict(size=13, color="white"),
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=20, t=10, b=10),
        xaxis=dict(title="Confidence (%)", range=[0, 100], showgrid=True, gridcolor="#e0e0e0"),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    # Placeholder when no image is loaded
    st.markdown("")
    col_p = st.columns([1, 2, 1])
    with col_p[1]:
        st.info("Upload an image or capture from webcam to start classification.")
