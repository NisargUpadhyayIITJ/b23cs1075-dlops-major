"""
app.py - Streamlit app for CityScape Image Segmentation
Page 1: Training plots + test metrics
Page 2: Upload test images → show GT mask + predicted mask
"""

import os
import sys
import json
import numpy as np
import cv2
import torch
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unet_model import UNet

# ============================================================
# Configuration
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "unet_cityscapes.pth")
METRICS_PATH = os.path.join(SCRIPT_DIR, "metrics.json")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
SPLIT_PATH = os.path.join(SCRIPT_DIR, "test_split.json")
NUM_CLASSES = 23

# Color map for visualization (23 classes)
np.random.seed(42)
COLORMAP = np.random.randint(0, 255, (NUM_CLASSES, 3), dtype=np.uint8)
COLORMAP[0] = [0, 0, 0]  # Background black


def colorize_mask(mask):
    """Convert class indices to RGB color mask."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(NUM_CLASSES):
        color_mask[mask == cls] = COLORMAP[cls]
    return color_mask


@st.cache_resource
def load_model():
    """Load the trained UNet model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=3, n_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device


def predict(model, device, image_array):
    """Run inference on an image."""
    img = cv2.resize(image_array, (128, 96), interpolation=cv2.INTER_NEAREST)
    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    return pred


# ============================================================
# Streamlit App
# ============================================================
st.set_page_config(page_title="CityScape Segmentation", layout="wide")

page = st.sidebar.selectbox("Select Page", ["📊 Training Results", "🖼️ Predict Segmentation"])

if page == "📊 Training Results":
    st.title("📊 CityScape Image Segmentation - Training Results")
    st.markdown("---")

    # Load metrics
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)

        # Test set metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Test mIOU", f"{metrics['test_miou']:.4f}")
        with col2:
            st.metric("Test mDice", f"{metrics['test_mdice']:.4f}")

        st.markdown("---")

        # Training plots
        st.subheader("Training Curves")

        plot_files = ["training_loss.png", "training_miou.png", "training_mdice.png"]
        plot_titles = ["Training Loss", "Training mIOU", "Training mDice"]

        cols = st.columns(3)
        for col, pf, pt in zip(cols, plot_files, plot_titles):
            plot_path = os.path.join(PLOTS_DIR, pf)
            if os.path.exists(plot_path):
                with col:
                    st.image(plot_path, caption=pt, use_container_width=True)
            else:
                with col:
                    st.warning(f"Plot not found: {pf}")
    else:
        st.warning("No metrics found. Please run training first.")

elif page == "🖼️ Predict Segmentation":
    st.title("🖼️ CityScape Image Segmentation - Prediction")
    st.markdown("Upload 4 test images to see ground-truth and predicted segmentation masks.")
    st.markdown("---")

    model, device = load_model()

    # Load test split to get GT masks
    test_mask_map = {}
    if os.path.exists(SPLIT_PATH):
        with open(SPLIT_PATH, "r") as f:
            split = json.load(f)
        for img_path, mask_path in zip(split["test_images"], split["test_masks"]):
            basename = os.path.basename(img_path)
            test_mask_map[basename] = mask_path

    uploaded_files = st.file_uploader(
        "Upload 4 test images", type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for i, uploaded_file in enumerate(uploaded_files[:4]):
            st.subheader(f"Image {i+1}: {uploaded_file.name}")

            # Load uploaded image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Predict
            pred_mask = predict(model, device, img_rgb)
            pred_color = colorize_mask(pred_mask)

            # Try to load ground truth
            gt_color = None
            basename = uploaded_file.name
            if basename in test_mask_map:
                gt_mask_full = cv2.imread(test_mask_map[basename])
                gt_mask_full = cv2.cvtColor(gt_mask_full, cv2.COLOR_BGR2RGB)
                gt_mask_full = cv2.resize(gt_mask_full, (128, 96), interpolation=cv2.INTER_NEAREST)
                gt_mask = np.max(gt_mask_full, axis=-1)
                gt_color = colorize_mask(gt_mask)

            # Display
            img_display = cv2.resize(img_rgb, (128, 96), interpolation=cv2.INTER_NEAREST)
            if gt_color is not None:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(img_display, caption="Input Image", use_container_width=True)
                with col2:
                    st.image(gt_color, caption="Ground Truth Mask", use_container_width=True)
                with col3:
                    st.image(pred_color, caption="Predicted Mask", use_container_width=True)
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img_display, caption="Input Image", use_container_width=True)
                with col2:
                    st.image(pred_color, caption="Predicted Mask", use_container_width=True)

            st.markdown("---")
