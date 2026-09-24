import json
import os

def get_file_content(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

cells = []

# Title & Overview
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 🚀 Industrial Fluid-Bed Pellet Detection & Agglomeration Classifier\n",
        "\n",
        "This master notebook integrates an end-to-end **two-stage Spatial-Temporal AI Pipeline**:\n",
        "1. **Spatial Detector & Tracker**: High-resolution **YOLO11** model with **ByteTrack** for persistent object identification.\n",
        "2. **Temporal Kinematic Classifier**: Deep **4-Stage ResNet + Bidirectional 2-Layer LSTM + Multi-Head Temporal Self-Attention** model to accurately classify tumbling agglomerates across 10 consecutive frames.\n",
        "\n",
        "All parameters can be tuned directly inside the cells below."
    ]
})

# 1. Prepare YOLO
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 📁 Step 1: Prepare High-Resolution YOLO Dataset\n",
        "Reorganizes raw simulation captures and bounding box ground truth into standard YOLO format (`images/` and `labels/`)."
    ]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [get_file_content("prepare_yolo_data.py")]
})

# 2. Train YOLO
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🧠 Step 2: Train High-Resolution YOLO11 Tracker\n",
        "Trains the **YOLO11s** (Small, 9.4M parameters) model at **1024x1024 resolution** with specialized small-object loss parameters."
    ]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [get_file_content("train_yolo.py")]
})

# 3. Extract Sequences
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🎞️ Step 3: Extract High-Detail Tumbling Sequences (96x96 Crops)\n",
        "Tracks each unique pellet across continuous 10-frame windows and generates square, normalized tensor sequences (`sequence_dataset/single/` and `sequence_dataset/agglomerate/`)."
    ]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [get_file_content("extract_sequences.py")]
})

# 4. Train LSTM
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🔬 Step 4: Train Deep ResNet-BiLSTM with Temporal Attention\n",
        "Trains the multi-stage ResNet spatial encoder and Bidirectional LSTM with Multi-Head Attention to identify rotating rigid agglomerates."
    ]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [get_file_content("train_sequence_model.py")]
})

# 5. Inference
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🎯 Step 5: Master Real-Time Spatial-Temporal Inference Pipeline\n",
        "Executes the full pipeline: YOLO11 tracking -> Suspicion Trigger -> 10-frame Temporal Buffering -> ResNet-BiLSTM Attention Classification -> Permanent Memory Lock."
    ]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [get_file_content("infer.py")]
})

# 6. Interactive Visualizer
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🖼️ Step 6: Interactive Frame Inspector & Visualizer\n",
        "Renders annotated detection frames directly inside the notebook with colored bounding boxes:\n",
        "- 🔵 **Blue**: Regular Tracked Pellets\n",
        "- 🟠 **Orange**: Suspicious Pellets (buffering 10-frame tumbling motion)\n",
        "- 🟢 **Green**: Confirmed Single Pellets (verified by Temporal Attention)\n",
        "- 🔴 **Red**: Confirmed Agglomerates (rigid multi-sphere tumble detected)"
    ]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "%matplotlib inline\n",
        "import matplotlib.pyplot as plt\n",
        "import cv2\n",
        "import random\n",
        "import glob\n",
        "\n",
        "# Find all processed frames\n",
        "output_images = sorted(glob.glob('output_frames/*.jpg'))\n",
        "\n",
        "if output_images:\n",
        "    # Randomly pick an annotated frame or specify an index (e.g. output_images[20])\n",
        "    selected_path = random.choice(output_images)\n",
        "    img_bgr = cv2.imread(selected_path)\n",
        "    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)\n",
        "    \n",
        "    plt.figure(figsize=(12, 12), dpi=120)\n",
        "    plt.imshow(img_rgb)\n",
        "    plt.axis('off')\n",
        "    plt.title(f\"Detection & Classification Output: {selected_path}\", fontsize=15, pad=12)\n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
        "else:\n",
        "    print(\"No annotated images found in output_frames/. Please run Step 5 (infer.py) first!\")\n"
    ]
})

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open("pipeline.ipynb", "w", encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Created updated pipeline.ipynb successfully.")
