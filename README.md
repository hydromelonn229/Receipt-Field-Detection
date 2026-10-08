# 🧾 Receipt Field Detection using YOLOv8

> An object detection system that automatically locates and identifies key fields in receipt and invoice images using a fine-tuned YOLOv8 model.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## Table of Contents

- [Project Overview](#-project-overview)
- [Detected Classes](#-detected-classes)
- [Dataset](#-dataset)
- [Pipeline](#-pipeline)
- [Data Augmentation](#-data-augmentation)
- [Model Training](#-model-training)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Running Inference](#-running-inference)
- [Requirements](#-requirements)

---

## Project Overview

This project was developed as part of the **AI Concepts course at Assumption University**. The goal is to train a **YOLOv8 object detection model** to automatically detect and localize seven key fields found in receipt/invoice documents.

Instead of reading text manually, this system draws bounding boxes around the relevant fields in any receipt image — enabling downstream automation such as data extraction, accounting workflows, or digitization pipelines.

The full pipeline covers:
- Raw image collection and annotation using **LabelMe**
- Label extraction via **Azure Document Intelligence**
- Label format conversion (LabelMe JSON → YOLO `.txt`)
- Dataset splitting and augmentation
- Model fine-tuning using **YOLOv8s** (pretrained on COCO)
- Inference on new receipt images

---

## Detected Classes

The model is trained to detect **7 field types** in receipt/invoice images:

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 1 | `ProductDescription` | Name or description of each purchased item |
| 2 | `Quantity` | Number of units for each line item |
| 3 | `Price` | Unit price or line-item price |
| 4 | `TotalDue` | Final total amount due |

---

## Dataset

- **Total Images**: 624 receipt photographs
- **Annotation Tool**: [LabelMe](https://github.com/labelmeai/labelme) (JSON format)
- **Label Extraction**: Azure Document Intelligence (AI-assisted pre-labeling)
- **Train / Val Split**: 80% / 20%
  - Training set: ~499 images
  - Validation set: ~125 images

### After Augmentation

| Split | Original | Augmented | Total |
|-------|----------|-----------|-------|
| Train | 499 | ~1,497 | ~1,996 |
| Val | 125 | ~375 | ~500 |
| **Total** | **624** | **~1,872** | **~2,496** |

---

## Pipeline

```
Raw Images + LabelMe JSON Annotations
            │
            ▼
  convert_labelme_to_yolo.py
  → Converts bounding boxes to normalized YOLO format
  → Output: yolo_labels_cleaned/ (624 .txt files)
            │
            ▼
  create_original_dataset.py
  → 80/20 train/val split
  → Creates YOLO folder structure + data.yaml
  → Output: original_dataset/
            │
            ▼
  augmentation_code.py
  → Applies 3 augmented versions per training image
  → Output: augmented_dataset/ (~2,496 images total)
            │
            ▼
  model.py / train_yolo.py
  → Fine-tunes YOLOv8s for 100 epochs (640×640, batch=16)
  → Output: runs/detect/partial_invoice_model/weights/best.pt
            │
            ▼
  infer.py
  → Run predictions on new receipt images
```

---

## Data Augmentation

To address the relatively small dataset size (624 images), each training image is augmented **3 times** using a combination of the following techniques:

| Technique | Description |
|-----------|-------------|
| **Horizontal Flip** | Mirrors the image left-to-right; bounding boxes adjusted accordingly |
| **Brightness Adjustment** | Randomly scales brightness by a factor of 0.7–1.3 (via HSV color space) |
| **Contrast Adjustment** | Scales contrast by a factor of 0.8–1.2 around the image mean |
| **Gaussian Noise** | Adds random Gaussian noise (std dev 5–15) to simulate scan artifacts |
| **Small Rotation** | Rotates image by ±10° to handle slight camera tilt |
| **Gaussian Blur** | Applies 3×3 or 5×5 blur kernel to simulate camera defocus |

Each augmented image retains its corresponding YOLO label file with adjusted bounding box coordinates.

> **Note**: For each image, 3 augmentation types are randomly selected from the 6 available techniques to maximize variety.

---

## Model Training

### Architecture
- **Base Model**: YOLOv8s (Small) — pretrained on COCO
- **Framework**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Image Size | 640 × 640 |
| Batch Size | 16 |
| Early Stopping Patience | 20 epochs |
| Optimizer | Auto (AdamW) |
| Confidence Threshold (inference) | 0.25 |
| NMS IoU Threshold | 0.45 |
| Device | CUDA (GPU) / CPU fallback |

### Training Command
```bash
python model.py
```
Or using the detailed training script:
```bash
python img/train_yolo.py
```

### Output
```
runs/detect/partial_invoice_model/
├── weights/
│   ├── best.pt     ← Best model checkpoint (lowest validation loss)
│   └── last.pt     ← Final epoch checkpoint
├── confusion_matrix.png
├── results.png
├── train_batch*.jpg
└── val_batch*.jpg
```

---

## Results

### Expected Performance (YOLOv8s on ~2,496 images)

| Metric | Expected Range |
|--------|---------------|
| mAP@50 | 0.75 – 0.85 |
| mAP@50-95 | 0.50 – 0.70 |
| Training Time | 30 – 60 minutes (GPU) |

> Actual results depend on dataset quality, GPU hardware, and annotation accuracy.

To regenerate metrics on the validation set after training:
```python
from ultralytics import YOLO

model = YOLO("runs/detect/partial_invoice_model/weights/best.pt")
metrics = model.val()
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
```

---

## Project Structure

```
Project/
├── README.md
│
├── model.py                      # Main training entry point
├── analyze_yolo_final.py         # Dataset class distribution analysis
├── yolov8s.pt                    # Pretrained YOLOv8s base model (COCO)
│
├── YOLODataset_aug/              # Final augmented dataset (used for training)
│   ├── dataset.yaml              # Dataset configuration (7 classes)
│   ├── images/
│   │   ├── train/                # ~1,996 training images
│   │   └── val/                  # ~500 validation images
│   └── labels/
│       ├── train/                # Corresponding YOLO .txt label files
│       └── val/
│
├── yolo_dataset/                 # Original split dataset (pre-augmentation)
│   ├── data.yaml
│   ├── images/
│   └── labels/
│
├── code/
│   ├── best.pt                   # Trained model weights
│   ├── augment_yolo.py           # Augmentation script (Albumentations-based)
│   ├── infer.py                  # Inference script
│   ├── 307.jpg                   # Sample receipt image
│   └── 581.jpg                   # Sample receipt image
│
├── img/                          # Development scripts and workflow docs
│   ├── PROJECT_WORKFLOW.md       # Step-by-step execution guide
│   ├── convert_labelme_to_yolo.py
│   ├── create_original_dataset.py
│   ├── augmentation_code.py      # Full augmentation pipeline
│   ├── augment_yolo.py           # Albumentations-based augmenter
│   ├── clean_labelme_data.py
│   ├── analyze_yolo_dataset.py
│   ├── train_yolo.py             # Detailed training script
│   ├── batch_preprocess_data.ipynb
│   ├── labelme_data/             # Raw LabelMe JSON annotations
│   ├── images/                   # Raw receipt images
│   ├── original_dataset/         # Original dataset structure
│   ├── yolo_annotations/
│   └── yolo_labels_cleaned/
│
└── runs/
    └── detect/
        ├── partial_invoice_model/    # Training run 1
        ├── partial_invoice_model2/   # Training run 2
        ├── predict/                  # Inference results
        └── predict2/
```

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/receipt-field-detection.git
cd receipt-field-detection
```

### 2. Create a Virtual Environment
```bash
python -m venv ENV
source ENV/bin/activate        # Linux / macOS
ENV\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install ultralytics opencv-python numpy albumentations scikit-learn pillow torch
```

### 4. Reproduce the Dataset (Optional)
If you want to rebuild from raw annotations:
```bash
# Step 1: Convert LabelMe JSON → YOLO format
python img/convert_labelme_to_yolo.py

# Step 2: Create original dataset with train/val split
python img/create_original_dataset.py

# Step 3: Apply augmentations (3x factor)
python img/augmentation_code.py
```

### 5. Train the Model
```bash
python model.py
```
Or with the more configurable script:
```bash
python img/train_yolo.py
```

---

## Running Inference

Use the pretrained weights in `code/best.pt` to run predictions on any receipt image:

```python
from ultralytics import YOLO

model = YOLO("code/best.pt")
results = model.predict(source="path/to/receipt.jpg", conf=0.25, save=True)
print("Saved to:", results[0].save_dir)
```

Or run the provided script directly:
```bash
python code/infer.py
```

### Output
The model will draw bounding boxes on detected fields and save the annotated image to `runs/detect/predict/`.

---

## 📋 Requirements

| Package | Purpose |
|---------|---------|
| `ultralytics` | YOLOv8 model training & inference |
| `opencv-python` | Image reading, augmentation, and saving |
| `numpy` | Numerical operations |
| `albumentations` | Advanced augmentation pipeline |
| `torch` | PyTorch backend for YOLO |
| `pillow` | Image utilities |
| `scikit-learn` | Train/val splitting |

Install all at once:
```bash
pip install ultralytics opencv-python numpy albumentations torch pillow scikit-learn
```

> **GPU Recommended**: Training on CPU is supported but significantly slower. A CUDA-capable GPU is strongly recommended for the 100-epoch training run.

---
