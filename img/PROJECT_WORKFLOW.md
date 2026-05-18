# Receipt Detection Project - Complete Workflow

## 📋 Project Overview
**Objective**: Train a YOLO model to detect key fields in receipt images:
- Class 0: product_description
- Class 1: price
- Class 2: quantity_item
- Class 3: total_due_amount

**Dataset**: 624 receipt images with annotations

---

## 🚀 Step-by-Step Execution Guide

### **STEP 1: Convert LabelMe to YOLO Format**
```bash
python convert_labelme_to_yolo.py
```
**What it does:**
- Reads cleaned labelme JSON files from `labelme_data/`
- Converts bounding boxes to YOLO format (normalized coordinates)
- Saves annotations to `yolo_labels_cleaned/`

**Expected output:**
- 624 `.txt` files in `yolo_labels_cleaned/`
- Each line format: `class_id x_center y_center width height`

---

### **STEP 2: Create Original Dataset (BEFORE Augmentation)**
```bash
python create_original_dataset.py
```
**What it does:**
- Creates proper YOLO dataset structure
- Splits data into train (80%) and validation (20%)
- Generates `data.yaml` configuration file

**Output structure:**
```
original_dataset/
├── images/
│   ├── train/        (~499 images)
│   └── val/          (~125 images)
├── labels/
│   ├── train/        (~499 .txt files)
│   └── val/          (~125 .txt files)
└── data.yaml
```

**⚠️ IMPORTANT**: Save `original_dataset/` folder - this is your **"Dataset BEFORE augmentation"** requirement!

---

### **STEP 3: Data Augmentation**
```bash
python augmentation_code.py
```
**What it does:**
- Reads from `original_dataset/`
- Applies 3 augmentation techniques per image:
  - Horizontal flip
  - Brightness adjustment
  - Contrast adjustment
  - Gaussian noise
  - Small rotation (-10° to +10°)
  - Gaussian blur

**Output structure:**
```
augmented_dataset/
├── images/
│   ├── train/        (~1996 images: 499 original + ~1497 augmented)
│   └── val/          (~500 images: 125 original + ~375 augmented)
├── labels/
│   ├── train/        (~1996 .txt files)
│   └── val/          (~500 .txt files)
└── data.yaml
```

**Augmentation factor**: 3 (creates 3 augmented versions per image)

---

### **STEP 4: Install Training Requirements**
```bash
pip install ultralytics opencv-python scikit-learn pillow
```

---

### **STEP 5: Train YOLO Model**
```bash
python train_yolo.py
```
**What it does:**
- Loads YOLOv8n (nano) pretrained model
- Trains on augmented dataset for 100 epochs
- Saves checkpoints and best model
- Generates training plots and confusion matrix

**Training parameters:**
- Epochs: 100
- Image size: 640x640
- Batch size: 16
- Model: YOLOv8n (fastest, good for receipts)
- Early stopping: patience=20

**Output:**
```
runs/detect/receipt_detection/
├── weights/
│   ├── best.pt       ← **SUBMIT THIS**
│   └── last.pt
├── confusion_matrix.png
├── results.png
├── train_batch*.jpg
└── val_batch*.jpg
```

**Training time**: ~30-60 minutes (depends on GPU)

---

## 📦 Submission Checklist

### ✅ 1. best.pt (Ready to use)
**Location**: `runs/detect/receipt_detection/weights/best.pt`
**Usage**:
```python
from ultralytics import YOLO
model = YOLO('best.pt')
results = model.predict('receipt_image.jpg')
```

### ✅ 2. Dataset BEFORE Augmentation
**Submit the entire folder**: `original_dataset/`
**Contents**:
- Original images (train: ~499, val: ~125)
- YOLO labels (.txt files)
- data.yaml configuration

### ✅ 3. Augmentation Code (Runnable)
**File**: `augmentation_code.py`
**How to run**:
```bash
python augmentation_code.py
```
**Requirements**: opencv-python, numpy

### ✅ 4. ONE PDF Report

**Required sections:**

#### A. Introduction
- Project objective
- Dataset description (624 images, 4 classes)
- Why YOLO for receipt detection

#### B. Data Preparation (with screenshots)
1. **Raw data collection**
   - Screenshot of `images/` folder
   - Screenshot of Azure Document Intelligence output

2. **Label cleaning**
   - Screenshot showing label analysis
   - Before/after label counts
   - Class mapping table

3. **Dataset organization**
   - Original dataset structure
   - Train/val split ratios
   - Sample images with annotations

#### C. Data Augmentation (with code snippets)
- Augmentation techniques used:
  * Horizontal flip
  * Brightness adjustment
  * Contrast adjustment
  * Gaussian noise
  * Rotation
  * Gaussian blur
- Before/after image examples
- Dataset size comparison:
  * Original: 624 images
  * After split: ~499 train, ~125 val
  * After augmentation: ~1996 train, ~500 val

#### D. Model Training (with screenshots)
- Model architecture: YOLOv8n
- Training parameters
- Screenshot of training command
- Training curves (loss, mAP)
- Training time and hardware used

#### E. Results & Evaluation
- **Confusion Matrix** ← Screenshot from `runs/detect/receipt_detection/confusion_matrix.png`
- Metrics:
  * mAP50
  * mAP50-95
  * Precision
  * Recall
  * F1-score
- Per-class performance
- Sample predictions (with confidence scores)

#### F. Conclusion
- Model performance summary
- Challenges faced
- Possible improvements

#### G. Appendix
- Code snippets
- Full dataset statistics
- Hardware specifications

---

## 📊 Expected Results

### Dataset Statistics:
| Split | Original | After Augmentation |
|-------|----------|-------------------|
| Train | ~499     | ~1996             |
| Val   | ~125     | ~500              |
| Total | 624      | ~2496             |

### Model Performance (Expected):
- mAP50: 0.75-0.85 (depends on data quality)
- mAP50-95: 0.50-0.70
- Training time: 30-60 minutes

---

## 🔍 How to Generate Confusion Matrix

The confusion matrix is automatically generated during training at:
`runs/detect/receipt_detection/confusion_matrix.png`

To generate it separately:
```python
from ultralytics import YOLO

model = YOLO('runs/detect/receipt_detection/weights/best.pt')
metrics = model.val()  # This generates the confusion matrix
```

---

## 💡 Tips for Report

1. **Take screenshots at each step** (before running the script)
2. **Include command outputs** (copy terminal text)
3. **Show sample images**:
   - Original image
   - Image with annotations
   - Augmented versions
   - Prediction results
4. **Explain your choices**:
   - Why these augmentation techniques?
   - Why 80/20 split?
   - Why YOLOv8n?

---

## ⚠️ Common Issues & Solutions

### Issue 1: CUDA out of memory
**Solution**: Reduce batch size in `train_yolo.py`:
```python
BATCH_SIZE = 8  # or even 4
```

### Issue 2: Training too slow
**Solution**: Use smaller image size:
```python
IMAGE_SIZE = 416  # instead of 640
```

### Issue 3: Missing labels
**Error**: "No label found for X.jpg"
**Solution**: Check that `convert_labelme_to_yolo.py` completed successfully

---

## 📝 Final Checklist Before Submission

- [ ] `best.pt` file exists and can load
- [ ] `original_dataset/` folder is complete
- [ ] `augmentation_code.py` runs without errors
- [ ] PDF report has all sections
- [ ] PDF includes confusion matrix screenshot
- [ ] Can explain all steps in the workflow
- [ ] Code has comments explaining key parts

---

## 🎯 Quick Execution Summary

```bash
# Step 1: Convert labels
python convert_labelme_to_yolo.py

# Step 2: Create original dataset (SAVE THIS!)
python create_original_dataset.py

# Step 3: Augment dataset
python augmentation_code.py

# Step 4: Train model
python train_yolo.py

# Step 5: Submit:
# - runs/detect/receipt_detection/weights/best.pt
# - original_dataset/ (entire folder)
# - augmentation_code.py
# - report.pdf
```

**Total time**: ~1-2 hours (mostly training time)

---

Good luck with your project! 🚀
