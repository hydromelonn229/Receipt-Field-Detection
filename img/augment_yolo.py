import os
import cv2
import glob
import random
from pathlib import Path

import albumentations as A

# ====== SETTINGS ======
SRC = Path(r"D:\Assumption University\AIConcepts\Project\img\original_dataset")                 # your original dataset folder
DST = Path("YOLODataset_aug")             # output augmented dataset folder
COPIES_PER_IMAGE = 5                      # how many augmented versions per train image

random.seed(42)

# Albumentations pipeline (safe, common augmentations for receipts)
transform = A.Compose(
    [
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.05, 0.05),
            rotate=(-5, 5),
            p=0.5
        ),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.3),
        A.GaussianBlur(p=0.15),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.4,
        clip=True
    )
)

def read_yolo_labels(label_path):
    bboxes = []
    class_labels = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            # Clamp values to [0, 1] to fix floating-point precision errors
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            bboxes.append([x, y, w, h])
            class_labels.append(cls)
    return bboxes, class_labels

def write_yolo_labels(label_path, bboxes, class_labels):
    with open(label_path, "w", encoding="utf-8") as f:
        for cls, (x, y, w, h) in zip(class_labels, bboxes):
            # Clamp to [0, 1] to prevent any floating-point errors
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def copy_tree():
    # Copy val set as-is (no augmentation)
    for split in ["val"]:
        (DST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / split).mkdir(parents=True, exist_ok=True)

        for img_path in glob.glob(str(SRC / "images" / split / "*.*")):
            img_path = Path(img_path)
            label_path = SRC / "labels" / split / (img_path.stem + ".txt")
            if not label_path.exists():
                continue
            os.makedirs(DST / "images" / split, exist_ok=True)
            os.makedirs(DST / "labels" / split, exist_ok=True)
            cv2.imwrite(str(DST / "images" / split / img_path.name), cv2.imread(str(img_path)))
            with open(label_path, "r", encoding="utf-8") as f:
                (DST / "labels" / split / label_path.name).write_text(f.read(), encoding="utf-8")

    # Copy dataset.yaml
    (DST / "data.yaml").write_text((SRC / "data.yaml").read_text(encoding="utf-8"), encoding="utf-8")

def augment_train():
    (DST / "images" / "train").mkdir(parents=True, exist_ok=True)
    (DST / "labels" / "train").mkdir(parents=True, exist_ok=True)

    train_imgs = glob.glob(str(SRC / "images" / "train" / "*.*"))
    for img_path in train_imgs:
        img_path = Path(img_path)
        label_path = SRC / "labels" / "train" / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        bboxes, class_labels = read_yolo_labels(label_path)

        # Clamp YOLO boxes before passing to Albumentations (fixes floating-point errors)
        clean_bboxes = []
        for bbox in bboxes:
            x, y, w, h = bbox
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            clean_bboxes.append([x, y, w, h])

        # 1) Copy original
        cv2.imwrite(str(DST / "images" / "train" / img_path.name), image)
        (DST / "labels" / "train" / label_path.name).write_text(label_path.read_text(encoding="utf-8"), encoding="utf-8")

        # 2) Make augmented copies
        for k in range(COPIES_PER_IMAGE):
            augmented = transform(image=image, bboxes=clean_bboxes, class_labels=class_labels)
            aug_img = augmented["image"]
            aug_boxes = augmented["bboxes"]
            aug_cls = augmented["class_labels"]

            # Skip if augmentation removed all boxes (can happen rarely)
            if len(aug_boxes) == 0:
                continue

            new_name = f"{img_path.stem}_aug{k}{img_path.suffix}"
            new_lbl = f"{img_path.stem}_aug{k}.txt"

            cv2.imwrite(str(DST / "images" / "train" / new_name), aug_img)
            write_yolo_labels(DST / "labels" / "train" / new_lbl, aug_boxes, aug_cls)

if __name__ == "__main__":
    copy_tree()
    augment_train()
    print("✅ Augmented dataset created at:", DST)
