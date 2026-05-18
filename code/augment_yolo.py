import os
import cv2
import glob
import random
from pathlib import Path
import shutil

import albumentations as A

# ====== SETTINGS ======
SRC = Path(r"D:\Assumption University\AIConcepts\Project\img\yolo_dataset")
DST = Path(r"D:\Assumption University\AIConcepts\Project\img\yolo_dataset_aug")
COPIES_PER_IMAGE = 3   # Reduced for stability (3 is enough)

random.seed(42)

# ====== SAFE AUGMENTATION PIPELINE FOR RECEIPTS ======
transform = A.Compose(
    [
        A.RandomBrightnessContrast(p=0.4),
        A.HueSaturationValue(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0.10,
            rotate_limit=3,
            border_mode=cv2.BORDER_REPLICATE,
            p=0.5
        ),
        A.GaussianBlur(p=0.05),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.3,
        clip=True
    )
)

# ====== UTIL FUNCTIONS ======

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

            # Skip invalid boxes early
            if w <= 0 or h <= 0:
                continue

            bboxes.append([x, y, w, h])
            class_labels.append(cls)

    return bboxes, class_labels


def write_yolo_labels(label_path, bboxes, class_labels):
    with open(label_path, "w", encoding="utf-8") as f:
        for cls, (x, y, w, h) in zip(class_labels, bboxes):
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def clean_destination():
    if DST.exists():
        shutil.rmtree(DST)
    (DST / "images" / "train").mkdir(parents=True, exist_ok=True)
    (DST / "images" / "val").mkdir(parents=True, exist_ok=True)
    (DST / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (DST / "labels" / "val").mkdir(parents=True, exist_ok=True)


def copy_val_set():
    val_imgs = glob.glob(str(SRC / "images" / "val" / "*.*"))

    for img_path in val_imgs:
        img_path = Path(img_path)
        label_path = SRC / "labels" / "val" / (img_path.stem + ".txt")

        if not label_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        cv2.imwrite(str(DST / "images" / "val" / img_path.name), image)
        shutil.copy(label_path, DST / "labels" / "val" / label_path.name)


def augment_train():
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

        # Skip images without valid boxes
        if len(bboxes) == 0:
            continue

        # ---- Copy original ----
        cv2.imwrite(str(DST / "images" / "train" / img_path.name), image)
        shutil.copy(label_path, DST / "labels" / "train" / label_path.name)

        # ---- Augment ----
        for k in range(COPIES_PER_IMAGE):
            try:
                augmented = transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
            except Exception:
                continue

            aug_img = augmented["image"]
            aug_boxes = augmented["bboxes"]
            aug_cls = augmented["class_labels"]

            # Filter invalid boxes
            valid_boxes = []
            valid_cls = []

            for box, cls in zip(aug_boxes, aug_cls):
                x, y, w, h = box
                if w > 0 and h > 0:
                    valid_boxes.append(box)
                    valid_cls.append(cls)

            if len(valid_boxes) == 0:
                continue

            new_img_name = f"{img_path.stem}_aug{k}{img_path.suffix}"
            new_lbl_name = f"{img_path.stem}_aug{k}.txt"

            cv2.imwrite(str(DST / "images" / "train" / new_img_name), aug_img)
            write_yolo_labels(
                DST / "labels" / "train" / new_lbl_name,
                valid_boxes,
                valid_cls
            )


def copy_yaml():
    yaml_path = SRC / "data.yaml"
    if yaml_path.exists():
        content = yaml_path.read_text(encoding="utf-8")
        content = content.replace(str(SRC), str(DST.absolute()))
        (DST / "data.yaml").write_text(content, encoding="utf-8")


# ====== MAIN ======
if __name__ == "__main__":
    print("Cleaning old augmented dataset...")
    clean_destination()

    print("Copying validation set...")
    copy_val_set()

    print("Augmenting training set...")
    augment_train()

    print("Copying data.yaml...")
    copy_yaml()

    print("✅ Augmented dataset created at:", DST)
    print("📊 Train images:",
          len(glob.glob(str(DST / "images" / "train" / "*.*"))))
    print("📊 Val images:",
          len(glob.glob(str(DST / "images" / "val" / "*.*"))))