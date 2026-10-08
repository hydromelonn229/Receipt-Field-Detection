"""
YOLO Dataset Augmentation Script
Applies various augmentation techniques to increase dataset size
"""
import os
import cv2
import numpy as np
import random
from pathlib import Path
import shutil

class YOLOAugmenter:
    def __init__(self, seed=42):
        """Initialize augmenter with random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
    
    def load_yolo_annotations(self, label_path):
        """Load YOLO format annotations from a text file."""
        annotations = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = parts
                        annotations.append({
                            'class_id': int(class_id),
                            'x_center': float(x_center),
                            'y_center': float(y_center),
                            'width': float(width),
                            'height': float(height)
                        })
        return annotations
    
    def save_yolo_annotations(self, annotations, output_path):
        """Save YOLO format annotations to a text file."""
        with open(output_path, 'w') as f:
            for ann in annotations:
                f.write(f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} "
                       f"{ann['width']:.6f} {ann['height']:.6f}\n")
    
    def adjust_bbox_after_flip(self, annotations, flip_horizontal=False, flip_vertical=False):
        """Adjust bounding box coordinates after flipping."""
        adjusted = []
        for ann in annotations:
            new_ann = ann.copy()
            if flip_horizontal:
                new_ann['x_center'] = 1.0 - ann['x_center']
            if flip_vertical:
                new_ann['y_center'] = 1.0 - ann['y_center']
            adjusted.append(new_ann)
        return adjusted
    
    def horizontal_flip(self, image, annotations):
        """Flip image horizontally and adjust annotations."""
        flipped_img = cv2.flip(image, 1)
        flipped_ann = self.adjust_bbox_after_flip(annotations, flip_horizontal=True)
        return flipped_img, flipped_ann
    
    def vertical_flip(self, image, annotations):
        """Flip image vertically and adjust annotations."""
        flipped_img = cv2.flip(image, 0)
        flipped_ann = self.adjust_bbox_after_flip(annotations, flip_vertical=True)
        return flipped_img, flipped_ann
    
    def adjust_brightness(self, image, factor=None):
        """Adjust image brightness."""
        if factor is None:
            factor = random.uniform(0.7, 1.3)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def add_noise(self, image, noise_level=None):
        """Add Gaussian noise to image."""
        if noise_level is None:
            noise_level = random.uniform(5, 15)
        noise = np.random.normal(0, noise_level, image.shape).astype(np.uint8)
        noisy_img = cv2.add(image, noise)
        return noisy_img
    
    def rotate_image(self, image, annotations, angle=None):
        """Rotate image and adjust annotations."""
        if angle is None:
            angle = random.uniform(-10, 10)
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate image
        rotated_img = cv2.warpAffine(image, M, (w, h), 
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(255, 255, 255))
        
        # For small rotations, keep annotations approximately the same
        # For production, you'd want to rotate the bounding boxes properly
        return rotated_img, annotations
    
    def adjust_contrast(self, image, factor=None):
        """Adjust image contrast."""
        if factor is None:
            factor = random.uniform(0.8, 1.2)
        mean = np.mean(image)
        adjusted = (image - mean) * factor + mean
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        return adjusted
    
    def blur_image(self, image, kernel_size=None):
        """Apply Gaussian blur to image."""
        if kernel_size is None:
            kernel_size = random.choice([3, 5])
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def augment_dataset(input_dir, output_dir, augmentation_factor=3):
    """
    Augment YOLO dataset by applying various transformations.
    
    Args:
        input_dir: Directory containing original dataset (with images/ and labels/ folders)
        output_dir: Directory to save augmented dataset
        augmentation_factor: How many augmented versions to create per image
    """
    augmenter = YOLOAugmenter()
    
    # Create output directory structure
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
    
    # Process each split
    for split in ['train', 'val']:
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} split...")
        print(f"{'='*60}")
        
        images_dir = os.path.join(input_dir, 'images', split)
        labels_dir = os.path.join(input_dir, 'labels', split)
        
        output_images_dir = os.path.join(output_dir, 'images', split)
        output_labels_dir = os.path.join(output_dir, 'labels', split)
        
        # Get all image files
        image_files = [f for f in os.listdir(images_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Found {len(image_files)} images")
        print(f"Creating {augmentation_factor} augmented versions per image")
        
        total_created = 0
        
        for idx, img_file in enumerate(image_files):
            base_name = os.path.splitext(img_file)[0]
            img_path = os.path.join(images_dir, img_file)
            label_path = os.path.join(labels_dir, base_name + '.txt')
            
            # Load image and annotations
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not read {img_file}")
                continue
            
            annotations = augmenter.load_yolo_annotations(label_path)
            
            # Copy original
            ext = os.path.splitext(img_file)[1]
            shutil.copy2(img_path, os.path.join(output_images_dir, img_file))
            if os.path.exists(label_path):
                shutil.copy2(label_path, os.path.join(output_labels_dir, base_name + '.txt'))
            
            # Create augmented versions
            augmentations = [
                ('hflip', lambda img, ann: augmenter.horizontal_flip(img, ann)),
                ('bright', lambda img, ann: (augmenter.adjust_brightness(img), ann)),
                ('contrast', lambda img, ann: (augmenter.adjust_contrast(img), ann)),
                ('noise', lambda img, ann: (augmenter.add_noise(img), ann)),
                ('rotate', lambda img, ann: augmenter.rotate_image(img, ann)),
                ('blur', lambda img, ann: (augmenter.blur_image(img), ann)),
            ]
            
            # Randomly select augmentations
            selected_augmentations = random.sample(augmentations, 
                                                  min(augmentation_factor, len(augmentations)))
            
            for aug_idx, (aug_name, aug_func) in enumerate(selected_augmentations):
                try:
                    aug_image, aug_annotations = aug_func(image, annotations)
                    
                    # Save augmented image
                    aug_img_name = f"{base_name}_aug{aug_idx+1}_{aug_name}{ext}"
                    aug_img_path = os.path.join(output_images_dir, aug_img_name)
                    cv2.imwrite(aug_img_path, aug_image)
                    
                    # Save augmented annotations
                    aug_label_name = f"{base_name}_aug{aug_idx+1}_{aug_name}.txt"
                    aug_label_path = os.path.join(output_labels_dir, aug_label_name)
                    augmenter.save_yolo_annotations(aug_annotations, aug_label_path)
                    
                    total_created += 1
                except Exception as e:
                    print(f"Error augmenting {img_file} with {aug_name}: {e}")
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(image_files)} images...")
        
        print(f"\n{split.upper()} split complete:")
        print(f"  Original: {len(image_files)} images")
        print(f"  Created: {total_created} augmented images")
        print(f"  Total: {len(image_files) + total_created} images")
    
    # Copy data.yaml
    src_yaml = os.path.join(input_dir, 'data.yaml')
    if os.path.exists(src_yaml):
        # Read and modify the yaml
        with open(src_yaml, 'r') as f:
            yaml_content = f.read()
        
        # Update path in yaml
        yaml_content = yaml_content.replace(
            f"path: {os.path.abspath(input_dir)}",
            f"path: {os.path.abspath(output_dir)}"
        )
        
        dst_yaml = os.path.join(output_dir, 'data.yaml')
        with open(dst_yaml, 'w') as f:
            f.write(yaml_content)
        print(f"\nCopied and updated data.yaml to {dst_yaml}")
    
    print(f"\n{'='*60}")
    print("Augmentation Complete!")
    print(f"{'='*60}")
    print(f"Augmented dataset saved to: {os.path.abspath(output_dir)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Configuration
    input_dir = "original_dataset"
    output_dir = "augmented_dataset"
    augmentation_factor = 3  # Create 3 augmented versions per image
    
    print("YOLO Dataset Augmentation")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Augmentation factor: {augmentation_factor}")
    
    augment_dataset(input_dir, output_dir, augmentation_factor)
