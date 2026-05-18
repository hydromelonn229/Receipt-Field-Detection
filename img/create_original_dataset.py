import os
import shutil
from sklearn.model_selection import train_test_split

def create_yolo_dataset(images_folder, labels_folder, output_dir, test_size=0.2, random_state=42):
    """
    Create YOLO dataset structure with train/val split.
    
    Args:
        images_folder: Folder containing original images
        labels_folder: Folder containing YOLO label files (.txt)
        output_dir: Output directory for the dataset
        test_size: Fraction of data to use for validation (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
    """
    # Create directory structure
    dirs = {
        'train_images': os.path.join(output_dir, 'images', 'train'),
        'val_images': os.path.join(output_dir, 'images', 'val'),
        'train_labels': os.path.join(output_dir, 'labels', 'train'),
        'val_labels': os.path.join(output_dir, 'labels', 'val')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(images_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    
    print(f"Found {len(image_files)} images")
    
    # Filter images that have corresponding labels
    valid_images = []
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        label_file = base_name + '.txt'
        label_path = os.path.join(labels_folder, label_file)
        if os.path.exists(label_path):
            valid_images.append(img_file)
        else:
            print(f"Warning: No label found for {img_file}")
    
    print(f"Valid images with labels: {len(valid_images)}")
    
    # Split into train and validation
    train_imgs, val_imgs = train_test_split(
        valid_images, 
        test_size=test_size, 
        random_state=random_state
    )
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_imgs)} images ({100*(1-test_size):.0f}%)")
    print(f"  Validation: {len(val_imgs)} images ({100*test_size:.0f}%)")
    
    # Copy training files
    print("\nCopying training files...")
    for img_file in train_imgs:
        base_name = os.path.splitext(img_file)[0]
        
        # Copy image
        src_img = os.path.join(images_folder, img_file)
        dst_img = os.path.join(dirs['train_images'], img_file)
        shutil.copy2(src_img, dst_img)
        
        # Copy label
        label_file = base_name + '.txt'
        src_label = os.path.join(labels_folder, label_file)
        dst_label = os.path.join(dirs['train_labels'], label_file)
        shutil.copy2(src_label, dst_label)
    
    # Copy validation files
    print("Copying validation files...")
    for img_file in val_imgs:
        base_name = os.path.splitext(img_file)[0]
        
        # Copy image
        src_img = os.path.join(images_folder, img_file)
        dst_img = os.path.join(dirs['val_images'], img_file)
        shutil.copy2(src_img, dst_img)
        
        # Copy label
        label_file = base_name + '.txt'
        src_label = os.path.join(labels_folder, label_file)
        dst_label = os.path.join(dirs['val_labels'], label_file)
        shutil.copy2(src_label, dst_label)
    
    # Create data.yaml file
    yaml_content = f"""# YOLO Dataset Configuration
# Receipt Detection - 4 Classes

# Paths (relative to this file or absolute)
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

# Number of classes
nc: 4

# Class names
names:
  0: product_description
  1: price
  2: quantity_item
  3: total_due_amount
"""
    
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"\n{'='*60}")
    print("Dataset creation complete!")
    print(f"{'='*60}")
    print(f"Dataset location: {os.path.abspath(output_dir)}")
    print(f"Configuration file: {yaml_path}")
    print(f"\nDataset structure:")
    print(f"  {output_dir}/")
    print(f"    ├── images/")
    print(f"    │   ├── train/ ({len(train_imgs)} images)")
    print(f"    │   └── val/ ({len(val_imgs)} images)")
    print(f"    ├── labels/")
    print(f"    │   ├── train/ ({len(train_imgs)} labels)")
    print(f"    │   └── val/ ({len(val_imgs)} labels)")
    print(f"    └── data.yaml")
    print(f"{'='*60}")

if __name__ == "__main__":
    images_folder = "images"
    labels_folder = "yolo_labels_cleaned"
    output_dir = "original_dataset"
    
    create_yolo_dataset(images_folder, labels_folder, output_dir)
