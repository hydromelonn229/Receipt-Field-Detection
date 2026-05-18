import os
from collections import Counter

# Class names
CLASS_NAMES = {
    0: 'product_description',
    1: 'price',
    2: 'quantity_item',
    3: 'total_due_amount'
}

def analyze_yolo_labels(labels_dir):
    """Analyze YOLO format label files."""
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    class_counts = Counter()
    bbox_counts = []
    total_annotations = 0
    files_with_labels = 0
    empty_files = 0
    
    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        if len(lines) == 0:
            empty_files += 1
        else:
            files_with_labels += 1
            bbox_counts.append(len(lines))
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(float(parts[0]))  # Convert float to int
                class_counts[class_id] += 1
                total_annotations += 1
    
    return {
        'total_files': len(label_files),
        'files_with_labels': files_with_labels,
        'empty_files': empty_files,
        'class_counts': class_counts,
        'total_annotations': total_annotations,
        'bbox_counts': bbox_counts
    }

# Analyze train and val sets
base_dir = 'YOLODataset_aug'
train_labels = os.path.join(base_dir, 'labels', 'train')
val_labels = os.path.join(base_dir, 'labels', 'val')

print('='*70)
print('YOLO DATASET AUGMENTED - LABEL ANALYSIS')
print('='*70)

# Count images
train_images = len([f for f in os.listdir(os.path.join(base_dir, 'images', 'train')) if f.endswith(('.jpg', '.png', '.jpeg'))])
val_images = len([f for f in os.listdir(os.path.join(base_dir, 'images', 'val')) if f.endswith(('.jpg', '.png', '.jpeg'))])

print(f'\nDataset Size:')
print(f'  Train images: {train_images}')
print(f'  Val images:   {val_images}')
print(f'  Total images: {train_images + val_images}')

# Analyze train set
print(f'\n{"="*70}')
print('TRAIN SET ANALYSIS')
print('='*70)
train_stats = analyze_yolo_labels(train_labels)

print(f'\nFiles:')
print(f'  Total label files: {train_stats["total_files"]}')
print(f'  Files with labels: {train_stats["files_with_labels"]}')
print(f'  Empty files:       {train_stats["empty_files"]}')

print(f'\nAnnotations:')
print(f'  Total annotations: {train_stats["total_annotations"]}')
if train_stats['bbox_counts']:
    print(f'  Avg per image:     {sum(train_stats["bbox_counts"]) / len(train_stats["bbox_counts"]):.2f}')
    print(f'  Min per image:     {min(train_stats["bbox_counts"])}')
    print(f'  Max per image:     {max(train_stats["bbox_counts"])}')

print(f'\nClass Distribution:')
for class_id in sorted(train_stats['class_counts'].keys()):
    count = train_stats['class_counts'][class_id]
    percentage = (count / train_stats['total_annotations'] * 100) if train_stats['total_annotations'] > 0 else 0
    print(f'  Class {class_id} ({CLASS_NAMES.get(class_id, "unknown"):25}): {count:5} ({percentage:5.1f}%)')

# Analyze val set
print(f'\n{"="*70}')
print('VALIDATION SET ANALYSIS')
print('='*70)
val_stats = analyze_yolo_labels(val_labels)

print(f'\nFiles:')
print(f'  Total label files: {val_stats["total_files"]}')
print(f'  Files with labels: {val_stats["files_with_labels"]}')
print(f'  Empty files:       {val_stats["empty_files"]}')

print(f'\nAnnotations:')
print(f'  Total annotations: {val_stats["total_annotations"]}')
if val_stats['bbox_counts']:
    print(f'  Avg per image:     {sum(val_stats["bbox_counts"]) / len(val_stats["bbox_counts"]):.2f}')
    print(f'  Min per image:     {min(val_stats["bbox_counts"])}')
    print(f'  Max per image:     {max(val_stats["bbox_counts"])}')

print(f'\nClass Distribution:')
for class_id in sorted(val_stats['class_counts'].keys()):
    count = val_stats['class_counts'][class_id]
    percentage = (count / val_stats['total_annotations'] * 100) if val_stats['total_annotations'] > 0 else 0
    print(f'  Class {class_id} ({CLASS_NAMES.get(class_id, "unknown"):25}): {count:5} ({percentage:5.1f}%)')

# Overall summary
print(f'\n{"="*70}')
print('OVERALL SUMMARY')
print('='*70)
total_annotations = train_stats['total_annotations'] + val_stats['total_annotations']
print(f'\nTotal Dataset:')
print(f'  Images:      {train_images + val_images}')
print(f'  Annotations: {total_annotations}')

print(f'\nCombined Class Distribution:')
combined_counts = Counter()
for class_id in range(4):
    combined_counts[class_id] = train_stats['class_counts'].get(class_id, 0) + val_stats['class_counts'].get(class_id, 0)

for class_id in sorted(combined_counts.keys()):
    count = combined_counts[class_id]
    percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
    print(f'  Class {class_id} ({CLASS_NAMES[class_id]:25}): {count:5} ({percentage:5.1f}%)')

print('='*70)

# Check for class imbalance
print('\nClass Balance Analysis:')
if total_annotations > 0:
    max_count = max(combined_counts.values())
    min_count = min(combined_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else 0
    print(f'  Imbalance ratio (max/min): {imbalance_ratio:.2f}')
    if imbalance_ratio > 3:
        print('  ⚠️  WARNING: Dataset is imbalanced (ratio > 3)')
    else:
        print('  ✓ Dataset is reasonably balanced')

print('='*70)
