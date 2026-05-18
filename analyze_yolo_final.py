import os
from collections import Counter

# Path to the yolo_dataset_final labels
base_path = r"d:\Assumption University\AIConcepts\Project\img\yolo_dataset_final\labels"

# Class names
class_names = ['ProductDescription', 'Quantity', 'Price', 'TotalDue']

# Counter for each class
class_counts = Counter()

# Analyze train labels
train_path = os.path.join(base_path, 'train')
val_path = os.path.join(base_path, 'val')

for split, split_path in [('train', train_path), ('val', val_path)]:
    if os.path.exists(split_path):
        label_files = [f for f in os.listdir(split_path) if f.endswith('.txt')]
        print(f"\n{split.upper()} set: {len(label_files)} label files")
        
        split_counts = Counter()
        for label_file in label_files:
            label_path = os.path.join(split_path, label_file)
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        class_id = int(line.split()[0])
                        split_counts[class_id] += 1
                        class_counts[class_id] += 1
        
        print(f"\n{split.upper()} class distribution:")
        for class_id in sorted(split_counts.keys()):
            print(f"  Class {class_id} ({class_names[class_id]}): {split_counts[class_id]} entries")

# Overall summary
print("\n" + "="*60)
print("OVERALL SUMMARY")
print("="*60)
print(f"\nTotal number of classes: {len(class_names)}")
print(f"Classes: {class_names}")
print(f"\nTotal entries per class:")
for class_id in sorted(class_counts.keys()):
    print(f"  Class {class_id} ({class_names[class_id]}): {class_counts[class_id]} entries")

print(f"\nGrand total entries: {sum(class_counts.values())}")
