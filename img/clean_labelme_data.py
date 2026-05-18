import json
import os
from collections import Counter

# Define the label mapping and labels to remove
LABELS_TO_REMOVE = {'DocumentDate', 'SellerVAT', 'SellerName', 'SellerVATNumber'}

# Mapping from old labels to new standardized labels
LABEL_MAPPING = {
    'ProductDescription': 'product_description',
    'product_description': 'product_description',
    'Quantity': 'quantity_item',
    'quantity_item': 'quantity_item',
    'Price': 'price',
    'price': 'price',
    'TotalDue': 'total_due_amount',
    'total_due_amount': 'total_due_amount',
}

# New class order (as per user request)
CLASS_ORDER = ['product_description', 'price', 'quantity_item', 'total_due_amount']

def clean_labelme_file(json_path):
    """Clean a single labelme JSON file by removing unwanted labels and standardizing others."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return False

    # Filter and update shapes
    new_shapes = []
    removed_count = 0
    updated_count = 0
    
    for shape in data.get('shapes', []):
        label = shape.get('label', '')
        
        # Remove unwanted labels
        if label in LABELS_TO_REMOVE:
            removed_count += 1
            continue
        
        # Standardize label names
        if label in LABEL_MAPPING:
            new_label = LABEL_MAPPING[label]
            if label != new_label:
                updated_count += 1
            shape['label'] = new_label
            new_shapes.append(shape)
        else:
            # Keep unknown labels as is
            new_shapes.append(shape)
    
    # Update the data with cleaned shapes
    data['shapes'] = new_shapes
    
    # Write back to file
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        return True, removed_count, updated_count
    except Exception as e:
        print(f"Error writing {json_path}: {e}")
        return False, 0, 0

def clean_all_labelme_files(labelme_folder):
    """Clean all labelme JSON files in the folder."""
    json_files = [f for f in os.listdir(labelme_folder) if f.endswith('.json')]
    json_files.sort(key=lambda x: int(x.split('.')[0]))
    
    print(f"Processing {len(json_files)} labelme files...")
    print(f"Removing labels: {', '.join(LABELS_TO_REMOVE)}")
    print(f"Standardizing labels to: {', '.join(CLASS_ORDER)}\n")
    
    total_removed = 0
    total_updated = 0
    total_processed = 0
    
    for json_file in json_files:
        json_path = os.path.join(labelme_folder, json_file)
        result = clean_labelme_file(json_path)
        
        if result:
            success, removed, updated = result
            if success:
                total_processed += 1
                total_removed += removed
                total_updated += updated
    
    print(f"\n{'='*60}")
    print(f"Cleaning Complete!")
    print(f"{'='*60}")
    print(f"Files processed: {total_processed}")
    print(f"Labels removed: {total_removed}")
    print(f"Labels standardized: {total_updated}")
    print(f"{'='*60}")
    
    # Verify the results
    print("\nVerifying cleaned data...")
    label_counts = Counter()
    
    for json_file in json_files:
        json_path = os.path.join(labelme_folder, json_file)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for shape in data.get('shapes', []):
                    label = shape.get('label', '')
                    if label:
                        label_counts[label] += 1
        except:
            pass
    
    print("\nFinal Label Distribution:")
    print(f"{'='*60}")
    for i, class_name in enumerate(CLASS_ORDER):
        count = label_counts.get(class_name, 0)
        print(f"Class {i} - {class_name:25} : {count:5} annotations")
    
    # Check for any unexpected labels
    unexpected = set(label_counts.keys()) - set(CLASS_ORDER)
    if unexpected:
        print(f"\nWarning: Found unexpected labels: {unexpected}")
    
    print(f"{'='*60}")
    print(f"Total annotations: {sum(label_counts.values())}")

if __name__ == "__main__":
    labelme_folder = "labelme_data"
    clean_all_labelme_files(labelme_folder)
