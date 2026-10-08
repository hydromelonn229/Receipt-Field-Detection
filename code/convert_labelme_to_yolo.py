import json
import os
from PIL import Image

# Class mapping as per user requirement
CLASS_MAPPING = {
    'product_description': 0,
    'price': 1,
    'quantity_item': 2,
    'total_due_amount': 3
}

def labelme_to_yolo(labelme_json_path, images_folder, output_folder):
    """Convert a single labelme JSON to YOLO format."""
    try:
        with open(labelme_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {labelme_json_path}: {e}")
        return False
    
    # Get image dimensions
    image_width = data.get('imageWidth')
    image_height = data.get('imageHeight')
    
    # If dimensions not in JSON, try to get from image file
    if not image_width or not image_height:
        image_filename = data.get('imagePath', '')
        image_path = os.path.join(images_folder, image_filename)
        try:
            with Image.open(image_path) as img:
                image_width, image_height = img.size
        except:
            print(f"Could not get image dimensions for {labelme_json_path}")
            return False
    
    # Convert shapes to YOLO format
    yolo_lines = []
    for shape in data.get('shapes', []):
        label = shape.get('label', '')
        if label not in CLASS_MAPPING:
            continue
        
        class_id = CLASS_MAPPING[label]
        points = shape.get('points', [])
        
        if len(points) != 2:
            continue
        
        # Get bounding box coordinates
        x1, y1 = points[0]
        x2, y2 = points[1]
        
        # Calculate YOLO format (normalized center x, center y, width, height)
        x_center = ((x1 + x2) / 2.0) / image_width
        y_center = ((y1 + y2) / 2.0) / image_height
        width = abs(x2 - x1) / image_width
        height = abs(y2 - y1) / image_height
        
        # Ensure values are between 0 and 1
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    # Save YOLO annotation file
    base_name = os.path.splitext(os.path.basename(labelme_json_path))[0]
    output_path = os.path.join(output_folder, base_name + '.txt')
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in yolo_lines:
                f.write(line + '\n')
        return True
    except Exception as e:
        print(f"Error writing {output_path}: {e}")
        return False

def convert_all_labelme_to_yolo(labelme_folder, images_folder, output_folder):
    """Convert all labelme JSON files to YOLO format."""
    os.makedirs(output_folder, exist_ok=True)
    
    json_files = [f for f in os.listdir(labelme_folder) if f.endswith('.json')]
    json_files.sort(key=lambda x: int(x.split('.')[0]))
    
    print(f"Converting {len(json_files)} labelme files to YOLO format...")
    print(f"Class mapping: {CLASS_MAPPING}\n")
    
    success_count = 0
    for json_file in json_files:
        json_path = os.path.join(labelme_folder, json_file)
        if labelme_to_yolo(json_path, images_folder, output_folder):
            success_count += 1
            if success_count % 100 == 0:
                print(f"Processed {success_count}/{len(json_files)} files...")
    
    print(f"\n{'='*60}")
    print(f"Conversion Complete!")
    print(f"{'='*60}")
    print(f"Successfully converted: {success_count}/{len(json_files)} files")
    print(f"Output folder: {output_folder}")
    print(f"{'='*60}")

if __name__ == "__main__":
    labelme_folder = "labelme_data"
    images_folder = "images"
    output_folder = "yolo_labels_cleaned"
    
    convert_all_labelme_to_yolo(labelme_folder, images_folder, output_folder)
