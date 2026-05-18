"""
YOLOv8 Training Script for Receipt Detection
Trains a model on the augmented dataset and saves the best model
"""
import os
from ultralytics import YOLO
import torch

def train_yolo_model(data_yaml, epochs=100, imgsz=640, batch_size=16, model_size='n'):
    """
    Train YOLOv8 model on receipt detection dataset.
    
    Args:
        data_yaml: Path to data.yaml configuration file
        epochs: Number of training epochs
        imgsz: Image size for training
        batch_size: Batch size for training
        model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
    """
    print("="*60)
    print("YOLOv8 Receipt Detection Training")
    print("="*60)
    print(f"Data config: {data_yaml}")
    print(f"Model size: YOLOv8{model_size}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("="*60)
    
    # Load a pretrained YOLOv8 model
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        project='runs/detect',
        name='receipt_detection',
        patience=20,  # Early stopping patience
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        device=0 if torch.cuda.is_available() else 'cpu',
        workers=4,
        plots=True,  # Generate training plots
        verbose=True
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best model saved at: runs/detect/receipt_detection/weights/best.pt")
    print(f"Last model saved at: runs/detect/receipt_detection/weights/last.pt")
    print("="*60)
    
    # Validate the model
    print("\nValidating model on validation set...")
    metrics = model.val()
    
    print("\nValidation Metrics:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print("="*60)
    
    return model, results

def test_model(model_path, test_images_dir, output_dir='runs/detect/test'):
    """Test the trained model on test images."""
    model = YOLO(model_path)
    
    # Run inference
    results = model.predict(
        source=test_images_dir,
        save=True,
        project=output_dir,
        name='predictions',
        conf=0.25,  # Confidence threshold
        iou=0.45,   # NMS IOU threshold
        show_labels=True,
        show_conf=True
    )
    
    print(f"\nTest predictions saved to: {output_dir}/predictions")
    return results

if __name__ == "__main__":
    # Configuration
    data_yaml = "augmented_dataset/data.yaml"
    
    # Check if data.yaml exists
    if not os.path.exists(data_yaml):
        print(f"Error: {data_yaml} not found!")
        print("Please run the augmentation script first.")
        exit(1)
    
    # Check if ultralytics is installed
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not found!")
        print("Please install it using: pip install ultralytics")
        exit(1)
    
    # Training parameters
    EPOCHS = 100
    IMAGE_SIZE = 640
    BATCH_SIZE = 16
    MODEL_SIZE = 'n'  # Options: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
    
    # Train the model
    model, results = train_yolo_model(
        data_yaml=data_yaml,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        model_size=MODEL_SIZE
    )
    
    print("\nTo use the trained model:")
    print("  model = YOLO('runs/detect/receipt_detection/weights/best.pt')")
    print("  results = model.predict('path/to/image.jpg')")
