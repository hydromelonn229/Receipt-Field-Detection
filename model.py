from ultralytics import YOLO
import os

# Get absolute path to the dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "YOLODataset_aug", "dataset.yaml")

model = YOLO("yolov8s.pt")

model.train(
    data=data_path,
    epochs=100,
    imgsz=640,
    batch=16,
    name="partial_invoice_model"
)

