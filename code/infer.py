from ultralytics import YOLO

model = YOLO(r"code/best.pt")
results = model.predict(source=r"code/581.jpg", conf=0.25, save=True)
print("Saved to:", results[0].save_dir)
