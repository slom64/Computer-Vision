from ultralytics import YOLO

def main():
    print("Loading YOLO11 Nano model...")
    # Load a model
    model = YOLO("yolo11n.pt")

    print("Starting training...")
    # Train the model on our custom dataset
    # We use a small number of epochs (e.g. 5) just for rapid prototyping right now.
    # It can be increased later for full accuracy.
    results = model.train(data="dataset.yaml", epochs=5, imgsz=512, device=0)

if __name__ == '__main__':
    main()
