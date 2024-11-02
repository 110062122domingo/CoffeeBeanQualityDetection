from ultralytics import YOLO
import argparse
import os

def train_model(model_path, epochs, imgsz, batch_size, device):
    """
    Train a YOLOv8 model with specified parameters.
    
    Args:
        model_path (str): Path to the pre-trained YOLOv8 model.
        epochs (int): Number of training epochs.
        imgsz (int): Size of images for training.
        batch_size (int): Batch size for training.
        device (int): GPU device ID.
    """
    try:
        model = YOLO(model_path)
        model.train(
            data=r'D:\coffee\CoffeeBeanQualityDetection\data\1014\data.yaml',
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device
        )
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model for coffee bean quality detection.")
    parser.add_argument('--model_path', type=str, default=r'yolov8s.pt', help='Path to the YOLOv8 model file.')
    parser.add_argument('--epochs', type=int, default=240, help='Number of training epochs.')
    parser.add_argument('--imgsz', type=int, default=640, help='Size of images for training.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID.')

    args = parser.parse_args()

    train_model(args.model_path, args.epochs, args.imgsz, args.batch_size, args.device)

if __name__ == '__main__':
    main()
