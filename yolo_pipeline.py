import fiftyone as fo
import fiftyone.zoo as foz
from ultralytics import YOLO
import os
import time
from tqdm import tqdm

def convert_fiftyone_to_yolo():
    """
    Loads the fiftyone quickstart dataset and exports it in YOLO format.
    """
    print("Loading FiftyOne quickstart dataset... (This may take a few minutes if downloading)")
    start_time = time.time()
    dataset = foz.load_zoo_dataset("quickstart")
    load_time = time.time() - start_time
    print(".2f")

    print("Converting dataset to YOLO format...")
    export_dir = "yolo_dataset"
    dataset.export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5DatasetType,
        label_field="ground_truth",
    )
    print("Conversion complete. Dataset saved to yolo_dataset/")

    return os.path.join(export_dir, "dataset.yaml")

def train_yolo_model(data_yaml_path):
    """
    Trains a YOLOv8 model on the provided dataset.

    Args:
        data_yaml_path (str): The path to the data.yaml file for the dataset.
    """
    print("Loading pretrained YOLOv8 model...")
    model = YOLO("yolov8n.pt")
    print("Model loaded successfully.")

    print("Starting YOLOv8 model training for 10 epochs... (This may take 10-30 minutes depending on your hardware)")
    start_time = time.time()
    model.train(data=data_yaml_path, epochs=10, imgsz=640)
    train_time = time.time() - start_time
    print(".2f")

if __name__ == "__main__":
    data_yaml = convert_fiftyone_to_yolo()
    train_yolo_model(data_yaml)

