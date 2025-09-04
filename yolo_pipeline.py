import fiftyone as fo
import fiftyone.zoo as foz
from ultralytics import YOLO
import os

def convert_fiftyone_to_yolo():
    """
    Loads the fiftyone quickstart dataset and exports it in YOLO format.
    """
    dataset = foz.load_zoo_dataset("quickstart")
    
    # Explicitly define train and validation splits
    train_dataset = dataset.match_tags("train")
    val_dataset = dataset.match_tags("validation")

    export_dir = "yolo_dataset"
    
    # Export the training and validation splits
    train_dataset.export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        split="train"
    )
    
    val_dataset.export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        split="val"
    )

    return os.path.join(export_dir, "dataset.yaml")

def train_yolo_model(data_yaml_path):
    """
    Trains a YOLOv8 model on the provided dataset.

    Args:
        data_yaml_path (str): The path to the data.yaml file for the dataset.
    """
    model = YOLO("yolov8n.pt")
    model.train(data=data_yaml_path, epochs=10, imgsz=640)

if __name__ == "__main__":
    data_yaml = convert_fiftyone_to_yolo()
    train_yolo_model(data_yaml)
    print("YOLO model training pipeline completed successfully.")

