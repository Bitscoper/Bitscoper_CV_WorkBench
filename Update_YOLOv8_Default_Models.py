# By Abdullah As-Sadeed

import os
import glob
from ultralytics import YOLO

# Local modules
import Default_Paths

YOLOv8_default_Models = glob.glob(
    str(Default_Paths.YOLO_DEFAULT_MODEL_DIRECTORY) + "/*"
)

for YOLO_default_model in YOLOv8_default_Models:
    os.remove(YOLO_default_model)
    print(f"Deleted {YOLO_default_model}")

YOLO_model_suffixes = ["", "-seg", "-pose"]
YOLO_model_weight_suffixes = ["n", "s", "m", "l", "x"]

for YOLO_model_suffix in YOLO_model_suffixes:
    for YOLO_model_weight_suffix in YOLO_model_weight_suffixes:
        YOLO_model_path = (
            str(Default_Paths.YOLO_DEFAULT_MODEL_DIRECTORY)
            + "/yolov8"
            + YOLO_model_weight_suffix
            + YOLO_model_suffix
            + ".pt"
        )

        try:
            YOLO_model = YOLO(YOLO_model_path)

        except Exception as exception:
            print(f"Error downloading YOLOv8 model {YOLO_model_path}: {exception}")
