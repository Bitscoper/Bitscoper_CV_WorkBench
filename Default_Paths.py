# By Abdullah As-Sadeed

import sys
from pathlib import Path

file_path = Path(__file__).resolve()
root_path = file_path.parent

if root_path not in sys.path:
    sys.path.append(str(root_path))

ROOT = root_path.relative_to(Path.cwd())

YOLO_DEFAULT_MODEL_DIRECTORY = ROOT / "YOLOv8-Weights"
