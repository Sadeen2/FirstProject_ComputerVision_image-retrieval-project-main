from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np



# Paths configuration

# project_root/src/dataset.py  -> parents[1] = project_root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# data \ raw \ caltech-101 \ 101_ObjectCategories \ 101_ObjectCategories \ accordion ...
CALTECH_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "caltech-101"
    / "101_ObjectCategories"
    / "101_ObjectCategories"
)


def list_images(
    extensions=(".jpg", ".jpeg", ".png")
) -> List[Tuple[Path, str]]:
    """
    Scan Caltech-101 and return a list of (image_path, class_label).

    class_label = name of the folder (e.g. 'airplanes', 'accordion', ...)
    """
    if not CALTECH_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset folder not found:\n  {CALTECH_ROOT}\n"
            "Make sure you have structure like:\n"
            "  data/raw/caltech-101/101_ObjectCategories/101_ObjectCategories/accordion/...\n"
        )

    items: List[Tuple[Path, str]] = []

    # Each subfolder is a class (accordion, airplanes, ...)
    for class_dir in sorted(CALTECH_ROOT.iterdir()):
        if not class_dir.is_dir():
            continue

        label = class_dir.name

        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in extensions:
                items.append((img_path, label))

    return items


def load_image(path: Path, target_size=(256, 256)) -> np.ndarray:
    """
    Read an image from disk and resize it to target_size (width, height).

    Returns BGR image (OpenCV format).
    """
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    return img


def load_image_rgb(path: Path, target_size=(256, 256)) -> np.ndarray:
    """
    Read image and convert it to RGB (useful for matplotlib).
    """
    img_bgr = load_image(path, target_size)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb



