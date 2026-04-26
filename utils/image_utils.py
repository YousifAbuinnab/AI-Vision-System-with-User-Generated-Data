"""Image and file utility helpers used by the Streamlit app."""

import io
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def ensure_directories(paths: Iterable[Path]) -> None:
    """Create directories when they do not already exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _sanitize_filename(name: str) -> str:
    """Sanitize filename stem to safe characters."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", name)
    return cleaned.strip("_") or "image"


def generate_timestamped_filename(original_name: str) -> str:
    """Create a safe unique filename while preserving extension."""
    original_path = Path(original_name)
    stem = _sanitize_filename(original_path.stem)

    suffix = original_path.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        suffix = ".png"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{stem}{suffix}"


def save_uploaded_file(uploaded_file, upload_dir: Path) -> Tuple[str, Image.Image]:
    """Save uploaded Streamlit file and return saved path plus PIL image."""
    if uploaded_file is None:
        raise ValueError("No file provided.")

    file_bytes = uploaded_file.getvalue()
    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    filename = generate_timestamped_filename(uploaded_file.name)
    save_path = upload_dir / filename
    image.save(save_path)

    return str(save_path), image


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to OpenCV BGR format."""
    rgb_array = np.array(image)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR image to RGB for Streamlit display."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
