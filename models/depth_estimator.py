"""Depth-like map generation helpers based on OpenCV gradients."""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def estimate_depth_like_map(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a depth-like visualization from one RGB image using OpenCV gradients."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is empty. Cannot estimate depth map.")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)

    depth_gray = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Invert before coloring so smoother/near-flat regions appear farther.
    depth_colored = cv2.applyColorMap(255 - depth_gray, cv2.COLORMAP_INFERNO)

    return depth_colored, depth_gray


def save_depth_output(depth_colored: np.ndarray, output_dir: Path, base_filename: str) -> str:
    """Save the colored depth map and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(base_filename).stem}_depth.png"

    success = cv2.imwrite(str(output_path), depth_colored)
    if not success:
        raise IOError(f"Failed to save depth output to {output_path}")

    return str(output_path)
