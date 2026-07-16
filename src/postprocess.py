"""
postprocess.py — Output enhancement, CLAHE, whitening, and binarisation.
"""

import cv2
import numpy as np
import src.config as config
from src.utils import log

def _enhance_clahe(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE to the L channel (LAB) for contrast enhancement."""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_TILE_GRID)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

def _adaptive_threshold_output(img: np.ndarray) -> np.ndarray:
    """Return a binarised version suitable for OCR."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        config.ADAPT_BLOCK_SIZE, config.ADAPT_C
    )

def _whiten_background(img: np.ndarray) -> np.ndarray:
    """Normalise uneven lighting by dividing out a blurred background estimate."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        is_color = True
    else:
        gray = img.copy()
        is_color = False

    ksize = config.WHITEN_KERNEL_SIZE
    bg = cv2.GaussianBlur(gray, (ksize, ksize), 0).astype(np.float32)
    bg = np.maximum(bg, 1.0)

    norm = (gray.astype(np.float32) / bg * 255.0).clip(0, 255).astype(np.uint8)

    if is_color:
        out = np.zeros_like(img)
        for c in range(3):
            ch = img[:, :, c].astype(np.float32)
            out[:, :, c] = (ch / bg * 255.0).clip(0, 255).astype(np.uint8)
        return out
    return norm

def _sharpen(img: np.ndarray) -> np.ndarray:
    """Apply unsharp masking."""
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    strength = config.SHARPEN_STRENGTH
    return cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)

def enhance_output(img: np.ndarray, method: str = 'clahe', do_whiten: bool = True, do_sharpen: bool = False) -> np.ndarray:
    """Full postprocessing pipeline applied after dewarping.

    Args:
        img: Input image (BGR or grayscale).
        method: 'clahe' or 'adaptive'
        do_whiten: Apply background whitening.
        do_sharpen: Apply unsharp masking.

    Returns:
        Enhanced image.
    """
    log(f"[postprocess] Enhancing output (method={method}, whiten={do_whiten}, sharpen={do_sharpen}).")
    result = img.copy()
    if len(result.shape) == 2 and method != 'adaptive':
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    if do_whiten:
        result = _whiten_background(result)

    if method == 'adaptive':
        result = _adaptive_threshold_output(result)
    else:
        result = _enhance_clahe(result)

    if do_sharpen:
        result = _sharpen(result)

    return result
