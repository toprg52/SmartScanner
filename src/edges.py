"""
edges.py — Edge detection, multi-scale Canny, and morphological operations.
"""

import cv2
import numpy as np
import src.config as config
from src.utils import log

def _canny_at_scale(gray: np.ndarray, scale: float) -> np.ndarray:
    """Helper to apply Canny edge detection at a given scale."""
    h, w = gray.shape[:2]
    if scale != 1.0:
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        g  = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_LINEAR)
    else:
        g = gray
    edges = cv2.Canny(g, config.CANNY_LO, config.CANNY_HI, apertureSize=3)
    if scale != 1.0:
        edges = cv2.resize(edges, (w, h), interpolation=cv2.INTER_LINEAR)
        _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
    return edges

def detect_edges(blurred: np.ndarray) -> tuple:
    """Build a robust combined edge map from a blurred grayscale image.
    
    Args:
        blurred: Grayscale blurred image.
        
    Returns:
        (padded_edges, sobel_norm)
        padded_edges: (H+2P) × (W+2P) uint8 padded edge map
        sobel_norm: H × W uint8 Sobel-X magnitude for Hough lines
    """
    log("[edges] Performing multi-scale Canny and Sobel edge detection.")
    
    # Multi-scale Canny
    e05 = _canny_at_scale(blurred, config.CANNY_SCALES[0])
    e10 = _canny_at_scale(blurred, config.CANNY_SCALES[1])
    e15 = _canny_at_scale(blurred, config.CANNY_SCALES[2])
    edge_ms = cv2.bitwise_or(e05, cv2.bitwise_or(e10, e15))

    # Sobel-X strengthens vertical (left/right document) edges
    sx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sx = np.abs(sx)
    sobel_norm = cv2.normalize(sx, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, sx_bin = cv2.threshold(sobel_norm, config.SOBEL_THRESH, 255, cv2.THRESH_BINARY)

    # Sobel-Y for horizontal (top/bottom) edges
    sy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sy = np.abs(sy)
    sy_norm = cv2.normalize(sy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, sy_bin = cv2.threshold(sy_norm, config.SOBEL_THRESH, 255, cv2.THRESH_BINARY)

    # Combine all edge sources
    combined = cv2.bitwise_or(edge_ms, cv2.bitwise_or(sx_bin, sy_bin))

    # Morphological closing to connect broken document borders
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (config.CLOSE_KSIZE, config.CLOSE_KSIZE))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Pad to prevent contour clipping at image boundary
    padded = cv2.copyMakeBorder(
        closed,
        config.PAD_SIZE, config.PAD_SIZE, config.PAD_SIZE, config.PAD_SIZE,
        cv2.BORDER_CONSTANT, value=0
    )

    return padded, sobel_norm
