"""
utils.py — Shared helper functions and centralized logging.
"""

import os
import cv2
import numpy as np
import src.config as config

def log(msg: str):
    """Log a message to stdout if DEBUG is enabled."""
    if config.DEBUG:
        print(msg)

def save_debug(img: np.ndarray, name: str):
    """Save an intermediate debug image to outputs/debug if SAVE_STEPS is enabled."""
    if config.SAVE_STEPS:
        debug_dir = os.path.join("outputs", "debug")
        os.makedirs(debug_dir, exist_ok=True)
        path = os.path.join(debug_dir, f"{name}.png")
        cv2.imwrite(path, img)

def save_comparison(original: np.ndarray, result: np.ndarray, name: str):
    """Save side-by-side comparison to outputs/comparisons if SAVE_COMPARISONS is enabled."""
    if config.SAVE_COMPARISONS:
        comp_dir = os.path.join("outputs", "comparisons")
        os.makedirs(comp_dir, exist_ok=True)
        
        target_h = min(original.shape[0], 1600)

        def _to_bgr_resized(im, h):
            bgr = im if len(im.shape) == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            w   = int(bgr.shape[1] * h / bgr.shape[0])
            return cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)

        left  = _to_bgr_resized(original, target_h)
        right = _to_bgr_resized(result,   target_h)

        h = min(left.shape[0], right.shape[0])
        side = np.hstack([left[:h], right[:h]])
        
        path = os.path.join(comp_dir, f"{name}_comparison.jpg")
        cv2.imwrite(path, side)

def draw_quad(img: np.ndarray, corners: np.ndarray, color=(0, 0, 255),
              thickness: int = 3) -> np.ndarray:
    """Draw a filled/outlined quadrilateral on a copy of *img*."""
    vis = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    pts = corners.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=thickness,
                  lineType=cv2.LINE_AA)
    for pt in corners:
        cv2.circle(vis, tuple(pt.astype(int)), 8, (255, 255, 0), -1, cv2.LINE_AA)
    return vis

def draw_curves_debug(img: np.ndarray, top_curve: np.ndarray,
                      bottom_curve: np.ndarray,
                      curvature_score: float) -> np.ndarray:
    """Visualise the top/bottom boundary curves with the curvature score overlay."""
    vis = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = vis.shape[:2]

    def _draw_curve(curve, color):
        xs = np.linspace(0, w - 1, len(curve)).astype(int)
        for i in range(len(xs) - 1):
            pt1 = (xs[i],     int(np.clip(curve[i],     0, h - 1)))
            pt2 = (xs[i + 1], int(np.clip(curve[i + 1], 0, h - 1)))
            cv2.line(vis, pt1, pt2, color, 2, cv2.LINE_AA)

    _draw_curve(top_curve,    (0,   200, 255))   # cyan-ish
    _draw_curve(bottom_curve, (255, 100,   0))   # orange

    label = f"Curvature: {curvature_score:.4f}"
    cv2.putText(vis, label, (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, label, (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
    return vis
