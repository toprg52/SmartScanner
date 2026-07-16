"""
transform.py — Hough-line corner refinement and perspective crop.
"""

import cv2
import numpy as np
import src.config as config
from src.utils import log

def _detect_hough_lines(edge_map: np.ndarray) -> tuple:
    """Return (h_lines, v_lines) from probabilistic Hough on *edge_map*."""
    segs = cv2.HoughLinesP(
        edge_map,
        config.HOUGH_RHO, config.HOUGH_THETA, config.HOUGH_THRESHOLD,
        minLineLength=config.HOUGH_MIN_LEN,
        maxLineGap=config.HOUGH_MAX_GAP
    )
    h_lines, v_lines = [], []
    if segs is None:
        return h_lines, v_lines
    for seg in segs[:, 0]:
        x1, y1, x2, y2 = seg
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 25:
            h_lines.append(seg)
        elif angle > 65:
            v_lines.append(seg)
    return h_lines, v_lines

def _line_intersection(sa, sb):
    x1, y1, x2, y2 = [float(v) for v in sa]
    x3, y3, x4, y4 = [float(v) for v in sb]
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
    return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], dtype=np.float32)

def refine_corners(corners: np.ndarray, sobel_map: np.ndarray) -> np.ndarray:
    """Snap each corner to the nearest Hough H×V intersection from sobel edge map."""
    log("[transform] Refining corners via Hough intersections.")
    h_lines, v_lines = _detect_hough_lines(sobel_map)

    if not h_lines or not v_lines:
        return corners

    isects = []
    for h in h_lines:
        for v in v_lines:
            pt = _line_intersection(h, v)
            if pt is not None:
                isects.append(pt)

    if not isects:
        return corners

    isects = np.array(isects, dtype=np.float32)
    refined = corners.copy()

    for i, c in enumerate(corners):
        dists = np.linalg.norm(isects - c, axis=1)
        idx = int(np.argmin(dists))
        if dists[idx] < config.CORNER_SNAP_DIST:
            refined[i] = isects[idx]

    return refined

def _order_corners(corners: np.ndarray) -> np.ndarray:
    """Return corners as [top-left, top-right, bottom-right, bottom-left]."""
    pts = corners.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL: smallest x+y
    rect[2] = pts[np.argmax(s)]   # BR: largest  x+y

    diff = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(diff)]  # TR: smallest y-x
    rect[3] = pts[np.argmax(diff)]  # BL: largest  y-x

    return rect

def apply_perspective_transform(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp *img* so the four *corners* map to an upright rectangle."""
    log("[transform] Applying perspective warp.")
    rect = _order_corners(corners)
    tl, tr, br, bl = rect

    out_w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    out_h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))

    if out_w < 10 or out_h < 10:
        return img.copy()

    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(
        img, M, (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
