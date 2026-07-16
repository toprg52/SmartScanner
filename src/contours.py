"""
contours.py — Contour search, simplification, and scoring logic.
"""

import cv2
import numpy as np
import src.config as config
from src.utils import log

def _contour_to_quad(cnt) -> np.ndarray | None:
    """Simplify *cnt* to exactly 4 points using multiple epsilon values."""
    hull = cv2.convexHull(cnt)
    hull_peri = cv2.arcLength(hull, True)

    for eps in config.APPROX_EPSILONS:
        approx = cv2.approxPolyDP(hull, eps * hull_peri, True)
        if len(approx) == 4:
            return approx

    # Last resort: pick the 4 hull points that are most spread out
    if len(hull) >= 4:
        pts = hull[:, 0, :]
        centroid = pts.mean(axis=0)
        dists = np.linalg.norm(pts - centroid, axis=1)
        idxs = np.argsort(dists)[-4:]
        pts4 = pts[idxs]
        angles = np.arctan2(pts4[:, 1] - centroid[1], pts4[:, 0] - centroid[0])
        pts4 = pts4[np.argsort(angles)]
        return pts4.reshape(4, 1, 2).astype(np.int32)

    return None

def _score_quad(corners: np.ndarray, img_area: float) -> float:
    """Score a quad on area coverage, rectangularity, and aspect ratio.

    Returns -1 if it fails hard filters, otherwise a value in (0, 1].
    """
    cnt_arr = corners.reshape(-1, 1, 2).astype(np.float32)
    area = float(cv2.contourArea(cnt_arr))

    if area < config.DOC_MIN_AREA_RATIO * img_area:
        return -1.0

    x, y, bw, bh = cv2.boundingRect(cnt_arr.astype(np.int32))
    bbox_area = float(max(bw * bh, 1))
    rect_score = area / bbox_area

    if rect_score < config.DOC_MIN_RECT_SCORE:
        return -1.0

    # Area coverage: how much of the image does this quad cover?
    area_coverage = min(area / img_area, 1.0)

    aspect = bh / float(max(bw, 1))
    aspect_penalty = max(0.0, abs(aspect - config.DOC_TARGET_ASPECT) - config.DOC_ASPECT_TOLERANCE)
    aspect_score = 1.0 / (1.0 + aspect_penalty)

    peri = cv2.arcLength(cnt_arr, True)
    compactness = (4 * np.pi * area) / (peri * peri + 1e-6)

    # Area coverage is dominant: we want the LARGEST valid quad
    return (0.50 * area_coverage +
            0.20 * rect_score +
            0.15 * aspect_score +
            0.15 * compactness)

def full_image_quad(img_shape: tuple) -> np.ndarray:
    """Return a quad that covers the full image with a small inset margin."""
    h, w = img_shape[:2]
    m = min(h, w) // 50      # 2% inset
    return np.array([
        [m, m],
        [w - m, m],
        [w - m, h - m],
        [m, h - m],
    ], dtype=np.float32)

def find_document_contour(padded_edges: np.ndarray, img_shape: tuple) -> np.ndarray | None:
    """Search *padded_edges* for the best quadrilateral document boundary.

    The full-image quad is always scored as a baseline candidate. The
    detected contour is only returned if it scores strictly higher,
    preventing a narrow partial-page strip from winning over the full page.

    Args:
        padded_edges: Edge map padded by PAD_SIZE.
        img_shape: Shape of the original (non-padded) image.

    Returns:
        (4, 2) float32 corners in original image space, or None if fallback wins.
    """
    log("[contours] Finding best document contour.")
    h_orig, w_orig = img_shape[:2]
    img_area = float(h_orig * w_orig)

    # Baseline: score the full-image quad
    fallback_corners = full_image_quad(img_shape)
    best_score = _score_quad(fallback_corners, img_area)
    best_corners = None   # None means "use fallback"

    contours, _ = cv2.findContours(padded_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process largest contours first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        raw_area = cv2.contourArea(cnt)
        if raw_area < config.DOC_MIN_AREA_RATIO * img_area * 0.5:
            break

        approx = _contour_to_quad(cnt)
        if approx is None:
            continue

        # Convert from padded space to original image space
        corners = (approx.reshape(-1, 2).astype(np.float32) - 
                   np.array([config.PAD_SIZE, config.PAD_SIZE], dtype=np.float32))
        corners[:, 0] = corners[:, 0].clip(0, w_orig - 1)
        corners[:, 1] = corners[:, 1].clip(0, h_orig - 1)

        score = _score_quad(corners, img_area)

        if score > best_score:
            best_score = score
            best_corners = corners

    return best_corners
