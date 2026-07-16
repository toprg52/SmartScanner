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

    # Fallback: pick the 4 hull points most spread from the centroid
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


def _border_penalty(corners: np.ndarray, img_shape: tuple) -> float:
    """Compute a graduated penalty for how many sides of the quad touch the image border.
    
    Each side that touches the image border adds DOC_BORDER_PENALTY_PER_SIDE to the penalty.
    The penalty is capped at 3 sides (anything beyond that covers the full image anyway).
    """
    h, w = img_shape[:2]
    mx = w * config.DOC_BORDER_MARGIN
    my = h * config.DOC_BORDER_MARGIN

    x_coords = corners[:, 0]
    y_coords = corners[:, 1]

    touches = 0
    if np.any(x_coords < mx):       touches += 1   # left border
    if np.any(x_coords > w - mx):   touches += 1   # right border
    if np.any(y_coords < my):       touches += 1   # top border
    if np.any(y_coords > h - my):   touches += 1   # bottom border

    return min(touches, 3) * config.DOC_BORDER_PENALTY_PER_SIDE


def _score_quad(corners: np.ndarray, img_shape: tuple) -> float:
    """Score a quad on multiple geometric factors. Returns -1.0 if hard filters fail.

    Scoring components:
      - Area coverage (dominant)
      - Rectangularity (soft — curved pages may score lower)
      - Aspect ratio (soft)
      - Compactness
      - Centrality (prefer contours near image center)
      - Convexity bonus (soft — not a hard filter)
      - Border penalty (graduated, per touching side)
    """
    h, w = img_shape[:2]
    img_area = float(h * w)

    cnt_arr = corners.reshape(-1, 1, 2).astype(np.float32)
    area = float(cv2.contourArea(cnt_arr))

    # ---- Hard filters ----
    if area < config.DOC_MIN_AREA_RATIO * img_area:
        return -1.0

    x, y, bw, bh = cv2.boundingRect(cnt_arr.astype(np.int32))
    bbox_area = float(max(bw * bh, 1))

    # Minimum bounding box size check — reject degenerate shapes
    if bw < 0.1 * w or bh < 0.1 * h:
        return -1.0

    rect_score = min(area / bbox_area, 1.0)
    # Soft floor — allow down to 0.35 for heavily curved pages
    if rect_score < 0.35:
        return -1.0

    # ---- Soft scoring components ----
    area_coverage = min(area / img_area, 1.0)

    aspect = bh / float(max(bw, 1))
    aspect_penalty = max(0.0, abs(aspect - config.DOC_TARGET_ASPECT) - config.DOC_ASPECT_TOLERANCE)
    aspect_score = 1.0 / (1.0 + aspect_penalty)

    peri = cv2.arcLength(cnt_arr, True)
    compactness = (4 * np.pi * area) / (peri * peri + 1e-6)

    # Centrality: prefer contours centered in the image
    cx_img, cy_img = w / 2.0, h / 2.0
    cx_cnt = x + bw / 2.0
    cy_cnt = y + bh / 2.0
    dist_norm = np.sqrt(((cx_cnt - cx_img) / w) ** 2 + ((cy_cnt - cy_img) / h) ** 2)
    centrality_score = max(0.0, 1.0 - dist_norm * 2.0)

    # Convexity: soft bonus (not a hard filter — curved pages may be slightly non-convex)
    convexity_bonus = 1.0 if cv2.isContourConvex(cnt_arr.astype(np.int32)) else 0.0

    # Border penalty: graduated per side that touches the image border
    penalty = _border_penalty(corners, img_shape)

    score = (
        0.35 * area_coverage
        + 0.20 * rect_score
        + 0.15 * aspect_score
        + 0.05 * compactness
        + config.DOC_CENTER_WEIGHT * centrality_score
        + config.DOC_CONVEXITY_WEIGHT * convexity_bonus
        - penalty
    )

    return score


def full_image_quad(img_shape: tuple) -> np.ndarray:
    """Return a quad that covers the full image with a small inset margin."""
    h, w = img_shape[:2]
    m = min(h, w) // 50      # ~2% inset
    return np.array([
        [m, m],
        [w - m, m],
        [w - m, h - m],
        [m, h - m],
    ], dtype=np.float32)


def find_document_contour(padded_edges: np.ndarray, img_shape: tuple) -> np.ndarray | None:
    """Search *padded_edges* for the best quadrilateral document boundary.

    Each candidate contour is scored independently. The best score must exceed
    DOC_MIN_SCORE (confidence threshold) for the contour to be accepted.
    If nothing passes, returns None — the pipeline will safely use the full-image fallback.

    Args:
        padded_edges: Edge map padded by PAD_SIZE.
        img_shape: Shape of the original (non-padded) image.

    Returns:
        (4, 2) float32 corners in original image space, or None if no reliable contour found.
    """
    log("[contours] Finding best document contour.")
    h_orig, w_orig = img_shape[:2]
    img_area = float(h_orig * w_orig)

    best_score = config.DOC_MIN_SCORE   # Only accept above this confidence threshold
    best_corners = None

    contours, _ = cv2.findContours(padded_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        raw_area = cv2.contourArea(cnt)
        # Skip tiny contours early for performance
        if raw_area < config.DOC_MIN_AREA_RATIO * img_area * 0.5:
            break

        approx = _contour_to_quad(cnt)
        if approx is None:
            continue

        # Convert from padded space to original image space
        corners = (approx.reshape(-1, 2).astype(np.float32)
                   - np.array([config.PAD_SIZE, config.PAD_SIZE], dtype=np.float32))
        corners[:, 0] = corners[:, 0].clip(0, w_orig - 1)
        corners[:, 1] = corners[:, 1].clip(0, h_orig - 1)

        score = _score_quad(corners, img_shape)

        if score > best_score:
            best_score = score
            best_corners = corners

    if best_corners is not None:
        log(f"[contours] Valid contour accepted (confidence: {best_score:.3f})")
    else:
        log("[contours] No contour exceeded confidence threshold — fallback will be used.")

    return best_corners
