"""
config.py — Centralized configuration parameters.
"""

import numpy as np

# ---------------------------------------------------------------------------
# DEBUG FLAGS
# ---------------------------------------------------------------------------
DEBUG = False
SAVE_STEPS = False
SAVE_COMPARISONS = True
USE_CURVATURE = True

# ---------------------------------------------------------------------------
# PREPROCESS PARAMS
# ---------------------------------------------------------------------------
MEDIAN_BLUR_KSIZE = 7  # Keep at 7 — aggressive enough for grain, safe enough for doc edges
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# ---------------------------------------------------------------------------
# EDGE PARAMS
# ---------------------------------------------------------------------------
# Canny thresholds are now dynamic via Otsu. CANNY_LO/HI are removed.
CANNY_SCALES = [0.5, 1.0, 1.5]
SOBEL_THRESH = 35
CLOSE_KSIZE = 9
OPEN_KSIZE = 3   # Morph opening before closing to prune thin noise (spiral rings, grain)
PAD_SIZE = 30

# ---------------------------------------------------------------------------
# CONTOUR PARAMS
# ---------------------------------------------------------------------------
DOC_MIN_AREA_RATIO = 0.10      # Reduced to allow smaller pages/notebooks
DOC_TARGET_ASPECT = 1.41
DOC_ASPECT_TOLERANCE = 0.80

# Scoring / validation parameters
DOC_MIN_SCORE = 0.55           # Confidence threshold — only warp when detection is reliable
DOC_CENTER_WEIGHT = 0.15       # Weight for contour centrality (soft heuristic)
DOC_BORDER_MARGIN = 0.03       # Margin (%) to consider touching border
DOC_BORDER_PENALTY_PER_SIDE = 0.12  # Score deduction per border-touching side (max ~4 sides)
DOC_CONVEXITY_WEIGHT = 0.10    # Soft scoring bonus for convex quads (not a hard filter)

APPROX_EPSILONS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.09, 0.12]

# ---------------------------------------------------------------------------
# TRANSFORM PARAMS
# ---------------------------------------------------------------------------
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 50
HOUGH_MIN_LEN = 40
HOUGH_MAX_GAP = 30
CORNER_SNAP_DIST = 60

# ---------------------------------------------------------------------------
# CURVATURE PARAMS
# ---------------------------------------------------------------------------
FLAT_CURVATURE_THRESH = 0.010
MODERATE_CURVATURE_THRESH = 0.080
N_BOUNDARY_COLS = 64
BOUNDARY_EDGE_THRESH_FRAC = 0.15
BOUNDARY_SMOOTH_KERNEL = 9

PAGE_MARGIN_X = 50
PAGE_MARGIN_Y = 20
REMAP_DECIMATE = 16
ADAPTIVE_WINSZ = 55
TEXT_MIN_WIDTH = 15
TEXT_MIN_HEIGHT = 2
TEXT_MIN_ASPECT = 1.5
TEXT_MAX_THICKNESS = 10

EDGE_MAX_OVERLAP = 1.0
EDGE_MAX_LENGTH = 100.0
EDGE_ANGLE_COST = 10.0
EDGE_MAX_ANGLE = 7.5

SPAN_MIN_WIDTH = 30
SPAN_PX_PER_STEP = 20
FOCAL_LENGTH = 1.2
MIN_SPANS = 3

# ---------------------------------------------------------------------------
# POSTPROCESS PARAMS
# ---------------------------------------------------------------------------
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
ADAPT_BLOCK_SIZE = 31
ADAPT_C = 10
WHITEN_KERNEL_SIZE = 61
SHARPEN_STRENGTH = 1.5
