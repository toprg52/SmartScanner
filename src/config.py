"""
config.py — Centralized configuration parameters.
"""

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
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# ---------------------------------------------------------------------------
# EDGE PARAMS
# ---------------------------------------------------------------------------
CANNY_LO = 15
CANNY_HI = 60
CANNY_SCALES = [0.5, 1.0, 1.5]
SOBEL_THRESH = 35
CLOSE_KSIZE = 9
PAD_SIZE = 30

# ---------------------------------------------------------------------------
# CONTOUR PARAMS
# ---------------------------------------------------------------------------
DOC_MIN_AREA_RATIO = 0.05
DOC_PREFER_AREA_RATIO = 0.25
DOC_MIN_RECT_SCORE = 0.25
DOC_TARGET_ASPECT = 1.41
DOC_ASPECT_TOLERANCE = 0.80
APPROX_EPSILONS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.09, 0.12]

# ---------------------------------------------------------------------------
# TRANSFORM PARAMS
# ---------------------------------------------------------------------------
HOUGH_RHO = 1
import numpy as np
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
