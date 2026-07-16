"""
curvature.py — Curved page boundary detection, estimation, and cubic sheet model.
"""

import cv2
import numpy as np
import scipy.optimize
import src.config as config
from src.utils import log, draw_curves_debug, save_debug

# ---------------------------------------------------------------------------
# Boundary detection
# ---------------------------------------------------------------------------
def _detect_top_bottom_boundaries(img: np.ndarray) -> tuple:
    """Robustly detect the top and bottom boundary curve of a document."""
    h, w = img.shape[:2]

    # Downscale for speed; we'll map results back to full-res coords
    scale = min(1.0, 800.0 / max(h, w))
    small_w = max(1, int(w * scale))
    small_h = max(1, int(h * scale))

    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 20, 60)

    n = config.N_BOUNDARY_COLS
    strip_w = max(1, small_w // n)

    top_raw = np.zeros(n, dtype=np.float32)
    bot_raw = np.full(n, float(small_h - 1), dtype=np.float32)

    for i in range(n):
        x0 = i * strip_w
        x1 = min(x0 + strip_w, small_w)
        col = edges[:, x0:x1]

        row_sums = col.sum(axis=1).astype(float)
        max_sum = row_sums.max()

        if max_sum == 0:
            top_raw[i] = 0
            bot_raw[i] = small_h - 1
            continue

        thresh = max_sum * config.BOUNDARY_EDGE_THRESH_FRAC
        active = np.where(row_sums >= thresh)[0]

        top_raw[i] = float(active[0])
        bot_raw[i] = float(active[-1])

    valid = (bot_raw - top_raw) > small_h * 0.1
    if valid.sum() >= 2:
        xs_all = np.arange(n, dtype=np.float32)
        xs_valid = xs_all[valid]
        top_interp = np.interp(xs_all, xs_valid, top_raw[valid])
        bot_interp = np.interp(xs_all, xs_valid, bot_raw[valid])
    else:
        top_interp = np.zeros(n, dtype=np.float32)
        bot_interp = np.full(n, float(small_h - 1), dtype=np.float32)

    kernel = config.BOUNDARY_SMOOTH_KERNEL
    top_smooth = cv2.GaussianBlur(top_interp.reshape(1, -1), (kernel, 1), 0).flatten()
    bot_smooth = cv2.GaussianBlur(bot_interp.reshape(1, -1), (kernel, 1), 0).flatten()

    top_full = top_smooth / scale
    bot_full = bot_smooth / scale

    return top_full, bot_full

def _estimate_curvature(top_curve: np.ndarray, bottom_curve: np.ndarray, img_height: int) -> float:
    """Estimate page curvature normalised by image height."""
    xs = np.linspace(0, 1, len(top_curve))

    def _residual_score(curve):
        coeffs = np.polyfit(xs, curve, 1)
        fitted = np.polyval(coeffs, xs)
        residuals = np.abs(curve - fitted)
        return float(residuals.mean()) / float(max(img_height, 1))

    return max(_residual_score(top_curve), _residual_score(bottom_curve))

def _select_strategy(curvature_score: float) -> str:
    """Choose dewarping strategy from curvature score."""
    if curvature_score < config.FLAT_CURVATURE_THRESH:
        return 'perspective'
    else:
        return 'cubic'

# ---------------------------------------------------------------------------
# Cubic Sheet Model
# ---------------------------------------------------------------------------

def _round_nearest_multiple(i, factor):
    i = int(i)
    rem = i % factor
    return i if not rem else i + factor - rem

def _pix2norm(shape, pts):
    h, w = shape[:2]
    scl = 2.0 / max(h, w)
    off = np.array([w, h], dtype=pts.dtype).reshape((-1, 1, 2)) * 0.5
    return (pts - off) * scl

def _norm2pix(shape, pts, as_integer):
    h, w = shape[:2]
    scl = max(h, w) * 0.5
    off = np.array([0.5 * w, 0.5 * h], dtype=pts.dtype).reshape((-1, 1, 2))
    rval = pts * scl + off
    return (rval + 0.5).astype(int) if as_integer else rval

def _box(w, h):
    return np.ones((h, w), dtype=np.uint8)

def _blob_mean_and_tangent(contour):
    m = cv2.moments(contour)
    area = m['m00']
    if area == 0:
        return None, None
    cx = m['m10'] / area
    cy = m['m01'] / area
    mm = np.array([[m['mu20'], m['mu11']], [m['mu11'], m['mu02']]]) / area
    _, svd_u, _ = cv2.SVDecomp(mm)
    return np.array([cx, cy]), svd_u[:, 0].flatten().copy()

class _ContourInfo:
    def __init__(self, contour, rect, mask):
        self.contour = contour
        self.rect = rect
        self.mask = mask

        ctr, tan = _blob_mean_and_tangent(contour)
        if ctr is None:
            raise ValueError("Degenerate contour")
        self.center = ctr
        self.tangent = tan
        self.angle = np.arctan2(tan[1], tan[0])

        clx = [self._px(p) for p in contour]
        lxmin, lxmax = min(clx), max(clx)
        self.local_xrng = (lxmin, lxmax)
        self.point0 = self.center + tan * lxmin
        self.point1 = self.center + tan * lxmax
        self.pred = None
        self.succ = None

    def _px(self, pt):
        return float(np.dot(self.tangent, pt.flatten() - self.center))

    def local_overlap(self, other):
        xmin = self._px(other.point0)
        xmax = self._px(other.point1)
        return min(self.local_xrng[1], xmax) - max(self.local_xrng[0], xmin)

def _angle_dist(b, a):
    d = b - a
    while d > np.pi: d -= 2 * np.pi
    while d < -np.pi: d += 2 * np.pi
    return abs(d)

def _get_page_extents(small):
    h, w = small.shape[:2]
    xmin, ymin = config.PAGE_MARGIN_X, config.PAGE_MARGIN_Y
    xmax, ymax = w - config.PAGE_MARGIN_X, h - config.PAGE_MARGIN_Y
    page = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(page, (xmin, ymin), (xmax, ymax), 255, -1)
    outline = np.array([[xmin, ymin], [xmin, ymax],
                        [xmax, ymax], [xmax, ymin]])
    return page, outline

def _get_text_mask(small, pagemask):
    sgray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small.copy()
    adapt = cv2.adaptiveThreshold(sgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                  config.ADAPTIVE_WINSZ, 25)
    try:
        mser = cv2.MSER_create(_delta=5, _min_area=20, _max_area=4000)
        regions, _ = mser.detectRegions(sgray)
        mser_mask = np.zeros_like(sgray)
        for pts in regions:
            hull = cv2.convexHull(pts.reshape(-1, 1, 2))
            cv2.fillPoly(mser_mask, [hull], 255)
        combined = cv2.bitwise_or(adapt, mser_mask)
    except Exception:
        combined = adapt

    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, _box(9, 3))
    combined = cv2.dilate(combined, _box(9, 1))
    combined = cv2.erode(combined, _box(1, 3))
    return np.minimum(combined, pagemask)

def _get_text_contours(small, pagemask):
    mask = _get_text_mask(small, pagemask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for cnt in contours:
        rect = cv2.boundingRect(cnt)
        xmin, ymin, bw, bh = rect
        if bw < config.TEXT_MIN_WIDTH or bh < config.TEXT_MIN_HEIGHT or bw < config.TEXT_MIN_ASPECT * bh:
            continue
        tight = np.zeros((bh, bw), dtype=np.uint8)
        tc = cnt - np.array((xmin, ymin)).reshape((-1, 1, 2))
        cv2.drawContours(tight, [tc], 0, 1, -1)
        if tight.sum(axis=0).max() > config.TEXT_MAX_THICKNESS:
            continue
        try:
            out.append(_ContourInfo(cnt, rect, tight))
        except ValueError:
            pass
    return out

def _candidate_edge(a, b):
    if a.point0[0] > b.point1[0]:
        a, b = b, a
    x_overlap = max(a.local_overlap(b), b.local_overlap(a))
    tangent = b.center - a.center
    oa = np.arctan2(tangent[1], tangent[0])
    da = max(_angle_dist(a.angle, oa), _angle_dist(b.angle, oa)) * 180 / np.pi
    dist = np.linalg.norm(b.point0 - a.point1)
    if dist > config.EDGE_MAX_LENGTH or x_overlap > config.EDGE_MAX_OVERLAP or da > config.EDGE_MAX_ANGLE:
        return None
    return (dist + da * config.EDGE_ANGLE_COST, a, b)

def _assemble_spans(cinfo_list):
    cinfo_list = sorted(cinfo_list, key=lambda c: c.rect[1])
    candidates = []
    for i, ci in enumerate(cinfo_list):
        for j in range(i):
            e = _candidate_edge(ci, cinfo_list[j])
            if e:
                candidates.append(e)
    candidates.sort()
    for _, a, b in candidates:
        if a.succ is None and b.pred is None:
            a.succ = b
            b.pred = a
    spans = []
    while cinfo_list:
        c = cinfo_list[0]
        while c.pred:
            c = c.pred
        span, width = [], 0.0
        while c:
            cinfo_list.remove(c)
            span.append(c)
            width += c.local_xrng[1] - c.local_xrng[0]
            c = c.succ
        if width > config.SPAN_MIN_WIDTH:
            spans.append(span)
    return spans

def _sample_spans(shape, spans):
    all_pts = []
    for span in spans:
        pts = []
        for ci in span:
            yv = np.arange(ci.mask.shape[0]).reshape((-1, 1))
            tot = (yv * ci.mask).sum(axis=0)
            means = tot / ci.mask.sum(axis=0)
            xmin, ymin = ci.rect[:2]
            step = config.SPAN_PX_PER_STEP
            start = int(((len(means) - 1) % step) / 2)
            pts += [(x + xmin, means[x] + ymin) for x in range(start, len(means), step)]
        pts_arr = np.array(pts, dtype=np.float32).reshape((-1, 1, 2))
        all_pts.append(_pix2norm(shape, pts_arr))
    return all_pts

def _keypoints_from_samples(shape, pagemask, page_outline, span_points):
    all_evecs, all_w = np.array([[0.0, 0.0]]), 0.0
    for pts in span_points:
        _, evec = cv2.PCACompute(pts.reshape(-1, 2), None, maxComponents=1)
        w = float(np.linalg.norm(pts[-1] - pts[0]))
        all_evecs += evec * w
        all_w += w
    evec = all_evecs / all_w
    x_dir = evec.flatten()
    if x_dir[0] < 0:
        x_dir = -x_dir
    y_dir = np.array([-x_dir[1], x_dir[0]])

    pc = cv2.convexHull(page_outline)
    pc = _pix2norm(pagemask.shape, pc.reshape((-1, 1, 2))).reshape(-1, 2)
    px = np.dot(pc, x_dir)
    py = np.dot(pc, y_dir)
    px0, px1, py0, py1 = px.min(), px.max(), py.min(), py.max()

    corners = np.vstack([
        px0 * x_dir + py0 * y_dir,
        px1 * x_dir + py0 * y_dir,
        px1 * x_dir + py1 * y_dir,
        px0 * x_dir + py1 * y_dir,
    ]).reshape((-1, 1, 2))

    ycoords, xcoords = [], []
    for pts in span_points:
        p2d = pts.reshape(-1, 2)
        ycoords.append(np.dot(p2d, y_dir).mean() - py0)
        xcoords.append(np.dot(p2d, x_dir) - px0)

    return corners, np.array(ycoords), xcoords

def _project_xy(xy_coords, pvec):
    RVEC_IDX = slice(0, 3)
    TVEC_IDX = slice(3, 6)
    CUBIC_IDX = slice(6, 8)
    K = np.array([[config.FOCAL_LENGTH, 0, 0], [0, config.FOCAL_LENGTH, 0], [0, 0, 1]], dtype=np.float32)

    alpha, beta = pvec[CUBIC_IDX]
    poly = np.array([alpha + beta, -2 * alpha - beta, alpha, 0])
    xy = xy_coords.reshape(-1, 2)
    z = np.polyval(poly, xy[:, 0])
    obj = np.hstack((xy, z.reshape(-1, 1)))
    pts, _ = cv2.projectPoints(obj, pvec[RVEC_IDX], pvec[TVEC_IDX], K, np.zeros(5))
    return pts

def _make_ki(span_counts):
    nspans = len(span_counts)
    npts = sum(span_counts)
    ki = np.zeros((npts + 1, 2), dtype=int)
    start = 1
    for i, count in enumerate(span_counts):
        ki[start:start + count, 1] = 8 + i
        start += count
    ki[1:, 0] = np.arange(npts) + 8 + nspans
    return ki

def _project_keypoints(pvec, ki):
    xy = pvec[ki]
    xy[0, :] = 0
    return _project_xy(xy, pvec)

def _default_params(corners, ycoords, xcoords):
    K = np.array([[config.FOCAL_LENGTH, 0, 0], [0, config.FOCAL_LENGTH, 0], [0, 0, 1]], dtype=np.float32)
    pw = float(np.linalg.norm(corners[1] - corners[0]))
    ph = float(np.linalg.norm(corners[-1] - corners[0]))
    obj3 = np.array([[0, 0, 0], [pw, 0, 0], [pw, ph, 0], [0, ph, 0]], dtype=np.float32)
    _, rvec, tvec = cv2.solvePnP(obj3, corners, K, np.zeros(5))
    counts = [len(xc) for xc in xcoords]
    params = np.hstack((rvec.flatten(), tvec.flatten(), [0.0, 0.0], ycoords.flatten()) + tuple(xcoords))
    return (pw, ph), counts, params

def _optimize(dstpoints, span_counts, params):
    ki = _make_ki(span_counts)
    def obj(pvec):
        return float(np.sum((_project_keypoints(pvec, ki) - dstpoints) ** 2))
    init_obj = obj(params)
    res = scipy.optimize.minimize(obj, params, method='Powell', options={'maxiter': 2000, 'ftol': 1e-6})
    if res.fun > init_obj or (init_obj > 1e-4 and res.fun > init_obj * 0.95):
        raise RuntimeError(f"Optimisation failed to converge (init={init_obj:.6f}, final={res.fun:.6f})")
    return res.x

def _remap(img, small, page_dims, params):
    height = 0.5 * page_dims[1] * img.shape[0]
    height = _round_nearest_multiple(height, config.REMAP_DECIMATE)
    width = _round_nearest_multiple(height * page_dims[0] / page_dims[1], config.REMAP_DECIMATE)
    hs, ws = height // config.REMAP_DECIMATE, width // config.REMAP_DECIMATE
    px_r = np.linspace(0, page_dims[0], ws)
    py_r = np.linspace(0, page_dims[1], hs)
    px, py = np.meshgrid(px_r, py_r)
    xy = np.hstack((px.flatten().reshape(-1, 1), py.flatten().reshape(-1, 1))).astype(np.float32)
    ipts = _project_xy(xy, params)
    ipts = _norm2pix(img.shape, ipts, False)
    ix = cv2.resize(ipts[:, 0, 0].reshape(px.shape), (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    iy = cv2.resize(ipts[:, 0, 1].reshape(py.shape), (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv2.remap(gray, ix, iy, cv2.INTER_CUBIC, None, cv2.BORDER_REPLICATE)

def _save_warp_grid(img: np.ndarray, params, small_shape):
    try:
        h, w = img.shape[:2]
        vis = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        grid_n = 20
        px_r = np.linspace(0, 1, grid_n)
        py_r = np.linspace(0, 1, grid_n)
        px, py = np.meshgrid(px_r, py_r)
        xy = np.hstack((px.flatten().reshape(-1, 1), py.flatten().reshape(-1, 1))).astype(np.float32)
        ipts = _project_xy(xy, params)
        ipts = _norm2pix(small_shape, ipts, True).reshape(grid_n, grid_n, 2)
        scale_x = w / float(small_shape[1])
        scale_y = h / float(small_shape[0])

        for r in range(grid_n):
            for c in range(grid_n - 1):
                p1 = (int(ipts[r, c, 0] * scale_x), int(ipts[r, c, 1] * scale_y))
                p2 = (int(ipts[r, c + 1, 0] * scale_x), int(ipts[r, c + 1, 1] * scale_y))
                cv2.line(vis, p1, p2, (0, 180, 255), 1, cv2.LINE_AA)
        for c in range(grid_n):
            for r in range(grid_n - 1):
                p1 = (int(ipts[r, c, 0] * scale_x), int(ipts[r, c, 1] * scale_y))
                p2 = (int(ipts[r + 1, c, 0] * scale_x), int(ipts[r + 1, c, 1] * scale_y))
                cv2.line(vis, p1, p2, (0, 180, 255), 1, cv2.LINE_AA)

        save_debug(vis, '07c_warp_grid')
    except Exception as e:
        log(f"[curvature] warp grid visualisation skipped: {e}")

def _cubic_sheet_dewarp(img: np.ndarray) -> np.ndarray:
    max_side = 1280
    scale = min(1.0, max_side / max(img.shape[:2]))
    small = cv2.resize(img, (0, 0), None, scale, scale, cv2.INTER_AREA) if scale < 1.0 else img.copy()
    pagemask, page_outline = _get_page_extents(small)
    cinfo_list = _get_text_contours(small, pagemask)
    spans = _assemble_spans(cinfo_list)

    if len(spans) < config.MIN_SPANS:
        mask2 = cv2.adaptiveThreshold(
            cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small,
            255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, config.ADAPTIVE_WINSZ, 7)
        mask2 = cv2.erode(mask2, _box(3, 1), iterations=3)
        mask2 = cv2.dilate(mask2, _box(8, 2))
        mask2 = np.minimum(mask2, pagemask)
        cnts2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cinfo_list2 = []
        for cnt in cnts2:
            rect = cv2.boundingRect(cnt)
            xm, ym, bw, bh = rect
            if bw < config.TEXT_MIN_WIDTH or bh < config.TEXT_MIN_HEIGHT or bw < config.TEXT_MIN_ASPECT * bh:
                continue
            tight = np.zeros((bh, bw), dtype=np.uint8)
            tc = cnt - np.array((xm, ym)).reshape((-1, 1, 2))
            cv2.drawContours(tight, [tc], 0, 1, -1)
            if tight.sum(axis=0).max() > config.TEXT_MAX_THICKNESS:
                continue
            try:
                cinfo_list2.append(_ContourInfo(cnt, rect, tight))
            except ValueError:
                pass
        spans2 = _assemble_spans(cinfo_list2)
        if len(spans2) > len(spans):
            spans = spans2

    if len(spans) < config.MIN_SPANS:
        raise RuntimeError(f"Only {len(spans)} text span(s) detected — need >= {config.MIN_SPANS}.")

    span_points = _sample_spans(small.shape, spans)
    corners, ycoords, xcoords = _keypoints_from_samples(small.shape, pagemask, page_outline, span_points)
    rough_dims, span_counts, params = _default_params(corners, ycoords, xcoords)
    dstpoints = np.vstack((corners[0].reshape((1, 1, 2),),) + tuple(span_points))
    params = _optimize(dstpoints, span_counts, params)

    if config.SAVE_STEPS:
        _save_warp_grid(img, params, small.shape)

    dst_br = corners[2].flatten()
    dims = np.array(rough_dims)
    def _obj_dims(d):
        return float(np.sum((dst_br - _project_xy(d, params).flatten()) ** 2))
    res_dims = scipy.optimize.minimize(_obj_dims, dims, method='Powell')
    page_dims = res_dims.x

    return _remap(img, small, page_dims, params)

def correct_curvature(img: np.ndarray) -> np.ndarray:
    """Analyze page curvature and apply cubic sheet dewarping if appropriate.

    Args:
        img: Cropped image from perspective transform.
    Returns:
        Dewarped image (grayscale if curvature was handled, else original input).
    """
    if not config.USE_CURVATURE:
        log("[curvature] Curvature correction disabled in config.")
        return img.copy()

    log("[curvature] Analyzing curvature.")
    top_curve, bottom_curve = _detect_top_bottom_boundaries(img)
    curvature_score = _estimate_curvature(top_curve, bottom_curve, img.shape[0])
    strategy = _select_strategy(curvature_score)

    log(f"[curvature] Score={curvature_score:.5f} -> Strategy='{strategy}'")

    if config.SAVE_STEPS:
        vis = draw_curves_debug(img, top_curve, bottom_curve, curvature_score)
        save_debug(vis, '07_curvature_viz')

    if strategy == 'perspective':
        log("[curvature] Flat page detected — using perspective crop only.")
        result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    else:
        try:
            result = _cubic_sheet_dewarp(img)
            log("[curvature] Cubic sheet model succeeded.")
        except RuntimeError as e:
            log(f"[curvature] Cubic model failed: {e}")
            log("[curvature] Falling back to perspective crop.")
            result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    if config.SAVE_STEPS:
        vis = result if len(result.shape) == 3 else cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        save_debug(vis, '08_dewarped')

    return result
