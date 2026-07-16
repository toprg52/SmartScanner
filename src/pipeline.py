"""
pipeline.py — Pure function orchestrator for the document scanning pipeline.
"""

import cv2
import numpy as np
import src.config as config
from src.utils import log, save_debug, save_comparison
from src.preprocess import preprocess_image
from src.edges import detect_edges
from src.contours import find_document_contour
from src.transform import apply_perspective_transform, refine_corners
from src.curvature import correct_curvature
from src.postprocess import enhance_output

def run_pipeline(img: np.ndarray, name: str, post_method: str = 'clahe', do_whiten: bool = True, do_sharpen: bool = False) -> np.ndarray:
    """Run the pure, side-effect-free scanner pipeline on a single image.

    Args:
        img: BGR input image.
        name: Name of the image for debug saving.
        post_method: Post-processing method ('clahe' or 'adaptive').
        do_whiten: Whether to normalize background lighting.
        do_sharpen: Whether to apply unsharp masking.

    Returns:
        The final processed image.
    """
    log(f"\n[pipeline] Starting processing for {name} ({img.shape[1]}x{img.shape[0]})")

    # 1. Preprocess
    blurred = preprocess_image(img)

    # 2. Edges
    padded_edges, sobel_norm = detect_edges(blurred)
    save_debug(padded_edges, f"{name}_01_edges")

    # 3. Contours
    corners = find_document_contour(padded_edges, img.shape)
    
    if corners is None:
        log("[pipeline] Full-image fallback selected (no better contour found).")
        from src.contours import full_image_quad
        corners = full_image_quad(img.shape)

    # 4. Transform
    corners = refine_corners(corners, sobel_norm)
    cropped = apply_perspective_transform(img, corners)
    save_debug(cropped, f"{name}_02_cropped")

    # 5. Curvature Correction
    dewarped = correct_curvature(cropped)

    # 6. Postprocess
    final_img = enhance_output(dewarped, method=post_method, do_whiten=do_whiten, do_sharpen=do_sharpen)
    
    # Save final comparison
    save_comparison(img, final_img, name)
    log(f"[pipeline] Finished processing {name}")

    return final_img
