"""
preprocess.py — Input normalization, grayscale conversion, and blurring.
"""

import cv2
import numpy as np
import src.config as config
from src.utils import log

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Preprocess the input image by applying a bilateral filter and blur.
    
    Args:
        img: BGR image.
        
    Returns:
        Blurred grayscale image.
    """
    log("[preprocess] Applying bilateral filter and Gaussian blur.")
    filtered = cv2.bilateralFilter(
        img, 
        d=config.BILATERAL_D, 
        sigmaColor=config.BILATERAL_SIGMA_COLOR, 
        sigmaSpace=config.BILATERAL_SIGMA_SPACE
    )
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred
