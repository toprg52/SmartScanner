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
    log("[preprocess] Applying median blur, bilateral filter, and Gaussian blur.")
    
    # Median blur suppresses salt-and-pepper noise and background texture (e.g. wood grain)
    gray_initial = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray_initial, config.MEDIAN_BLUR_KSIZE)
    
    # Convert back to BGR for bilateral filter (or just filter grayscale)
    # Bilateral works fine on grayscale too.
    filtered = cv2.bilateralFilter(
        median, 
        d=config.BILATERAL_D, 
        sigmaColor=config.BILATERAL_SIGMA_COLOR, 
        sigmaSpace=config.BILATERAL_SIGMA_SPACE
    )
    blurred = cv2.GaussianBlur(filtered, (5, 5), 0)
    return blurred
