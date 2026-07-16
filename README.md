# Document Scanner

A robust, modular, classical computer vision document scanning pipeline.
This project automatically detects, crops, perspective-transforms, dewarps (corrects curvature), and enhances photographs of documents and book pages. It uses no deep learning—only classical techniques via OpenCV, NumPy, and SciPy.

## Pipeline Architecture

The scanner is built on a clean, stateless pure-function pipeline (`src/pipeline.py`):

1. **Preprocess:** Bilateral filtering and Gaussian blurring for noise reduction.
2. **Edges:** Multi-scale Canny edge detection combined with Sobel-X/Y gradients.
3. **Contours:** Contour finding and evaluation using multi-epsilon quad simplification, scoring based on area, aspect ratio, and rectangularity.
4. **Transform:** Hough-line corner refinement and perspective warping.
5. **Curvature Correction:** Detects 3D page curvature. If curved, solves a cubic sheet model using text-span constraints to flatten the page.
6. **Postprocess:** Contrast enhancement (CLAHE), uneven lighting normalisation (background whitening), and optional adaptive thresholding.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the scanner via the CLI:

```bash
# Process a single image or a directory of images
python main.py --input inputs/ --output outputs/

# Enable debug terminal logs and save all intermediate images
python main.py --input photo.jpg --debug --save-steps

# Generate an OCR-ready binary image
python main.py --input photo.jpg --postprocess adaptive

# Skip curvature correction (perspective crop only)
python main.py --input photo.jpg --no-curvature
```

## Outputs
- **`outputs/`**: Final processed images.
- **`outputs/comparisons/`**: Side-by-side original vs. processed comparisons.
- **`outputs/debug/`**: Intermediate pipeline steps (if `--save-steps` is used).

## Acknowledgments
The cubic sheet curvature correction math and core logic in `src/curvature.py` are adapted from Matt Zucker's [page_dewarp](https://github.com/mzucker/page_dewarp) project.
