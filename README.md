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

## Project Structure

The codebase is split into modular, single-responsibility files for easy extensibility:
- **`main.py`**: CLI entry point and argument parsing.
- **`src/pipeline.py`**: The pure-function orchestrator that chains the modules together.
- **`src/config.py`**: Centralised configuration file containing all tunable magic numbers and thresholds.
- **`src/preprocess.py`**: Initial bilateral filtering and blurring.
- **`src/edges.py`**: Edge mapping via multi-scale Canny and Sobel derivatives.
- **`src/contours.py`**: Document boundary detection, multi-epsilon simplification, and quad scoring.
- **`src/transform.py`**: Perspective warping and Hough-line corner snapping.
- **`src/curvature.py`**: 3D curvature analysis and cubic sheet dewarping.
- **`src/postprocess.py`**: Final output enhancement (CLAHE, background whitening).
- **`src/utils.py`**: Shared logging, debug drawing, and file saving helpers.

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

## Tuning & Configuration
All internal parameters are exposed in `src/config.py`. If the scanner is failing on a specific type of document, you can adjust:
- **`FLAT_CURVATURE_THRESH`**: Controls how curved a page must be before the cubic solver kicks in (default `0.010`).
- **`DOC_MIN_AREA_RATIO`**: Minimum percentage of the image the document must cover (default `0.05`).
- **Edge Params**: `CANNY_LO`, `CANNY_HI`, and `SOBEL_THRESH` can be lowered for very low-contrast pages.

## Outputs
- **`outputs/`**: Final processed images.
- **`outputs/comparisons/`**: Side-by-side original vs. processed comparisons.
- **`outputs/debug/`**: Intermediate pipeline steps (if `--save-steps` is used).

## Acknowledgments
The cubic sheet curvature correction math and core logic in `src/curvature.py` are adapted from Matt Zucker's [page_dewarp](https://github.com/mzucker/page_dewarp) project.
