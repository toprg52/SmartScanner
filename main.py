"""
main.py — CLI entry point for the document scanner.

Usage:
    python main.py --input <path> --output <dir> [OPTIONS]
"""

import os
import sys
import argparse
import cv2
import src.config as config
from src.pipeline import run_pipeline

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

def collect_images(input_path: str) -> list:
    """Return sorted list of image file paths from *input_path* (file or dir)."""
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        return sorted(
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        )
    raise FileNotFoundError(f"Input path not found: {input_path}")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='main.py',
        description='Robust non-DL document scanner: detect -> crop -> dewarp -> enhance',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument('--input', '-i', required=True, help='Path to an image file or directory of images.')
    p.add_argument('--output', '-o', default='outputs', help='Base output directory (default: outputs).')
    p.add_argument('--debug', '-d', action='store_true', help='Enable console debug logs.')
    p.add_argument('--save-steps', action='store_true', help='Save all intermediate debug images.')
    p.add_argument('--no-curvature', action='store_true', help='Disable curvature correction (dewarping).')
    p.add_argument('--postprocess', choices=['clahe', 'adaptive', 'none'], default='clahe', help='Post-processing method (default: clahe).')
    p.add_argument('--sharpen', action='store_true', help='Apply unsharp masking after postprocessing.')
    p.add_argument('--no-whiten', action='store_true', help='Disable background illumination normalisation.')
    return p

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Update global config flags based on CLI args
    config.DEBUG = args.debug
    config.SAVE_STEPS = args.save_steps
    config.USE_CURVATURE = not args.no_curvature

    try:
        images = collect_images(args.input)
    except FileNotFoundError as e:
        print(f"[main] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if not images:
        print(f"[main] No images found in: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"[main] Found {len(images)} image(s). Outputting to {args.output}")

    success = 0
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[main] ERROR: Cannot read image: {img_path}", file=sys.stderr)
            continue
            
        stem = os.path.splitext(os.path.basename(img_path))[0]
        
        final_img = run_pipeline(
            img=img,
            name=stem,
            post_method=args.postprocess,
            do_whiten=not args.no_whiten,
            do_sharpen=args.sharpen
        )
        # The pipeline handles saving the comparison image internally
        # if SAVE_COMPARISONS is enabled in config.
        success += 1

    print(f"\n[main] Done. {success}/{len(images)} image(s) processed successfully.")

if __name__ == '__main__':
    main()
