import argparse
import os
import sys
import cv2 # For INPAINT_TELEA and INPAINT_NS constants

# --- Robust sys.path modification ---
# Get the directory containing the currently running script (project root)
_CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the 'scripts' subdirectory
_SCRIPTS_DIR = os.path.join(_CURRENT_FILE_DIR, 'scripts')
# Add the 'scripts' directory to sys.path if it's not already there, prioritizing it
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
# --- End sys.path modification ---

try:
    from image_restoration import (
        denoise_image_opencv,
        denoise_tv_chambolle_skimage,
        denoise_nl_means_skimage,
        denoise_wavelet_skimage,
        inpaint_image_opencv
    )
    from image_vectorization import vectorize_image_vtracer # New import
except ImportError as e:
    print(f"Error importing from 'scripts/image_restoration.py' or 'scripts/image_vectorization.py': {e}")
    print(f"SCRIPT_DIR: {_CURRENT_FILE_DIR}, SCRIPTS_DIR: {_SCRIPTS_DIR}")
    print(f"sys.path: {sys.path}")
    print("Please ensure these files are in the 'scripts' directory relative to the CLI script, and all dependencies are installed.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Image processing CLI for restoration and vectorization.") # Updated description
    subparsers = parser.add_subparsers(dest="command", required=True, help="Processing command to execute")

    # --- Denoise OpenCV --- #
    denoise_cv_parser = subparsers.add_parser("denoise_opencv", help="Denoise image using OpenCV fastNlMeansDenoising")
    denoise_cv_parser.add_argument("-i", "--input", required=True, help="Input image path")
    denoise_cv_parser.add_argument("-o", "--output", required=True, help="Output image path")
    denoise_cv_parser.add_argument("--h-luminance", type=float, default=10.0, help="Filter strength for luminance component (default: 10.0)")
    denoise_cv_parser.add_argument("--h-color", type=float, default=10.0, help="Filter strength for color components (default: 10.0)")
    denoise_cv_parser.add_argument("--template-window-size", type=int, default=7, help="Template window size, must be odd (default: 7)")
    denoise_cv_parser.add_argument("--search-window-size", type=int, default=21, help="Search window size, must be odd (default: 21)")

    # --- Denoise TV Chambolle (skimage) --- #
    denoise_tv_parser = subparsers.add_parser("denoise_tv_chambolle", help="Denoise image using skimage TV Chambolle")
    denoise_tv_parser.add_argument("-i", "--input", required=True, help="Input image path")
    denoise_tv_parser.add_argument("-o", "--output", required=True, help="Output image path")
    denoise_tv_parser.add_argument("--weight", type=float, default=0.1, help="Denoising weight (default: 0.1)")
    denoise_tv_parser.add_argument("--multichannel", action=argparse.BooleanOptionalAction, default=True, help="Denoise each channel separately for color images")

    # --- Denoise NL-Means (skimage) --- #
    denoise_nlm_parser = subparsers.add_parser("denoise_nl_means", help="Denoise image using skimage Non-Local Means")
    denoise_nlm_parser.add_argument("-i", "--input", required=True, help="Input image path")
    denoise_nlm_parser.add_argument("-o", "--output", required=True, help="Output image path")
    denoise_nlm_parser.add_argument("--patch-size", type=int, default=7, help="Patch size for comparison (default: 7)")
    denoise_nlm_parser.add_argument("--patch-distance", type=int, default=11, help="Maximal distance to search for patches (default: 11)")
    denoise_nlm_parser.add_argument("--h-parameter", type=float, default=0.08, help="Cut-off distance for patch similarity (default: 0.08)")
    denoise_nlm_parser.add_argument("--multichannel", action=argparse.BooleanOptionalAction, default=True, help="Denoise each channel separately for color images")
    denoise_nlm_parser.add_argument("--fast-mode", action=argparse.BooleanOptionalAction, default=True, help="Use a faster version of the algorithm")

    # --- Denoise Wavelet (skimage) --- #
    denoise_wavelet_parser = subparsers.add_parser("denoise_wavelet", help="Denoise image using skimage Wavelet denoising")
    denoise_wavelet_parser.add_argument("-i", "--input", required=True, help="Input image path")
    denoise_wavelet_parser.add_argument("-o", "--output", required=True, help="Output image path")
    denoise_wavelet_parser.add_argument("--method", type=str, choices=['BayesShrink', 'VisuShrink'], default='BayesShrink', help="Thresholding method (default: BayesShrink)")
    denoise_wavelet_parser.add_argument("--mode", type=str, choices=['soft', 'hard'], default='soft', help="Thresholding mode (default: soft)")
    denoise_wavelet_parser.add_argument("--wavelet-levels", type=int, default=None, help="Number of wavelet decomposition levels (default: auto)")
    denoise_wavelet_parser.add_argument("--wavelet", type=str, default='db1', help="Type of wavelet to use (default: db1)")
    denoise_wavelet_parser.add_argument("--multichannel", action=argparse.BooleanOptionalAction, default=True, help="Denoise each channel separately for color images")
    denoise_wavelet_parser.add_argument("--rescale-sigma", action=argparse.BooleanOptionalAction, default=True, help="Rescale image intensities to have noise std dev of 1")

    # --- Inpaint OpenCV --- #
    inpaint_cv_parser = subparsers.add_parser("inpaint_opencv", help="Inpaint image using OpenCV")
    inpaint_cv_parser.add_argument("-i", "--input", required=True, help="Input image path")
    inpaint_cv_parser.add_argument("-m", "--mask", required=True, help="Mask image path (non-zero pixels are inpainted)")
    inpaint_cv_parser.add_argument("-o", "--output", required=True, help="Output image path")
    inpaint_cv_parser.add_argument("--radius", type=float, default=3.0, help="Radius of circular neighborhood for inpainting (default: 3.0)")
    inpaint_cv_parser.add_argument("--method-flag", type=str, choices=['telea', 'ns'], default='telea', help="Inpainting method: 'telea' or 'ns' (default: telea)")

    # --- Vectorize VTracer --- #
    vectorize_vtracer_parser = subparsers.add_parser("vectorize_vtracer", help="Vectorize a raster image to SVG using VTracer")
    vectorize_vtracer_parser.add_argument("-i", "--input", required=True, help="Input raster image path")
    vectorize_vtracer_parser.add_argument("-o", "--output", required=True, help="Output SVG file path")
    vectorize_vtracer_parser.add_argument("--mode", type=str, choices=["spline", "polygon", "none"], default='spline', help="Tracing mode: 'color' or 'binary' (default: color)")
    vectorize_vtracer_parser.add_argument("--colormode", type=str, choices=['color', 'binary'], default='color', help="VTracer colormode: 'color' or 'binary' (default: color)")
    vectorize_vtracer_parser.add_argument("--hierarchical", type=str, choices=['stacked', 'cutout'], default='stacked', help="Hierarchical mode: 'stacked' or 'cutout' (default: stacked)")
    vectorize_vtracer_parser.add_argument("--filter-speckle", type=int, default=4, help="Filter speckles smaller than this size (default: 4)")
    vectorize_vtracer_parser.add_argument("--color-precision", type=int, default=6, help="Number of bits for color quantization (default: 6)")
    vectorize_vtracer_parser.add_argument("--layer-difference", type=int, default=16, help="Layer difference (default: 16)")
    vectorize_vtracer_parser.add_argument("--corner-threshold", type=int, default=60, help="Corner detection threshold in degrees (default: 60)")
    vectorize_vtracer_parser.add_argument("--length-threshold", type=float, default=4.0, range=[3.5, 10.0], help="Minimum length of a curve segment (default: 4.0)")
    vectorize_vtracer_parser.add_argument("--max-iterations", type=int, default=10, help="Max iterations for tracing (default: 10)")
    vectorize_vtracer_parser.add_argument("--splice-threshold", type=int, default=45, help="Threshold for splicing segments in degrees (default: 45)")
    vectorize_vtracer_parser.add_argument("--path-precision", type=int, default=8, help="Number of decimal places for path coordinates (default: 8)")


    args = parser.parse_args()

    try:
        if args.command == "denoise_opencv":
            denoise_image_opencv(
                args.input, args.output,
                h_luminance=args.h_luminance,
                h_color=args.h_color,
                template_window_size=args.template_window_size,
                search_window_size=args.search_window_size
            )
        elif args.command == "denoise_tv_chambolle":
            denoise_tv_chambolle_skimage(
                args.input, args.output,
                weight=args.weight,
                multichannel=args.multichannel
            )
        elif args.command == "denoise_nl_means":
            denoise_nl_means_skimage(
                args.input, args.output,
                patch_size=args.patch_size,
                patch_distance=args.patch_distance,
                h_parameter=args.h_parameter,
                multichannel=args.multichannel,
                fast_mode=args.fast_mode
            )
        elif args.command == "denoise_wavelet":
            denoise_wavelet_skimage(
                args.input, args.output,
                method=args.method,
                mode=args.mode,
                wavelet_levels=args.wavelet_levels,
                wavelet=args.wavelet,
                multichannel=args.multichannel,
                rescale_sigma=args.rescale_sigma
            )
        elif args.command == "inpaint_opencv":
            method_flag_cv = cv2.INPAINT_TELEA if args.method_flag == 'telea' else cv2.INPAINT_NS
            inpaint_image_opencv(
                args.input, args.mask, args.output,
                radius=args.radius,
                method_flag=method_flag_cv
            )
        elif args.command == "vectorize_vtracer": # New command
            vectorize_image_vtracer(
                args.input, args.output,
                mode=args.mode,
                colormode=args.colormode,
                hierarchical=args.hierarchical,
                filter_speckle=args.filter_speckle,
                color_precision=args.color_precision,
                gradient_step=args.gradient_step,
                corner_threshold=args.corner_threshold,
                segment_length=args.segment_length,
                splice_threshold=args.splice_threshold,
                path_precision=args.path_precision
            )
        print(f"Operation '{args.command}' completed successfully.")
        print(f"Output saved to: {args.output}")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except IOError as e: # More specific than just Exception for IO issues
        print(f"IO Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e: # Catch-all for other unexpected errors
        print(f"An unexpected error occurred during command '{args.command}': {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
