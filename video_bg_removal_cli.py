#!/usr/bin/env python3
"""
CLI for video background removal using rembg or backgroundremover.
"""
import argparse
import os
import sys

# Add scripts directory to path to import our video processing module
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

try:
    from video_background_removal import (
        remove_background_video_rembg,
        remove_background_video_backgroundremover
    )
except ImportError:
    print("Error: Could not import video_background_removal module.")
    print("Ensure 'scripts/video_background_removal.py' exists and is in the Python path.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Video background removal CLI."
    )
    parser.add_argument("-i", "--input", required=True, help="Input video file path.")
    parser.add_argument("-o", "--output", required=True, help="Output video file path.")

    subparsers = parser.add_subparsers(dest="method", required=True, help="Method to use for background removal.")

    # rembg subparser
    rembg_parser = subparsers.add_parser("rembg", help="Use rembg for video background removal.")
    rembg_parser.add_argument(
        "--model",
        default="u2net",
        help="Model to use with rembg (e.g., u2net, u2net_human_seg, isnet-general-use, etc.). Default: u2net."
    )
    rembg_parser.add_argument(
        "--max-workers", 
        type=int, 
        default=4,
        help="Maximum number of parallel workers for frame processing. Default: 4."
    )
    rembg_parser.add_argument(
        "--quality", 
        type=int, 
        default=95,
        help="JPEG quality for temporary frame storage (1-100). Default: 95."
    )
    rembg_parser.add_argument(
        "--alpha-matting", 
        action="store_true",
        help="Enable alpha matting."
    )
    rembg_parser.add_argument(
        "--alpha-matting-foreground-threshold",
        type=int,
        default=240,
        help="Alpha matting foreground threshold. Default: 240."
    )
    rembg_parser.add_argument(
        "--alpha-matting-background-threshold",
        type=int,
        default=10,
        help="Alpha matting background threshold. Default: 10."
    )
    rembg_parser.add_argument(
        "--alpha-matting-erode-size",
        type=int,
        default=10,
        help="Alpha matting erode size. Default: 10."
    )

    # backgroundremover subparser
    br_parser = subparsers.add_parser("backgroundremover", help="Use backgroundremover for video background removal.")
    br_parser.add_argument(
        "--model",
        default="u2net",
        choices=["u2net", "u2net_human_seg", "u2netp"],
        help="Model to use with backgroundremover. Default: u2net."
    )
    br_parser.add_argument(
        "--alpha-matting",
        action="store_true",
        help="Enable alpha matting for backgroundremover."
    )
    br_parser.add_argument(
        "--fg-threshold",
        type=int,
        default=240,
        help="Alpha matting foreground threshold for backgroundremover. Default: 240."
    )
    br_parser.add_argument(
        "--bg-threshold",
        type=int,
        default=10,
        help="Alpha matting background threshold for backgroundremover. Default: 10."
    )
    br_parser.add_argument(
        "--erode-size",
        type=int,
        default=10,
        help="Alpha matting erode structure size for backgroundremover. Default: 10."
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input video file not found: {args.input}")
        sys.exit(1)
        
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    if args.method == "rembg":
        # Build extra args list based on command line parameters
        extra_args = []
        if args.alpha_matting:
            extra_args.extend([
                "--alpha-matting",
                "--alpha-matting-foreground-threshold", str(args.alpha_matting_foreground_threshold),
                "--alpha-matting-background-threshold", str(args.alpha_matting_background_threshold),
                "--alpha-matting-erode-size", str(args.alpha_matting_erode_size)
            ])
            
        remove_background_video_rembg(
            args.input,
            args.output,
            model=args.model,
            extra_args=extra_args,
            max_workers=args.max_workers,
            quality=args.quality
        )
    elif args.method == "backgroundremover":
        remove_background_video_backgroundremover(
            args.input,
            args.output,
            model_name=args.model,
            alpha_matting=args.alpha_matting,
            alpha_matting_foreground_threshold=args.fg_threshold,
            alpha_matting_background_threshold=args.bg_threshold,
            alpha_matting_erode_size=args.erode_size
        )
    else:
        print(f"Error: Unknown method '{args.method}'")
        sys.exit(1)

if __name__ == "__main__":
    main()