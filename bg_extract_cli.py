#!/usr/bin/env python3
"""
CLI for background extraction using various methods.
"""
import argparse
import os
from scripts.extraction import (
    remove_background_with_rembg_simple,
    remove_background_with_segment_anything_simple,
    remove_background_with_pixellib_simple,
    remove_background_with_dis_simple,
    extract_object_from_photo,
    enhance_extracted_object,
    remove_background_br
)

def main():
    parser = argparse.ArgumentParser(
        description="Background extraction CLI supporting multiple methods."
    )
    parser.add_argument("-i", "--input", required=True, help="Input image file")
    parser.add_argument("-o", "--output", help="Output image file")
    subparsers = parser.add_subparsers(dest="method", required=True, help="Extraction method to use")

    # rembg subparser
    rembg_parser = subparsers.add_parser("rembg", help="Use rembg for background removal")
    rembg_parser.add_argument("--model", help="Model name/path (if applicable)")
    rembg_parser.add_argument("--alpha-matting", action="store_true", help="Enable alpha matting (rembg)")
    rembg_parser.add_argument("--no-ring-hole", action="store_true", help="Do not process ring holes (rembg)")

    # segment_anything subparser
    sam_parser = subparsers.add_parser("segment_anything", help="Use Segment Anything for background removal")
    sam_parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Confidence threshold (segment_anything)")
    sam_parser.add_argument("--points-per-side", type=int, default=32, help="Points per side (segment_anything)")
    sam_parser.add_argument("--post-process", action="store_true", help="Enable post-processing (where supported)")

    # pixellib subparser
    pixellib_parser = subparsers.add_parser("pixellib", help="Use PixelLib for background removal")
    pixellib_parser.add_argument("--model", help="Model name/path (if applicable)")
    pixellib_parser.add_argument("--post-process", action="store_true", help="Enable post-processing (where supported)")

    # dis_bg subparser
    disbg_parser = subparsers.add_parser("dis_bg", help="Use DIS for background removal")
    disbg_parser.add_argument("--model", help="Model name/path (if applicable)")

    # backgroundremover subparser
    br_parser = subparsers.add_parser("backgroundremover", help="Use backgroundremover for background removal")
    br_parser.add_argument("--model", default="u2net", choices=["u2net", "u2net_human_seg", "u2netp"], help="Model to use (default: u2net)")
    br_parser.add_argument("--alpha-matting", action="store_true", default=True, help="Enable alpha matting (default: True)")
    br_parser.add_argument("--no-alpha-matting", action="store_false", dest="alpha_matting", help="Disable alpha matting")
    br_parser.add_argument("--fg-threshold", type=int, default=240, help="Alpha matting foreground threshold (default: 240)")
    br_parser.add_argument("--bg-threshold", type=int, default=10, help="Alpha matting background threshold (default: 10)")
    br_parser.add_argument("--erode-size", type=int, default=10, help="Alpha matting erode structure size (default: 10)")
    br_parser.add_argument("--base-size", type=int, default=1000, help="Alpha matting base size (default: 1000)")

    # custom subparser
    custom_parser = subparsers.add_parser("custom", help="Use custom method for background removal")
    custom_parser.add_argument("--shadow-threshold", type=int, default=20, help="Shadow threshold (custom)")
    custom_parser.add_argument("--color-distance-threshold", type=int, default=15, help="Color distance threshold (custom)")
    custom_parser.add_argument("--blur-size", type=int, default=5, help="Blur size (custom)")
    custom_parser.add_argument("--no-auto-bg", action="store_true", help="Disable auto background detection (custom)")
    custom_parser.add_argument("--output-format", default="png", help="Output format (custom)")
    custom_parser.add_argument("--no-preserve-colors", action="store_true", help="Do not preserve colors (custom)")
    custom_parser.add_argument("--no-metallic-fix", action="store_true", help="Disable metallic reflection fix (custom)")
    custom_parser.add_argument("--no-fill-holes", action="store_true", help="Disable hole filling (custom)")
    custom_parser.add_argument("--hole-size", type=int, default=5, help="Hole size (custom)")
    custom_parser.add_argument("--no-advanced-hole-filling", action="store_true", help="Disable advanced hole filling (custom)")
    custom_parser.add_argument("--debug-color", action="store_true", help="Enable debug color output (custom)")

    # Enhancement options (shared)
    parser.add_argument("--enhance", action="store_true", help="Apply enhancement after extraction")
    parser.add_argument("--enhance-brightness", type=float, default=1.0, help="Enhancement: brightness")
    parser.add_argument("--enhance-contrast", type=float, default=1.0, help="Enhancement: contrast")
    parser.add_argument("--enhance-saturation", type=float, default=1.0, help="Enhancement: saturation")
    parser.add_argument("--enhance-vibrance", type=float, default=0.0, help="Enhancement: vibrance")
    parser.add_argument("--enhance-sharpen", type=float, default=0.0, help="Enhancement: sharpen")
    parser.add_argument("--enhance-shadow-removal", type=float, default=0.5, help="Enhancement: shadow removal strength")

    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    method = args.method

    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return 1

    if not output_file:
        base, _ = os.path.splitext(input_file)
        output_file = f"{base}_{method}.png"

    # Extraction method dispatch
    if method == "rembg":
        result = remove_background_with_rembg_simple(
            input_file=input_file,
            output_file=output_file,
            model_name=args.model or "u2net",
            alpha_matting=args.alpha_matting,
            process_ring_hole=not args.no_ring_hole
        )
    elif method == "segment_anything":
        result = remove_background_with_segment_anything_simple(
            input_file=input_file,
            output_file=output_file,
            confidence_threshold=args.confidence_threshold,
            points_per_side=args.points_per_side,
            post_process=args.post_process
        )
    elif method == "pixellib":
        result = remove_background_with_pixellib_simple(
            input_file=input_file,
            output_file=output_file,
            model_type=args.model or "deeplabv3plus",
            post_process=args.post_process
        )
    elif method == "dis_bg":
        result = remove_background_with_dis_simple(
            input_file=input_file,
            output_file=output_file,
            model_path=args.model
        )
    elif method == "backgroundremover":
        result = remove_background_br(
            input_path=input_file,
            output_path=output_file,
            model_name=args.model,
            alpha_matting=args.alpha_matting,
            fg_threshold=args.fg_threshold,
            bg_threshold=args.bg_threshold,
            erode_size=args.erode_size,
            base_size=args.base_size
        )
    elif method == "custom":
        result, _ = extract_object_from_photo(
            input_file=input_file,
            output_file=output_file,
            shadow_threshold=args.shadow_threshold,
            color_distance_threshold=args.color_distance_threshold,
            blur_size=args.blur_size,
            auto_detect_background=not args.no_auto_bg,
            output_format=args.output_format,
            preserve_colors=not args.no_preserve_colors,
            metallic_reflection_fix=not args.no_metallic_fix,
            fill_holes=not args.no_fill_holes,
            hole_size=args.hole_size,
            advanced_hole_filling=not args.no_advanced_hole_filling,
            debug_color=args.debug_color
        )
    else:
        print(f"Unknown method: {method}")
        return 1

    if result is None:
        print("Extraction failed.")
        return 1

    print(f"Extraction complete. Output saved to: {output_file}")

    # Optional enhancement
    if args.enhance:
        enhanced_output = os.path.splitext(output_file)[0] + "_enhanced.png"
        enhance_extracted_object(
            input_file=output_file,
            output_file=enhanced_output,
            shadow_removal_strength=args.enhance_shadow_removal,
            brightness=args.enhance_brightness,
            contrast=args.enhance_contrast,
            saturation=args.enhance_saturation,
            vibrance=args.enhance_vibrance,
            sharpen=args.enhance_sharpen
        )
        print(f"Enhanced image saved to: {enhanced_output}")

if __name__ == "__main__":
    main()
