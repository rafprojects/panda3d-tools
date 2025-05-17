import argparse
import os
from image_upscale import upscale_with_cv2, pixel_art_upscale
import cv2
import sys

def main():
    parser = argparse.ArgumentParser(description="Image Upscaling CLI (supports regular and pixel art images)")
    parser.add_argument('-i', '--input', required=True, help='Input image file or directory')
    parser.add_argument('-o', '--output', required=True, help='Output image file or directory')
    parser.add_argument('-m', '--method', default='nearest', choices=['nearest', 'bilinear', 'bicubic', 'lanczos', 'pixel_art', 'realesrgan', 'superimage'], help='Upscaling method')
    parser.add_argument('-s', '--scale', type=int, default=2, help='Scale factor (2, 3, 4, etc.)')
    parser.add_argument('--edge-sharpness', type=float, default=1.0, help='Edge sharpness (for pixel_art method)')
    parser.add_argument('--batch', action='store_true', help='Batch mode: upscale all images in a directory')
    parser.add_argument('--clean-background', choices=['transparent', 'white', 'color'], help='Clean background before upscaling (choose replacement type)')
    parser.add_argument('--clean-bg-color', default='#FFFFFF', help='Hex color for background if --clean-background=color')
    parser.add_argument('--clean-bg-threshold', type=int, default=220, help='Threshold for background detection (default: 220)')
    parser.add_argument('--realesrgan-model', default='RealESRGAN_x4plus', help='Real-ESRGAN model name')
    parser.add_argument('--superimage-model', default='edsr-base', 
                      choices=['edsr-base', 'edsr', 'mdsr', 'a2n', 'han', 'real-world-sr'], 
                      help='Super-Image model name')
    parser.add_argument('--denoise', type=float, default=0.0, help='Denoise level (0.0-1.0) for superimage method')
    parser.add_argument('--sharpen', type=float, default=0.0, help='Sharpen level (0.0-1.0) for superimage method')
    args = parser.parse_args()

    if args.batch:
        # Batch mode: input/output must be directories
        if not os.path.isdir(args.input):
            print('Batch mode requires input to be a directory.')
            return 1
        os.makedirs(args.output, exist_ok=True)
        for fname in os.listdir(args.input):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                in_path = os.path.join(args.input, fname)
                out_path = os.path.join(args.output, f"{os.path.splitext(fname)[0]}_upscaled{os.path.splitext(fname)[1]}")
                pre_upscale_path = in_path
                if args.clean_background:
                    cleaned_path = os.path.join(args.output, f"{os.path.splitext(fname)[0]}_cleaned{os.path.splitext(fname)[1]}")
                    from image_upscale import clean_background
                    clean_background(in_path, cleaned_path, mode=args.clean_background, color_hex=args.clean_bg_color, threshold=args.clean_bg_threshold)
                    pre_upscale_path = cleaned_path
                run_upscale(pre_upscale_path, out_path, args)
        print(f"Batch upscaling complete. Output saved to: {args.output}")
    else:
        pre_upscale_path = args.input
        if args.clean_background:
            cleaned_path = os.path.splitext(args.output)[0] + '_cleaned' + os.path.splitext(args.output)[1]
            from image_upscale import clean_background
            clean_background(args.input, cleaned_path, mode=args.clean_background, color_hex=args.clean_bg_color, threshold=args.clean_bg_threshold)
            pre_upscale_path = cleaned_path
        run_upscale(pre_upscale_path, args.output, args)
        print(f"Upscaling complete. Output saved to: {args.output}")

def run_upscale(input_path, output_path, args):
    if args.method == 'pixel_art':
        pixel_art_upscale(input_path, output_path, scale_factor=args.scale, edge_sharpness=args.edge_sharpness)
    elif args.method == 'realesrgan':
        from image_upscale import ai_upscale_realesrgan
        ai_upscale_realesrgan(input_path, output_path, scale=args.scale, model_name=args.realesrgan_model)
    elif args.method == 'superimage':
        from image_upscale import ai_upscale_superimage
        ai_upscale_superimage(input_path, output_path, scale=args.scale, model_name=args.superimage_model, 
                            denoise_level=args.denoise, sharpen_level=args.sharpen)
    else:
        method_map = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }
        upscale_with_cv2(input_path, output_path, scale_factor=args.scale, method=method_map[args.method])

if __name__ == '__main__':
    main()
