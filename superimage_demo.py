#!/usr/bin/env python3
"""
SuperImage Upscaler Demo Script
-------------------------------
This script demonstrates the SuperImage upscaler with its new features:
1. Denoising before upscaling
2. Sharpening after upscaling
3. Support for real-world-sr model

Usage:
    python superimage_demo.py
"""

import os
import argparse
from scripts.image_upscale import ai_upscale_superimage

def main():
    parser = argparse.ArgumentParser(description="SuperImage Upscaler Demo")
    parser.add_argument('-i', '--input', default='scripts/character.png', help='Input image file')
    parser.add_argument('-o', '--output-dir', default='upscaled_sprites/superimage_demo', help='Output directory')
    parser.add_argument('-m', '--model', default='edsr-base', 
                       choices=['edsr-base', 'edsr', 'mdsr', 'a2n', 'han', 'real-world-sr'],
                       help='Model to use')
    parser.add_argument('-s', '--scale', type=int, default=2, help='Scale factor (2, 3, 4)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get input file name without extension
    input_basename = os.path.splitext(os.path.basename(args.input))[0]
    
    # Standard upscale
    output_path = os.path.join(args.output_dir, f"{input_basename}_{args.model}_standard_{args.scale}x.png")
    ai_upscale_superimage(args.input, output_path, scale=args.scale, model_name=args.model)
    print(f"Standard upscale saved to: {output_path}")
    
    # Upscale with denoising
    output_path = os.path.join(args.output_dir, f"{input_basename}_{args.model}_denoise0.3_{args.scale}x.png")
    ai_upscale_superimage(args.input, output_path, scale=args.scale, model_name=args.model, denoise_level=0.3)
    print(f"Denoised upscale saved to: {output_path}")
    
    # Upscale with sharpening
    output_path = os.path.join(args.output_dir, f"{input_basename}_{args.model}_sharpen0.5_{args.scale}x.png")
    ai_upscale_superimage(args.input, output_path, scale=args.scale, model_name=args.model, sharpen_level=0.5)
    print(f"Sharpened upscale saved to: {output_path}")
    
    # Upscale with both denoising and sharpening
    output_path = os.path.join(args.output_dir, f"{input_basename}_{args.model}_denoise0.3_sharpen0.5_{args.scale}x.png")
    ai_upscale_superimage(args.input, output_path, scale=args.scale, model_name=args.model, 
                         denoise_level=0.3, sharpen_level=0.5)
    print(f"Denoised and sharpened upscale saved to: {output_path}")
    
    print("\nDemo complete! All versions have been saved to the output directory.")
    print("You can compare them side by side to see the effects of each option.")

if __name__ == "__main__":
    main()
