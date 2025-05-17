#!/usr/bin/env python3
"""
Advanced Sprite Upscaling Demo
This script demonstrates the XBR upscaling algorithm specifically designed for pixel art.
"""
import cv2
import numpy as np
import os
import sys
import glob
from scripts.spritetools import upscale_sprite_xbr

def main():
    # Source sprite directory and pattern
    sprite_dir = 'assets/sprites/weapons'
    output_dir = 'upscaled_sprites/xbr'
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PNG files in the sprite directory
    sprite_files = glob.glob(os.path.join(sprite_dir, '*.png'))
    
    if not sprite_files:
        print(f"No PNG files found in {sprite_dir}")
        return
    
    print(f"Found {len(sprite_files)} sprite files to upscale")
    
    # Try both 2x and 4x scaling with XBR algorithm
    scale_factors = [2, 4]
    
    for sprite_file in sprite_files:
        filename = os.path.basename(sprite_file)
        print(f"\nUpscaling {filename}:")
        
        # Get original image size
        img = cv2.imread(sprite_file, cv2.IMREAD_UNCHANGED)
        h, w = img.shape[:2]
        print(f"  Original size: {w}x{h}")
        
        # Save a copy of the original for comparison
        orig_output = os.path.join(output_dir, f"original_{filename}")
        cv2.imwrite(orig_output, img)
        
        # Upscale with XBR at different scales
        for scale in scale_factors:
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_xbr{scale}x.png")
            upscaled = upscale_sprite_xbr(sprite_file, output_path, scale)
            upscaled_h, upscaled_w = upscaled.shape[:2]
            print(f"  • {scale}x XBR upscaled: {upscaled_w}x{upscaled_h}")
    
    print(f"\nAll upscaled images saved to {os.path.abspath(output_dir)}")
    print("\nXBR is a specialized algorithm that preserves pixel art characteristics:")
    print("  • Keeps hard edges sharp")
    print("  • Smooths gradients")
    print("  • Handles diagonal lines intelligently")
    print("  • Preserves alpha channel/transparency")

if __name__ == "__main__":
    main()