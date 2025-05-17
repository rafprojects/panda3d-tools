#!/usr/bin/env python3
"""
Minimal SuperImage Demo
-----------------------
A minimalist demo of super-image following the GitHub example exactly.
This shows how simple the library's API can be for basic usage.
"""

import os
import sys
from PIL import Image
from super_image import EdsrModel, ImageLoader

def main():
    if len(sys.argv) < 2:
        print("Usage: python super_image_minimal.py <input_image> [scale(2,3,4)]")
        return
    
    # Get parameters
    input_path = sys.argv[1]
    scale = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    
    if scale not in [2, 3, 4]:
        print("Scale must be 2, 3, or 4")
        return
    
    # Generate output filename
    base_name, ext = os.path.splitext(input_path)
    output_path = f"{base_name}_upscaled_x{scale}{ext}"
    
    print(f"Upscaling {input_path} by {scale}x...")
    
    # Load the image
    image = Image.open(input_path)
    
    # Load model
    model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=scale)
    
    # Process image (exactly as in GitHub example)
    inputs = ImageLoader.load_image(image)
    preds = model(inputs)
    
    # Save result
    ImageLoader.save_image(preds, output_path)
    
    print(f"Upscaled image saved to: {output_path}")
    
    # Create comparison image
    compare_path = f"{base_name}_comparison_x{scale}{ext}"
    ImageLoader.save_compare(inputs, preds, compare_path)
    print(f"Comparison image saved to: {compare_path}")

if __name__ == "__main__":
    main()
