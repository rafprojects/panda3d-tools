#!/usr/bin/env python3
"""
Simple SuperImage Example
-------------------------
This example demonstrates the basic usage of super-image package according to the official documentation.
"""

from super_image import EdsrModel, ImageLoader
from PIL import Image
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python super_image_simple.py <input_image> [output_image]")
        return
    
    input_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        # Generate output filename
        base_name, ext = os.path.splitext(input_path)
        output_path = f"{base_name}_super{ext}"
    
    # Load the image
    image = Image.open(input_path)
    
    # Create model - using edsr-base with 2x upscaling as default
    model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
    
    # Process image with ImageLoader helper
    inputs = ImageLoader.load_image(image)
    preds = model(inputs)
    
    # Save the result
    ImageLoader.save_image(preds, output_path)
    
    # Also create a comparison image
    compare_path = f"{os.path.splitext(output_path)[0]}_compare{os.path.splitext(output_path)[1]}"
    ImageLoader.save_compare(inputs, preds, compare_path)
    
    print(f"Upscaled image saved to: {output_path}")
    print(f"Comparison image saved to: {compare_path}")

if __name__ == "__main__":
    main()
