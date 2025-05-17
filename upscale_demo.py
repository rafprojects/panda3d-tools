#!/usr/bin/env python3
"""
Sprite Upscaling Demo Script
This script demonstrates different upscaling methods for pixel art sprites.
"""
import cv2
import numpy as np
import os
import sys

def upscale_with_cv2(image_path, output_path, scale_factor=2, method=cv2.INTER_NEAREST):
    """Upscale an image using OpenCV.
    
    Args:
        image_path: Path to the source image
        output_path: Path to save the upscaled image
        scale_factor: Scaling factor (2, 3, 4, etc.)
        method: OpenCV interpolation method
    """
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # Get dimensions
    h, w = img.shape[:2]
    
    # Calculate new dimensions
    new_h, new_w = h * scale_factor, w * scale_factor
    
    # Resize the image
    upscaled = cv2.resize(img, (new_w, new_h), interpolation=method)
    
    # Save the resized image
    cv2.imwrite(output_path, upscaled)
    
    return upscaled.shape[:2]  # Return the new height and width

def pixel_art_upscale(image_path, output_path, scale_factor=2, edge_sharpness=1.0):
    """Upscale pixel art preserving sharp edges.
    
    Args:
        image_path: Path to the source image
        output_path: Path to save the upscaled image
        scale_factor: Scaling factor (2, 3, 4, etc.)
        edge_sharpness: Higher values preserve sharper pixel edges
    """
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # Get dimensions
    h, w = img.shape[:2]
    
    # Calculate new dimensions
    new_h, new_w = h * scale_factor, w * scale_factor
    
    # First do a nearest-neighbor upscale to preserve pixel edges
    upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # If edge sharpness is less than 1, blend with a slightly smoother version
    if edge_sharpness < 1.0:
        # Create a smoother version
        smooth = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Blend the two versions
        alpha = edge_sharpness  # Weight for pixelated version
        beta = 1.0 - alpha      # Weight for smooth version
        upscaled = cv2.addWeighted(upscaled, alpha, smooth, beta, 0)
    
    # If edge sharpness is greater than 1, enhance edges
    elif edge_sharpness > 1.0:
        # Apply sharpening filter
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]]) * (edge_sharpness - 1.0) / 8.0 + np.eye(3)
        
        # Apply the filter
        upscaled = cv2.filter2D(upscaled, -1, kernel)
    
    # Save the upscaled image
    cv2.imwrite(output_path, upscaled)
    
    return upscaled.shape[:2]  # Return the new height and width

def main():
    # Source sprite file
    sprite_file = 'assets/sprites/weapons/bullet.png'
    output_dir = 'upscaled_sprites'
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get original image size
    img = cv2.imread(sprite_file, cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]
    print(f"Original sprite size: {w}x{h}")
    
    # Save a copy of the original for comparison
    cv2.imwrite(os.path.join(output_dir, 'original_bullet.png'), img)
    
    # Demo different upscaling methods
    methods = [
        (cv2.INTER_NEAREST, 'nearest', 'Nearest Neighbor (pixelated)'),
        (cv2.INTER_LINEAR, 'bilinear', 'Bilinear (smoother edges)'),
        (cv2.INTER_CUBIC, 'bicubic', 'Bicubic (smoother gradients)'),
        (cv2.INTER_LANCZOS4, 'lanczos', 'Lanczos (sharp details)'),
    ]
    
    # Upscale with each method
    scale_factors = [2, 4]
    
    for scale in scale_factors:
        print(f"\nTesting {scale}x upscaling:")
        for method, name, description in methods:
            output_path = os.path.join(output_dir, f'bullet_{name}_{scale}x.png')
            new_h, new_w = upscale_with_cv2(sprite_file, output_path, scale, method)
            print(f"  - {name}: {description} -> {new_w}x{new_h}")
    
    # Try the specialized pixel art upscaler with different edge preservation settings
    print("\nTesting pixel art upscaler:")
    edge_settings = [0.5, 1.0, 1.5, 2.0]
    for scale in scale_factors:
        for edge in edge_settings:
            output_path = os.path.join(output_dir, f'bullet_pixel_art_edge{edge}_{scale}x.png')
            new_h, new_w = pixel_art_upscale(sprite_file, output_path, scale, edge)
            print(f"  - {scale}x with edge sharpness {edge} -> {new_w}x{new_h}")
    
    print(f"\nAll upscaled images saved to {os.path.abspath(output_dir)}")
    print("Use an image viewer to compare the results and see which works best for your needs.")

if __name__ == "__main__":
    main()