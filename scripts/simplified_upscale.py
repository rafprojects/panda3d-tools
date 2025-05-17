"""
Simplified version of the image upscaling functions that more closely follows
the official super-image examples while preserving our added features.
"""

import cv2
import numpy as np
import torch
from PIL import Image

# Try to import super-image
try:
    import super_image
    HAS_SUPER_IMAGE = True
except ImportError:
    HAS_SUPER_IMAGE = False

def ai_upscale_superimage_simplified(image_path, output_path, scale=2, model_name='edsr-base', 
                                    denoise_level=0.0, sharpen_level=0.0):
    """
    Simplified AI upscaling using super-image package.
    This follows the GitHub examples more closely.
    
    Args:
        image_path: Path to input image
        output_path: Path to save upscaled image
        scale: Upscale factor (2, 3, 4)
        model_name: Model to use ('edsr-base', 'edsr', 'mdsr', 'a2n', 'han', 'real-world-sr')
        denoise_level: Amount of denoising to apply (0.0 to 1.0)
        sharpen_level: Amount of sharpening to apply after upscaling (0.0 to 1.0)
    """
    if not HAS_SUPER_IMAGE:
        raise ImportError("super-image package is not installed. Please install it with 'pip install super-image'")
    
    # Load image with PIL
    image = Image.open(image_path)
    
    # Handle alpha channel if present
    has_alpha = image.mode == 'RGBA'
    if has_alpha:
        alpha_channel = image.split()[3]
        image = image.convert('RGB')
    
    # Optional pre-processing: Denoise
    if denoise_level > 0.0:
        # Convert to OpenCV for denoising
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h_param = int(denoise_level * 15)  # Simplified parameter calculation
        denoised = cv2.fastNlMeansDenoisingColored(cv_img, None, h_param, h_param, 7, 21)
        image = Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
    
    # Select model based on model_name
    if model_name == 'edsr-base':
        model = super_image.EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=scale)
    elif model_name == 'edsr':
        model = super_image.EdsrModel.from_pretrained('eugenesiow/edsr', scale=scale)
    elif model_name == 'mdsr':
        model = super_image.MdsrModel.from_pretrained('eugenesiow/mdsr', scale=scale)
    elif model_name == 'a2n':
        model = super_image.A2nModel.from_pretrained('eugenesiow/a2n', scale=scale)
    elif model_name == 'han':
        model = super_image.HanModel.from_pretrained('eugenesiow/han', scale=scale)
    elif model_name == 'real-world-sr':
        model = super_image.EdsrModel.from_pretrained('eugenesiow/real-world-sr', scale=scale)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Use super_image's built-in ImageLoader
    inputs = super_image.ImageLoader.load_image(image)
    preds = model(inputs)
    output_img = super_image.ImageLoader.tensor_to_image(preds)
    
    # Optional post-processing: Sharpen
    if sharpen_level > 0.0:
        # Convert to OpenCV for sharpening
        img_array = np.array(output_img)
        bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Create and apply sharpening kernel
        k = sharpen_level * 0.5  # Simplified parameter
        kernel = np.array([[-k, -k, -k], [-k, 1 + 8*k, -k], [-k, -k, -k]])
        sharpened = cv2.filter2D(bgr, -1, kernel)
        
        output_img = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    
    # Handle alpha channel if present
    if has_alpha:
        alpha_upscaled = alpha_channel.resize(
            (output_img.width, output_img.height), 
            Image.BICUBIC
        )
        output_img.putalpha(alpha_upscaled)
    
    # Save the result
    output_img.save(output_path)
    
    return output_img.height, output_img.width


def ai_upscale_superimage_minimal(image_path, output_path, scale=2, model_name='edsr-base'):
    """
    Minimal version that exactly matches the GitHub example.
    No pre/post processing, just the core upscaling functionality.
    
    Args:
        image_path: Path to input image
        output_path: Path to save upscaled image
        scale: Upscale factor (2, 3, 4)
        model_name: Model to use ('edsr-base', 'edsr', 'mdsr', 'a2n', 'han', 'real-world-sr')
    """
    if not HAS_SUPER_IMAGE:
        raise ImportError("super-image package is not installed. Please install it with 'pip install super-image'")
    
    # Load the image
    image = Image.open(image_path)
    
    # Create model
    if model_name == 'edsr-base':
        model = super_image.EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=scale)
    elif model_name == 'edsr':
        model = super_image.EdsrModel.from_pretrained('eugenesiow/edsr', scale=scale)
    elif model_name == 'mdsr':
        model = super_image.MdsrModel.from_pretrained('eugenesiow/mdsr', scale=scale)
    else:
        # Default to edsr-base for other models
        model = super_image.EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=scale)
    
    # Process image with ImageLoader helper
    inputs = super_image.ImageLoader.load_image(image)
    preds = model(inputs)
    
    # Save the result
    super_image.ImageLoader.save_image(preds, output_path)
    
    # Return dimensions
    output_img = Image.open(output_path)
    return output_img.height, output_img.width
