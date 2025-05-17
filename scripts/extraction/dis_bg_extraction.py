"""
Background removal module using dis-bg-remover, a modern AI-powered background removal tool.
This module uses the isnet-dis model for high-quality background removal.
"""
import os
import numpy as np
from PIL import Image

def remove_background_with_dis_simple(input_file, output_file=None, model_path=None):
    """Extract object from a photo using the dis_bg_remover library (simplified version).
    
    This is a simplified implementation that closely matches the expected usage of the dis_bg_remover library.
    
    Args:
        input_file (str): Path to the input image file
        output_file (str, optional): Path to save the output image
        model_path (str): Path to the isnet_dis.onnx model file
    
    Returns:
        PIL.Image or None: Processed image with transparent background, or None if processing failed
    """
    try:
        # Import the library
        from dis_bg_remover import remove_background
        
        print(f"Processing image with dis_bg_remover: {input_file}")
        
        # If model_path is not provided, check for the model file in known locations
        if not model_path:
            possible_models = [
                "isnet_dis.onnx",
                "isnet_bg_extraction",
                os.path.join("scripts", "extraction", "isnet_dis.onnx"),
                os.path.join("scripts", "extraction", "isnet_bg_extraction"),
                os.path.join(".", "isnet_dis.onnx"),
                os.path.join(".", "isnet_bg_extraction")
            ]
            
            for possible_path in possible_models:
                if os.path.exists(possible_path):
                    model_path = possible_path
                    print(f"Using model at: {model_path}")
                    break
                    
            if not model_path:
                print("Model file not found. Please provide the path to the model file.")
                return None
        
        # Process the image with dis_bg_remover
        extracted_img, mask = remove_background(model_path, input_file)
        
        # Prepare the result image from the original image and mask
        # Load the original image
        orig_img = Image.open(input_file).convert("RGBA")
        orig_np = np.array(orig_img)
        
        # Convert mask to uint8 if it's float32
        if mask.dtype == np.float32 or mask.max() <= 1.0:
            mask_uint8 = (mask * 255).clip(0, 255).astype(np.uint8)
        else:
            mask_uint8 = mask
            
        # Apply the mask to the original image's alpha channel
        orig_np[..., 3] = mask_uint8
        
        # Create the final image
        result_img = Image.fromarray(orig_np)
        
        # Save the result if output path is provided
        if output_file:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
            result_img.save(output_file)
            print(f"Result saved to {output_file}")
        else:
            # Create default output path
            base_name = os.path.splitext(input_file)[0]
            default_output = f"{base_name}_dis.png"
            result_img.save(default_output)
            print(f"Result saved to {default_output}")
        
        return result_img
        
    except ImportError as e:
        print(f"Error: Could not import dis_bg_remover. Please install with 'pip install dis-bg-remover': {e}")
        return None
    except Exception as e:
        print(f"Error processing image with dis_bg_remover: {e}")
        import traceback
        traceback.print_exc()
        return None