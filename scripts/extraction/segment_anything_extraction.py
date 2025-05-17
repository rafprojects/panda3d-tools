"""
Segment Anything Model (SAM) extraction module for removing backgrounds from images using Meta's SAM.
SAM is a state-of-the-art segmentation model that can identify objects with minimal prompting.
"""
import os
import numpy as np
import cv2
from PIL import Image
import sys

def remove_background_with_segment_anything(input_file, output_file=None, 
                                           confidence_threshold=0.5,
                                           points_per_side=32,
                                           post_process=True):
    """Extract object from a photo using the Segment Anything Model (SAM) from Meta AI.
    
    This function provides an alternative to pixellib that works with modern Python environments.
    SAM is a state-of-the-art segmentation model that can identify objects with minimal prompting.
    
    Args:
        input_file (str): Path to the input image file
        output_file (str, optional): Path to save the output image
        confidence_threshold (float, optional): Threshold for confidence in segmentation
        points_per_side (int, optional): Number of points to sample (higher = more detail)
        post_process (bool, optional): Whether to smooth the edges of the mask
    
    Returns:
        PIL.Image or None: Processed image with transparent background, or None if processing failed
    """
    try:
        # Import required libraries
        import numpy as np
        import cv2
        import os
        import sys
        
        # Try to import segment anything
        try:
            import torch
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            print("Using Segment Anything Model (SAM) for background removal")
        except ImportError:
            print("Segment Anything Model not found. Installing required packages...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", 
                                  "git+https://github.com/facebookresearch/segment-anything.git"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
            import torch
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        
        # Check if model file exists, if not, download it
        model_path = "sam_vit_h_4b8939.pth"
        if not os.path.exists(model_path):
            print(f"Downloading SAM model... (this may take a few minutes)")
            import urllib.request
            model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            urllib.request.urlretrieve(model_url, model_path)
            print("Model downloaded successfully")
            
        # Load the image
        print(f"Processing image with Segment Anything: {input_file}")
        image = cv2.imread(input_file)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sam = sam_model_registry["vit_h"](checkpoint=model_path)
        sam.to(device=device)
        
        # Create mask generator
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=confidence_threshold,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2
        )
        
        # Generate masks
        print("Generating segmentation masks...")
        masks = mask_generator.generate(image_rgb)
        print(f"Found {len(masks)} potential object segments")
        
        if not masks:
            print("No objects detected in the image")
            return None
        
        # Sort masks by size (area) in descending order
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        
        # Get the largest mask that's not covering almost the entire image
        # (helps avoid selecting the background as the object)
        h, w = image.shape[:2]
        image_area = h * w
        
        # Find the main object mask (not too big, not too small)
        main_mask = None
        for mask in sorted_masks:
            # Skip masks that are too large (likely background) or too small (likely noise)
            if mask['area'] < 0.9 * image_area and mask['area'] > 0.01 * image_area:
                main_mask = mask
                break
        
        # If no suitable mask found, use the largest one
        if main_mask is None and sorted_masks:
            main_mask = sorted_masks[0]
        
        if main_mask is None:
            print("Failed to identify a suitable object mask")
            return None
        
        # Create a binary mask - SAM gives True for the object and False for background
        # Convert to 255 (white) for object, 0 (black) for background
        mask_array = main_mask['segmentation'].astype(np.uint8) * 255
        
        # Post-process the mask if requested
        if post_process:
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            mask_array = cv2.morphologyEx(mask_array, cv2.MORPH_CLOSE, kernel)
            mask_array = cv2.morphologyEx(mask_array, cv2.MORPH_OPEN, kernel)
            
            # Apply Gaussian blur for smoother edges
            mask_array = cv2.GaussianBlur(mask_array, (5, 5), 0)
            
            # Threshold again to get binary mask
            _, mask_array = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)
        
        # Save the original object mask for debugging
        original_mask_debug_path = os.path.splitext(output_file)[0] + "_original_mask_debug.png" if output_file else f"{os.path.splitext(input_file)[0]}_original_mask_debug.png"
        Image.fromarray(mask_array).save(original_mask_debug_path)
        print(f"Original object mask saved to {original_mask_debug_path}")
        
        # Create a transparent image
        result = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Copy RGB channels from original image
        result[:, :, 0:3] = image[:, :, ::-1]  # Convert BGR to RGB
        
        # IMPORTANT: For the alpha channel:
        # - 255 (white) means fully opaque
        # - 0 (black) means fully transparent
        # SAM has identified the ring with 255 (white) and background with 0 (black)
        # So we can directly use the mask for the alpha channel
        result[:, :, 3] = mask_array
        
        # Convert to PIL image
        pil_img = Image.fromarray(result)
        
        # Save the result if output path is provided
        if output_file:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
            pil_img.save(output_file)
            print(f"Result saved to {output_file}")
            
            # Save mask debug image
            mask_debug_path = os.path.splitext(output_file)[0] + "_mask_debug.png"
            Image.fromarray(mask_array).save(mask_debug_path)
            print(f"Mask debug image saved to {mask_debug_path}")
        else:
            # Create default output path
            base_name = os.path.splitext(input_file)[0]
            default_output = f"{base_name}_sam.png"
            pil_img.save(default_output)
            print(f"Result saved to {default_output}")
            output_file = default_output
            
            # Save mask debug image
            mask_debug_path = os.path.splitext(output_file)[0] + "_mask_debug.png"
            Image.fromarray(mask_array).save(mask_debug_path)
            print(f"Mask debug image saved to {mask_debug_path}")
        
        return pil_img
        
    except Exception as e:
        print(f"Error processing image with Segment Anything: {e}")
        import traceback
        traceback.print_exc()
        return None

def remove_background_with_segment_anything_simple(input_file, output_file=None, 
                                                 confidence_threshold=0.5,
                                                 points_per_side=32,
                                                 post_process=True):
    """Extract object from a photo using the Segment Anything Model (SAM) from Meta AI (simplified version).
    
    This is a simplified version that focuses on the core functionality without debug outputs.
    
    Args:
        input_file (str): Path to the input image file
        output_file (str, optional): Path to save the output image
        confidence_threshold (float, optional): Threshold for confidence in segmentation
        points_per_side (int, optional): Number of points to sample (higher = more detail)
        post_process (bool, optional): Whether to smooth the edges of the mask
    
    Returns:
        PIL.Image or None: Processed image with transparent background, or None if processing failed
    """
    try:
        # Import required libraries
        try:
            import torch
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            print("Using Segment Anything Model (SAM) for background removal")
        except ImportError:
            print("Segment Anything Model not found. Installing required packages...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", 
                                  "git+https://github.com/facebookresearch/segment-anything.git"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
            import torch
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        
        # Check if model file exists, if not, download it
        model_path = "sam_vit_h_4b8939.pth"
        if not os.path.exists(model_path):
            print(f"Downloading SAM model... (this may take a few minutes)")
            import urllib.request
            model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            urllib.request.urlretrieve(model_url, model_path)
            print("Model downloaded successfully")
            
        # Load the image
        print(f"Processing image with Segment Anything: {input_file}")
        image = cv2.imread(input_file)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sam = sam_model_registry["vit_h"](checkpoint=model_path)
        sam.to(device=device)
        
        # Create mask generator
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=confidence_threshold,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2
        )
        
        # Generate masks
        print("Generating segmentation masks...")
        masks = mask_generator.generate(image_rgb)
        print(f"Found {len(masks)} potential object segments")
        
        if not masks:
            print("No objects detected in the image")
            return None
        
        # Sort masks by size (area) in descending order
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        
        # Get the largest mask that's not covering almost the entire image
        # (helps avoid selecting the background as the object)
        h, w = image.shape[:2]
        image_area = h * w
        
        # Find the main object mask (not too big, not too small)
        main_mask = None
        for mask in sorted_masks:
            # Skip masks that are too large (likely background) or too small (likely noise)
            if mask['area'] < 0.9 * image_area and mask['area'] > 0.01 * image_area:
                main_mask = mask
                break
        
        # If no suitable mask found, use the largest one
        if main_mask is None and sorted_masks:
            main_mask = sorted_masks[0]
        
        if main_mask is None:
            print("Failed to identify a suitable object mask")
            return None
        
        # Create a binary mask
        mask_array = main_mask['segmentation'].astype(np.uint8) * 255
        
        # Post-process the mask if requested
        if post_process:
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            mask_array = cv2.morphologyEx(mask_array, cv2.MORPH_CLOSE, kernel)
            mask_array = cv2.morphologyEx(mask_array, cv2.MORPH_OPEN, kernel)
            
            # Apply Gaussian blur for smoother edges
            mask_array = cv2.GaussianBlur(mask_array, (5, 5), 0)
            
            # Threshold again to get binary mask
            _, mask_array = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)
        
        # Load the original image in PIL for transparency
        orig_img = Image.open(input_file).convert("RGBA")
        orig_np = np.array(orig_img)
        
        # Apply the mask to the alpha channel
        orig_np[..., 3] = mask_array
        
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
            default_output = f"{base_name}_sam.png"
            result_img.save(default_output)
            print(f"Result saved to {default_output}")
            output_file = default_output
        
        return result_img
        
    except Exception as e:
        print(f"Error processing image with Segment Anything: {e}")
        import traceback
        traceback.print_exc()
        return None