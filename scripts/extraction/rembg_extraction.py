"""
rembg extraction module for removing backgrounds from images using the rembg library.
rembg uses deep learning models to automatically remove backgrounds from images.
"""
import os
import numpy as np
import cv2
from PIL import Image
import glob

def remove_background_with_rembg(input_file, output_file=None, model_name="u2net", 
                                 alpha_matting=False, alpha_matting_foreground_threshold=240,
                                 alpha_matting_background_threshold=10, alpha_matting_erode_size=10,
                                 post_process_mask=True):
    """Extract object from a photo using the rembg library.
    
    Args:
        input_file (str): Path to the input image file
        output_file (str, optional): Path to save the output image
        model_name (str, optional): Model to use. Defaults to 'u2net'.
        alpha_matting (bool, optional): Whether to use alpha matting
        alpha_matting_foreground_threshold (int, optional): Alpha matting foreground threshold
        alpha_matting_background_threshold (int, optional): Alpha matting background threshold
        alpha_matting_erode_size (int, optional): Alpha matting erode size
        post_process_mask (bool, optional): Whether to post-process the mask
    
    Returns:
        PIL.Image or None: Processed image with transparent background, or None if processing failed
    """
    try:
        import rembg
        import numpy as np
        
        print(f"Processing image with rembg (model: {model_name})...")
        
        # Read the image
        input_img = Image.open(input_file)
        
        # Set up removal configuration
        session = rembg.new_session(model_name=model_name)
        
        # Process the image
        if alpha_matting:
            # Use alpha matting for improved edge quality
            result = rembg.remove(
                input_img, 
                session=session,
                alpha_matting=True,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold,
                alpha_matting_erode_size=alpha_matting_erode_size,
                post_process_mask=post_process_mask
            )
        else:
            # Standard removal
            result = rembg.remove(
                input_img, 
                session=session,
                post_process_mask=post_process_mask
            )
        
        # Save the result if output path is provided
        if output_file:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
            result.save(output_file)
            print(f"Result saved to {output_file}")
        else:
            # Create default output path
            base_name = os.path.splitext(input_file)[0]
            default_output = f"{base_name}_rembg.png"
            result.save(default_output)
            print(f"Result saved to {default_output}")
            output_file = default_output
        
        # Create a debug mask visualization
        result_array = np.array(result)
        if result_array.shape[2] == 4:  # Has alpha channel
            alpha_mask = result_array[:, :, 3]
            mask_img = Image.fromarray(alpha_mask)
            mask_debug_path = os.path.splitext(output_file)[0] + "_mask_debug.png"
            mask_img.save(mask_debug_path)
            print(f"Mask debug image saved to {mask_debug_path}")
        
        return result
    
    except ImportError as e:
        print(f"Error: Could not import rembg. Please install with 'pip install rembg': {e}")
        return None
    except Exception as e:
        print(f"Error processing image with rembg: {e}")
        return None

def remove_background_with_rembg_simple(input_file, output_file=None, model_name="u2net", 
                                        alpha_matting=False, alpha_matting_foreground_threshold=240,
                                        alpha_matting_background_threshold=10, alpha_matting_erode_size=10,
                                        process_ring_hole=True):
    """Extract object from a photo using the rembg library (simplified version).
    
    This is a simplified version that focuses on the core functionality without debug outputs.
    
    Args:
        input_file (str): Path to the input image file
        output_file (str, optional): Path to save the output image
        model_name (str, optional): Model to use. Defaults to 'u2net'.
        alpha_matting (bool, optional): Whether to use alpha matting
        alpha_matting_foreground_threshold (int, optional): Alpha matting foreground threshold
        alpha_matting_background_threshold (int, optional): Alpha matting background threshold
        alpha_matting_erode_size (int, optional): Alpha matting erode size
        process_ring_hole (bool, optional): Whether to apply additional processing to handle ring holes
    
    Returns:
        PIL.Image or None: Processed image with transparent background, or None if processing failed
    """
    try:
        import rembg
        
        print(f"Processing image with rembg (model: {model_name})...")
        
        # Read the original image
        orig_img = Image.open(input_file).convert("RGBA")
        
        # Set up removal configuration
        session = rembg.new_session(model_name=model_name)
        
        # Process the image
        if alpha_matting:
            # Use alpha matting for improved edge quality
            result = rembg.remove(
                orig_img, 
                session=session,
                alpha_matting=True,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold,
                alpha_matting_erode_size=alpha_matting_erode_size,
                post_process_mask=True
            )
        else:
            # Standard removal
            result = rembg.remove(
                orig_img, 
                session=session,
                post_process_mask=True
            )
        
        # Post-process for ring holes if requested
        if process_ring_hole:
            # Convert to numpy array to process
            result_np = np.array(result)
            
            # Extract alpha channel
            alpha = result_np[:, :, 3].copy()
            
            # Identify the ring area (where alpha > 0)
            ring_mask = alpha > 0
            
            # Find contours in the ring mask
            contours, _ = cv2.findContours(ring_mask.astype(np.uint8) * 255, 
                                          cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw all contours as filled masks to get the outer boundary
            outer_mask = np.zeros_like(ring_mask, dtype=np.uint8)
            cv2.drawContours(outer_mask, contours, -1, 1, thickness=cv2.FILLED)
            
            # For each contour, identify holes inside the ring
            for contour in contours:
                # Create a mask for this contour
                contour_mask = np.zeros_like(ring_mask, dtype=np.uint8)
                cv2.drawContours(contour_mask, [contour], 0, 1, thickness=cv2.FILLED)
                
                # Get bounding box of contour for efficiency
                x, y, w, h = cv2.boundingRect(contour)
                
                # Create a flood fill mask starting from the center of this contour
                flood_mask = np.zeros((ring_mask.shape[0] + 2, ring_mask.shape[1] + 2), dtype=np.uint8)
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Create a small region in the center for flood fill
                # This will help identify potential holes in rings
                test_radius = min(w, h) // 6  # Use a fraction of the object size
                center_region = np.zeros_like(ring_mask, dtype=np.uint8)
                cv2.circle(center_region, (center_x, center_y), test_radius, 1, thickness=cv2.FILLED)
                
                # If the center region has some transparent pixels from the rembg result,
                # it's likely there's a ring hole that was partially detected
                center_alpha = alpha[center_y-test_radius:center_y+test_radius, 
                                    center_x-test_radius:center_x+test_radius]
                if center_alpha.size > 0 and np.any(center_alpha == 0):
                    # There's already some transparency in the center - use it as a seed
                    # for the flood fill to capture the entire hole
                    mask_roi = ring_mask[y:y+h, x:x+w]
                    seed_points = []
                    
                    # Find transparent pixels in the center region to use as seeds
                    for cy in range(max(0, center_y-test_radius), min(ring_mask.shape[0], center_y+test_radius)):
                        for cx in range(max(0, center_x-test_radius), min(ring_mask.shape[1], center_x+test_radius)):
                            if alpha[cy, cx] == 0:
                                seed_points.append((cx, cy))
                    
                    # Use the first seed point for flood fill
                    if seed_points:
                        flood_x, flood_y = seed_points[0]
                        cv2.floodFill(contour_mask.copy(), flood_mask, (flood_x, flood_y), 0)
                        # The flood filled region in flood_mask is the hole
                        hole_mask = flood_mask[1:-1, 1:-1]
                        # Remove the hole from the alpha channel
                        alpha[hole_mask > 0] = 0
            
            # Update the result with the processed alpha channel
            result_np[:, :, 3] = alpha
            result = Image.fromarray(result_np)
        
        # Save the result if output path is provided
        if output_file:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
            result.save(output_file)
            print(f"Result saved to {output_file}")
        else:
            # Create default output path
            base_name = os.path.splitext(input_file)[0]
            default_output = f"{base_name}_rembg.png"
            result.save(default_output)
            print(f"Result saved to {default_output}")
            output_file = default_output
        
        return result
    
    except ImportError as e:
        print(f"Error: Could not import rembg. Please install with 'pip install rembg': {e}")
        return None
    except Exception as e:
        print(f"Error processing image with rembg: {e}")
        import traceback
        traceback.print_exc()
        return None

def batch_remove_background_with_rembg(input_dir, output_dir=None, model_name="birefnet-general", 
                                 keep_ring_hole=True, file_pattern="*.png"):
    """Process multiple images with rembg using birefnet-general model.
    
    This function processes all images in the input directory that match the file pattern,
    removing backgrounds using the rembg library with the specified model.
    
    Args:
        input_dir (str): Directory containing input images
        output_dir (str, optional): Directory to save processed images. If None, uses input_dir.
        model_name (str, optional): Model to use. Defaults to 'birefnet-general'.
        keep_ring_hole (bool, optional): If False, fills ring holes during processing. 
                                        If True, preserves any holes detected in rings.
        file_pattern (str, optional): Glob pattern to match input files. Defaults to "*.png".
    
    Returns:
        list: List of paths to processed output files
    """
    try:
        import rembg
        import glob
        import os
        
        print(f"Batch processing images with rembg (model: {model_name})...")
        
        # Ensure output directory exists
        if output_dir is None:
            output_dir = os.path.join(input_dir, "rembg_output")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all files matching the pattern
        input_files = glob.glob(os.path.join(input_dir, file_pattern))
        
        if not input_files:
            print(f"No files matching pattern '{file_pattern}' found in {input_dir}")
            return []
        
        print(f"Found {len(input_files)} files to process")
        
        # Process each file
        processed_files = []
        session = rembg.new_session(model_name=model_name)
        
        for i, input_file in enumerate(input_files):
            try:
                # Create output filename
                filename = os.path.basename(input_file)
                base_name = os.path.splitext(filename)[0]
                output_file = os.path.join(output_dir, f"{base_name}_rembg.png")
                
                print(f"Processing [{i+1}/{len(input_files)}]: {filename}")
                
                # Process with or without ring hole handling
                result = remove_background_with_rembg_simple(
                    input_file=input_file,
                    output_file=output_file,
                    model_name=model_name,
                    process_ring_hole=not keep_ring_hole  # Inverse the boolean since process_ring_hole fills holes
                )
                
                if result:
                    processed_files.append(output_file)
                    print(f"  - Saved: {output_file}")
                else:
                    print(f"  - Failed to process: {input_file}")
            
            except Exception as e:
                print(f"  - Error processing {input_file}: {e}")
        
        print(f"Batch processing complete. Processed {len(processed_files)}/{len(input_files)} images.")
        return processed_files
    
    except ImportError as e:
        print(f"Error: Could not import rembg. Please install with 'pip install rembg': {e}")
        return []
    except Exception as e:
        print(f"Error in batch processing: {e}")
        import traceback
        traceback.print_exc()
        return []

def batch_process_with_birefnet(input_dir, output_dir, process_ring_hole=True, 
                                alpha_matting=False, alpha_matting_foreground_threshold=240,
                                alpha_matting_background_threshold=10, alpha_matting_erode_size=10):
    """Process all images in a directory using rembg with birefnet-general model.
    
    Args:
        input_dir (str): Directory containing images to process
        output_dir (str): Directory to save processed images
        process_ring_hole (bool, optional): Whether to apply ring hole detection. Defaults to True.
        alpha_matting (bool, optional): Whether to use alpha matting. Defaults to False.
        alpha_matting_foreground_threshold (int, optional): Alpha matting foreground threshold. Defaults to 240.
        alpha_matting_background_threshold (int, optional): Alpha matting background threshold. Defaults to 10.
        alpha_matting_erode_size (int, optional): Alpha matting erode size. Defaults to 10.
    
    Returns:
        list: List of paths to processed images
    """
    try:
        import rembg
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files in the input directory
        image_files = glob.glob(os.path.join(input_dir, '*.png')) + \
                     glob.glob(os.path.join(input_dir, '*.jpg')) + \
                     glob.glob(os.path.join(input_dir, '*.jpeg'))
        
        if not image_files:
            print(f"No image files found in {input_dir}")
            return []
        
        print(f"Found {len(image_files)} images to process with birefnet-general model")
        
        # Set up removal configuration
        model_name = 'birefnet-general'  # Use birefnet-general model
        session = rembg.new_session(model_name=model_name)
        
        processed_files = []
        
        # Process each image
        for input_file in image_files:
            basename = os.path.basename(input_file)
            output_file = os.path.join(output_dir, f"{os.path.splitext(basename)[0]}_birefnet.png")
            
            print(f"Processing: {basename}")
            
            # Use the simple version of rembg processing
            result = remove_background_with_rembg_simple(
                input_file=input_file,
                output_file=output_file,
                model_name=model_name,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold,
                alpha_matting_erode_size=alpha_matting_erode_size,
                process_ring_hole=process_ring_hole
            )
            
            if result is not None:
                processed_files.append(output_file)
        
        print(f"Processed {len(processed_files)} images with birefnet-general model")
        return processed_files
    
    except ImportError as e:
        print(f"Error: Could not import rembg. Please install with 'pip install rembg': {e}")
        return []
    except Exception as e:
        print(f"Error during batch processing with rembg: {e}")
        import traceback
        traceback.print_exc()
        return []

def batch_process_cartier_with_birefnet(input_dir, 
                                        input_pattern,
                                        output_dir='assets/cartier/birefnet_batch', 
                                        process_ring_hole=True,
                                        model_name="birefnet-general"):
    """Process all images in the assets/cartier directory using rembg with birefnet-general model.
    
    Args:
        output_dir (str, optional): Directory to save processed images
        process_ring_hole (bool, optional): Whether to apply ring hole detection
        model_name (str, optional): Model name to use, defaults to birefnet-general
        
    Returns:
        list: List of tuples containing (input_file, output_file) for all processed images
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all PNG image
        input_files = glob.glob(f'{input_dir}/{input_pattern}')
        if not input_files:
            print("No image files found in assets/cartier/")
            return []
        
        results = []
        print(f"Found {len(input_files)} images to process with birefnet-general model...")
        
        # Process each image
        for input_file in input_files:
            basename = os.path.basename(input_file).replace(' ', '_')
            output_file = f'{output_dir}/{basename.split(".")[0]}_{"with_hole" if process_ring_hole else "no_hole"}.png'
            
            print(f"Processing: {basename}")
            
            # Process the image with rembg
            result = remove_background_with_rembg_simple(
                input_file=input_file,
                output_file=output_file,
                model_name=model_name,
                process_ring_hole=process_ring_hole
            )
            
            if result is not None:
                results.append((input_file, output_file))
                
        print(f"Batch processing complete. Processed {len(results)} of {len(input_files)} images.")
        return results
        
    except ImportError as e:
        print(f"Error: Could not import rembg. Please install with 'pip install rembg': {e}")
        return []
    except Exception as e:
        print(f"Error during batch processing: {e}")
        import traceback
        traceback.print_exc()
        return []
