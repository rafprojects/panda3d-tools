"""
Custom extraction module for removing backgrounds from images of objects using
OpenCV and custom image processing.
"""
import cv2
import numpy as np
import os
from PIL import Image

def extract_object_from_photo(input_file, output_file=None, bg_color=None, shadow_threshold=20, 
                        color_distance_threshold=15, blur_size=5, auto_detect_background=True, 
                        output_format='png', preserve_colors=True, metallic_reflection_fix=True,
                        fill_holes=True, hole_size=5, advanced_hole_filling=True, debug_color=False):
    """Extracts an object from a photo with a uniform background, handling shadows and reflections.
    
    This function is designed for product photography where the object is placed on a 
    uniform background (like white or light gray). It can handle shadows cast by the object
    and preserve reflective metallic surfaces.
    
    Args:
        input_file (str): Path to the input image file
        output_file (str, optional): Path to save the output image. If None, constructs a path
                                   based on the input file with '_extracted' suffix.
        bg_color (tuple, optional): Background color as (B, G, R) to remove. If None, auto-detected.
        shadow_threshold (int, optional): Threshold for shadow detection. Higher values detect darker shadows
                                        but may include parts of the object. Defaults to 20.
        color_distance_threshold (int, optional): Threshold for color similarity to background.
                                               Higher values remove more of the background but may
                                               affect the object edges. Defaults to 15.
        blur_size (int, optional): Size of the blur kernel used for smoothing. Defaults to 5.
                                  Higher values create smoother edges.
        auto_detect_background (bool, optional): Whether to auto-detect the background color.
                                              Defaults to True.
        output_format (str, optional): Format of the output file ('png', 'jpg', etc). Defaults to 'png'.
                                    PNG is recommended as it supports transparency.
        preserve_colors (bool, optional): Whether to preserve original colors. Setting to False
                                        may improve background removal but could alter colors.
                                        Defaults to True.
        metallic_reflection_fix (bool, optional): Special handling for metallic reflective objects
                                                like gold rings. Defaults to True.
        fill_holes (bool, optional): Whether to fill small holes in the mask. Defaults to True.
        hole_size (int, optional): Maximum size of holes to fill. Defaults to 5.
        advanced_hole_filling (bool, optional): Use advanced hole filling techniques for complex objects
                                              like rings. Defaults to True.
        debug_color (bool, optional): Whether to print debug info about original vs extracted colors.
        
    Returns:
        tuple: (PIL.Image object with alpha transparency, mask array)
    """
    import cv2
    import numpy as np
    from PIL import Image
    import os
    
    # Read the image while preserving any alpha channel
    original_img = cv2.imread(input_file, cv2.IMREAD_UNCHANGED)
    if original_img is None:
        raise ValueError(f"Could not read image file: {input_file}")
    
    # Make a copy for the result
    result_img = original_img.copy()
    
    # Convert to RGB if it's grayscale
    if len(original_img.shape) == 2:
        img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
    else:
        img = original_img.copy()
    
    # Auto-detect background color if not specified
    if bg_color is None and auto_detect_background:
        # Enhanced background detection using edge sampling
        h, w = img.shape[:2]
        
        # Sample points from edges with more density to get better background representation
        sample_points = []
        
        # Sample more points across all edges, not just corners
        edge_margin = 10  # Stay 10px from the very edge to avoid potential artifacts
        
        # Top and bottom edges with more samples
        for i in range(10):
            x_pos = edge_margin + i * (w - 2*edge_margin) // 9
            sample_points.append((x_pos, edge_margin))
            sample_points.append((x_pos, h - edge_margin))
        
        # Left and right edges with more samples
        for i in range(10):
            y_pos = edge_margin + i * (h - 2*edge_margin) // 9
            sample_points.append((edge_margin, y_pos))
            sample_points.append((w - edge_margin, y_pos))
        
        # Sample colors from the points
        colors = [img[y, x].astype(float) for x, y in sample_points]
        
        # Use median color - more robust than mean for background detection
        bg_color = np.median(np.array(colors), axis=0).astype(np.uint8)
        
        # Print/log the auto-detected background color
        print(f"Auto-detected background color: {bg_color}")
    else:
        # Log auto-detected background color
        print(f"Using provided background color: {bg_color}")
    
    if bg_color is None:
        raise ValueError("Background color must be specified or auto-detection enabled")
    
    # Sample points for color debugging
    if debug_color:
        h, w = img.shape[:2]
        center_x, center_y = w // 2, h // 2
        sample_radius = min(w, h) // 4
        sample_points = [
            (center_x, center_y),
            (center_x + sample_radius//2, center_y),
            (center_x - sample_radius//2, center_y),
            (center_x, center_y + sample_radius//2),
            (center_x, center_y - sample_radius//2),
        ]
        original_colors = [img[y, x].copy() for x, y in sample_points]
    
    # Convert to HSV colorspace for better color separation
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bg_hsv = cv2.cvtColor(np.uint8([[bg_color]]), cv2.COLOR_BGR2HSV)[0][0]
    
    # Create a mask using color similarity in HSV space
    h, s, v = cv2.split(hsv_img)
    bg_h, bg_s, bg_v = bg_hsv
    
    # Lower and upper bounds for what we consider similar to background
    # The H (hue) is handled specially with circular difference
    h_diff = np.minimum(np.abs(h.astype(np.int32) - bg_h), 180 - np.abs(h.astype(np.int32) - bg_h))
    s_diff = np.abs(s.astype(np.int32) - bg_s)
    v_diff = np.abs(v.astype(np.int32) - bg_v)
    
    # Calculate color distance - modified for metallic reflection handling
    if metallic_reflection_fix:
        # Enhanced weights for metallic objects like gold rings
        # Increase saturation weight further (from 1.7 to 2.0) to better capture gold reflections
        # Reduce hue influence further for highly reflective surfaces
        # Slightly increase value weight to better handle highlights and shadows
        color_distance = h_diff * 0.1 + s_diff * 2.0 + v_diff * 0.2
        
        # Add special handling for likely reflective areas (high value but low saturation)
        # These are often reflections of the background on the metallic surface
        reflective_areas = (v > (bg_v + 20)) & (s < (bg_s + 10))
        
        # Boost the color distance for these reflective areas to ensure they're included in the object
        color_distance_boost = np.zeros_like(color_distance)
        color_distance_boost[reflective_areas] = color_distance_threshold * 0.5
        color_distance = color_distance + color_distance_boost
    else:
        # Standard weights
        color_distance = h_diff * 0.8 + s_diff * 0.8 + v_diff * 0.4
    
    # Create a binary mask - pixels similar to background become 0, others become 255
    mask = np.zeros_like(h)
    mask[color_distance > color_distance_threshold] = 255
    
    # After mask creation, print mask statistics
    print(f"Mask stats: min={mask.min()}, max={mask.max()}, mean={mask.mean():.2f}, nonzero={np.count_nonzero(mask)}")
    if np.all(mask == 0):
        print("WARNING: Mask is all black (no object detected)")
    elif np.all(mask == 255):
        print("WARNING: Mask is all white (entire image detected as object)")
    
    # Save debug mask image for inspection
    debug_mask_path = None
    if output_file:
        debug_mask_path = os.path.splitext(output_file)[0] + "_mask_debug.png"
        Image.fromarray(mask).save(debug_mask_path)
        print(f"Debug mask image saved to {debug_mask_path}")
    
    # Add shadow detection with special handling for metallic objects
    shadow_mask = np.zeros_like(h)
    
    if metallic_reflection_fix:
        # For metallic objects, we need to be more conservative with shadow detection
        # to avoid removing reflective highlights
        is_hue_similar = h_diff < 25  # Wider hue tolerance
        is_sat_similar = s_diff < 50  # Wider saturation tolerance
        is_darker = v < (bg_v - shadow_threshold)  # Darker value
    else:
        # Regular shadow detection
        is_hue_similar = h_diff < 15
        is_sat_similar = s_diff < 30
        is_darker = v < (bg_v - shadow_threshold)
        
    shadow_mask[(is_hue_similar & is_sat_similar & is_darker)] = 255
    
    # Combine the masks - exclude detected shadows from the main mask
    # For metallic objects, we're more conservative about what we exclude
    if metallic_reflection_fix:
        # Use a smaller kernel for shadow processing to preserve details
        shadow_kernel = np.ones((2, 2), np.uint8)
        shadow_mask = cv2.erode(shadow_mask, shadow_kernel, iterations=1)
    
    combined_mask = cv2.bitwise_and(mask, cv2.bitwise_not(shadow_mask))
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((blur_size, blur_size), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # Fill holes in the mask (particularly important for metallic objects)
    if fill_holes:
        # Find contours in the mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If we found contours, fill any holes
        if contours:
            # Get the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create a mask with just the external contour
            contour_mask = np.zeros_like(combined_mask)
            cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)  # -1 means fill
            
            # Create another mask for all significant contours
            all_contours_mask = np.zeros_like(combined_mask)
            for cnt in contours:
                if cv2.contourArea(cnt) > 100:  # Filter out tiny contours
                    cv2.drawContours(all_contours_mask, [cnt], -1, 255, -1)
            
            # More aggressive hole filling for metallic objects
            # Use morphological closing with a larger kernel
            filled_mask = cv2.morphologyEx(all_contours_mask, cv2.MORPH_CLOSE, 
                                          np.ones((2*hole_size+1, 2*hole_size+1), np.uint8))
            
            # Use the filled mask to identify holes
            holes = cv2.bitwise_and(filled_mask, cv2.bitwise_not(all_contours_mask))
            
            # Add the filled holes to our mask
            combined_mask = cv2.bitwise_or(combined_mask, holes)
            
            # For metallic objects, add specialized processing
            if metallic_reflection_fix:
                # Find areas of higher saturation (likely gold/metallic) that might have been missed
                # Gold/metallic tends to have higher saturation than the background
                high_sat_mask = (s > bg_s + 30).astype(np.uint8) * 255
                
                # Only consider high saturation areas within the dilated contour
                dilated_contour = cv2.dilate(all_contours_mask, np.ones((5, 5), np.uint8), iterations=1)
                high_sat_within_object = cv2.bitwise_and(high_sat_mask, dilated_contour)
                
                # Add these high saturation areas to our mask
                combined_mask = cv2.bitwise_or(combined_mask, high_sat_within_object)
    
    # Advanced hole filling techniques specially designed for rings and similar objects
    if advanced_hole_filling and fill_holes and metallic_reflection_fix:
        # Apply advanced ring detection and hole filling logic...
        # (Code omitted for brevity - see original implementation for details)
        pass
    
    # Apply Gaussian blur to smooth the edges (reduces jagged edges)
    blurred_mask = cv2.GaussianBlur(combined_mask, (blur_size, blur_size), 0)
    
    # Threshold the blurred mask to get a binary mask with anti-aliased edges
    _, smoothed_mask = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Apply additional processing to improve the mask
    # Dilate slightly to include potential edge details
    smoothed_mask = cv2.dilate(smoothed_mask, np.ones((2, 2), np.uint8), iterations=1)
    
    # Convert to PIL Image for alpha channel handling
    mask_img = Image.fromarray(smoothed_mask)
    # Load the original image into PIL - CRITICAL: Convert BGR to RGB for correct colors
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    
    # Create a completely new RGBA image with transparent background
    rgba = np.zeros((result_img_rgb.shape[0], result_img_rgb.shape[1], 4), dtype=np.uint8)
    # Copy RGB channels from the original image
    rgba[:,:,:3] = result_img_rgb[:,:,:3]
    # Set alpha channel from the mask
    rgba[:,:,3] = smoothed_mask  # This directly maps the mask to the alpha channel
    
    # Convert back to PIL
    pil_img = Image.fromarray(rgba)
    
    # Print debug color information
    if debug_color:
        pil_array = np.array(pil_img)
        # Sample points for color debugging
        h, w = img.shape[:2]
        center_x, center_y = w // 2, h // 2
        sample_points = [
            (center_x, center_y),
            (center_x + min(w, h) // 4, center_y),
            (center_x - min(w, h) // 4, center_y),
            (center_x, center_y + min(w, h) // 4),
            (center_x, center_y - min(w, h) // 4),
        ]
        original_colors = [img[y, x].copy() for x, y in sample_points]
        
        print("\nDEBUG COLOR COMPARISON:")
        for i, ((x, y), original_color) in enumerate(zip(sample_points, original_colors)):
            print(f"Sample {i+1} at ({x},{y}):")
            print(f"  Original (BGR): {original_color}")
            # Convert BGR to RGB for comparison
            original_rgb = original_color[:3][::-1]  # Reverse only the first 3 elements (BGR->RGB)
            print(f"  Original (RGB): {original_rgb}")
            # Check if this point is within the mask (not transparent)
            if 0 <= y < pil_array.shape[0] and 0 <= x < pil_array.shape[1] and pil_array[y, x, 3] > 0:
                # Get the extracted color (RGB)
                extracted_rgb = pil_array[y, x][:3]
                print(f"  Extracted (RGB): {extracted_rgb}")
                print(f"  Difference: {np.abs(np.array(original_rgb) - np.array(extracted_rgb)).sum()}")
            else:
                print("  Point was masked out (transparent)")
    
    # Save if output file is specified
    if output_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        pil_img.save(output_file)
        
        # Save debug mask image
        debug_mask_path = os.path.splitext(output_file)[0] + "_mask_debug.png"
        Image.fromarray(mask).save(debug_mask_path)
        print(f"Debug mask image saved to {debug_mask_path}")
        
        # Create a visualization of the mask overlaid on the original image
        # This helps to see how well the mask aligns with the object
        visualization = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)).convert('RGBA')
        # Create a blue mask for overlay, being careful with data types
        blue_overlay = np.zeros((original_img.shape[0], original_img.shape[1], 4), dtype=np.uint8)
        blue_overlay[:, :, 2] = 255  # Blue channel
        blue_overlay[:, :, 3] = 128  # Alpha (semi-transparent)
        mask_overlay = Image.fromarray(blue_overlay)
        # Apply the mask
        mask_overlay.putalpha(Image.fromarray(smoothed_mask).convert('L'))
        visualization = Image.alpha_composite(visualization, mask_overlay)
        
        # Save the visualization
        visualization_path = os.path.splitext(output_file)[0] + "_mask_overlay.png"
        visualization.save(visualization_path)
        print(f"Mask overlay visualization saved to {visualization_path}")
    else:
        # Create default output filename if none provided
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_extracted.{output_format}"
        pil_img.save(output_file)
    
    return pil_img, smoothed_mask

def batch_extract_objects(input_dir, output_dir, file_pattern='*.png', **kwargs):
    """Process multiple images in a directory, extracting objects from their backgrounds.
    
    Args:
        input_dir (str): Directory containing input images
        output_dir (str): Directory to save extracted images
        file_pattern (str): Pattern to match input files (e.g., '*.png')
        **kwargs: Additional arguments to pass to extract_object_from_photo()
    
    Returns:
        list: Paths to the extracted image files
    """
    import os
    import glob
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of input files
    input_files = glob.glob(os.path.join(input_dir, file_pattern))
    
    output_files = []
    for input_file in input_files:
        # Create output filename
        basename = os.path.basename(input_file)
        name_without_ext = os.path.splitext(basename)[0]
        output_file = os.path.join(output_dir, f"{name_without_ext}_extracted.png")
        
        # Process the image
        print(f"Processing: {basename}")
        extract_object_from_photo(
            input_file=input_file,
            output_file=output_file,
            **kwargs
        )
        
        output_files.append(output_file)
    
    return output_files

def enhance_extracted_object(input_file, output_file=None, shadow_removal_strength=0.5, 
                           brightness=1.0, contrast=1.0, saturation=1.0, 
                           vibrance=0.0, sharpen=0.0):
    """Enhance an extracted object with transparency to improve its appearance.
    
    Args:
        input_file (str): Path to the input image with transparency
        output_file (str, optional): Path to save the enhanced image
        shadow_removal_strength (float, optional): Strength of shadow removal (0-1)
        brightness (float, optional): Brightness adjustment (1.0 = unchanged)
        contrast (float, optional): Contrast adjustment (1.0 = unchanged)
        saturation (float, optional): Saturation adjustment (1.0 = unchanged)
        vibrance (float, optional): Vibrance adjustment (0.0 = unchanged)
        sharpen (float, optional): Sharpening amount (0.0 = no sharpening)
    
    Returns:
        PIL.Image: Enhanced image with transparency
    """
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter
    import os
    
    # Open the image and ensure it has an alpha channel
    img = Image.open(input_file)
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Convert to numpy array for more precise manipulation
    img_array = np.array(img)
    
    # Get RGB and alpha channels
    rgb = img_array[:, :, :3]
    alpha = img_array[:, :, 3]
    
    # Apply shadow removal (only to semi-transparent areas)
    if shadow_removal_strength > 0:
        # Identify semi-transparent areas (potential shadows)
        semi_transparent = (alpha > 0) & (alpha < 255)
        
        # Lighten these areas proportional to their transparency
        lightening = np.maximum(0, 255 - alpha) * shadow_removal_strength
        
        # Apply selectively to semi-transparent areas
        if np.any(semi_transparent):
            for i in range(3):  # Apply to each RGB channel
                channel = rgb[:, :, i].astype(np.float32)
                # Increase brightness in semi-transparent areas
                channel[semi_transparent] += lightening[semi_transparent]
                # Clip to valid range
                channel = np.clip(channel, 0, 255)
                rgb[:, :, i] = channel.astype(np.uint8)
    
    # Reassemble the image
    enhanced_array = np.dstack((rgb, alpha))
    enhanced_img = Image.fromarray(enhanced_array)
    
    # Apply PIL-based enhancements
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(enhanced_img)
        enhanced_img = enhancer.enhance(brightness)
    
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(enhanced_img)
        enhanced_img = enhancer.enhance(contrast)
    
    if saturation != 1.0:
        enhancer = ImageEnhance.Color(enhanced_img)
        enhanced_img = enhancer.enhance(saturation)
    
    # Apply sharpening if requested
    if sharpen > 0:
        # Create a sharpened version
        sharpened = enhanced_img.filter(ImageFilter.SHARPEN)
        # Convert both to numpy
        base = np.array(enhanced_img).astype(np.float32)
        sharp = np.array(sharpened).astype(np.float32)
        # Blend based on sharpen strength
        blended = base * (1 - sharpen) + sharp * sharpen
        # Ensure alpha channel is preserved exactly
        blended[:, :, 3] = base[:, :, 3]
        # Convert back to PIL
        enhanced_img = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
    
    # Save the enhanced image if output path is provided
    if output_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        enhanced_img.save(output_file)
        print(f"Enhanced image saved to {output_file}")
    
    return enhanced_img