#!/usr/bin/env python3
"""
Cartier Ring Extraction Demo

This script demonstrates how to extract objects (Cartier rings) from photos with 
uniform backgrounds using the enhanced image processing functions.
"""
import os
import sys
import glob
import shutil
import argparse
from pathlib import Path

# Add scripts directory to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

# Import the extraction modules from our organized package
from scripts.extraction import (
    extract_object_from_photo, 
    batch_extract_objects, 
    enhance_extracted_object,
    remove_background_with_rembg_simple,
    remove_background_with_pixellib_simple,
    remove_background_with_segment_anything_simple,
    remove_background_with_dis_simple,
    batch_remove_background_with_rembg,
    batch_process_cartier_with_birefnet
)

def clear_output_folders(folders, confirm=True):
    """Clear the content of specified output folders.
    
    Args:
        folders (list): List of folder paths to clear
        confirm (bool): Whether to ask for confirmation before clearing
        
    Returns:
        bool: True if folders were cleared, False otherwise
    """
    if not folders:
        return False
        
    if confirm:
        print("The following folders will be cleared:")
        for folder in folders:
            print(f"  - {folder}")
        confirmation = input("Are you sure you want to clear these folders? (y/n): ")
        if confirmation.lower() != 'y':
            print("Operation cancelled.")
            return False
    
    for folder in folders:
        if os.path.exists(folder):
            # Remove all files in the folder but keep the folder itself
            for file_path in glob.glob(os.path.join(folder, "*")):
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            print(f"Cleared folder: {folder}")
    
    return True

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Cartier Ring Extraction Demo')
    parser.add_argument('--clear-output', action='store_true', help='Clear output folders before processing')
    parser.add_argument('--no-confirm', action='store_true', help='Do not ask for confirmation when clearing folders')
    parser.add_argument('--demo', type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], default=0, 
                        help='Which demo to run (0=all, 1=single image, 2=batch, 3=parameter tuning, 4=old vs new, 5=rembg, 6=pixellib, 7=sam, 8=dis-bg-remover, 9=batch-rembg, 10=cartier-batch-birefnet)')
    args = parser.parse_args()
    
    # Source and destination directories
    input_dir = 'assets/cartier'
    output_dir = 'assets/cartier/extracted'
    enhanced_dir = 'assets/cartier/enhanced'
    rembg_dir = 'assets/cartier/rembg'
    pixellib_dir = 'assets/cartier/pixellib'
    sam_dir = 'assets/cartier/segment_anything'
    dis_dir = 'assets/cartier/dis_bg'
    
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(enhanced_dir, exist_ok=True)
    os.makedirs(rembg_dir, exist_ok=True)
    os.makedirs(pixellib_dir, exist_ok=True)
    os.makedirs(sam_dir, exist_ok=True)
    os.makedirs(dis_dir, exist_ok=True)
    
    # Clear output directories if requested
    if args.clear_output:
        folders_to_clear = [output_dir, enhanced_dir, rembg_dir, pixellib_dir, sam_dir, dis_dir]
        clear_output_folders(folders_to_clear, confirm=not args.no_confirm)
    
    print(f"Processing Cartier ring images from {input_dir}")
    
    # Run selected demo(s)
    if args.demo == 0 or args.demo == 1:
        # Option 1: Process a single image to demonstrate detailed options
        demo_single_image_extraction()
    
    if args.demo == 0 or args.demo == 2:
        # Option 2: Batch process all images
        demo_batch_extraction(input_dir, output_dir, enhanced_dir)
    
    if args.demo == 0 or args.demo == 3:
        # Option 3: Parameter tuning extraction tests
        demo_parameter_tuning_extraction()
    
    if args.demo == 0 or args.demo == 4:
        # Option 4: Compare old and new extraction logic
        demo_compare_old_new_extraction()
    
    if args.demo == 0 or args.demo == 5:
        # Option 5: Background removal with rembg
        demo_rembg_extraction()
        
    if args.demo == 0 or args.demo == 6:
        # Option 6: Background removal with pixellib
        demo_pixellib_extraction()
    
    if args.demo == 0 or args.demo == 7:
        # Option 7: Background removal with Segment Anything
        demo_segment_anything_extraction()
    
    if args.demo == 0 or args.demo == 8:
        # Option 8: Background removal with dis-bg-remover
        demo_dis_extraction()
    
    if args.demo == 0 or args.demo == 9:
        # Option 9: Batch background removal with rembg
        demo_batch_rembg_extraction()
        
    if args.demo == 0 or args.demo == 10:
        # Option 10: Batch background removal with rembg optimized for Cartier rings
        demo_cartier_batch_birefnet()

def demo_single_image_extraction():
    """Demonstrate detailed extraction options with a single image"""
    # Get the first image file from the directory
    image_files = glob.glob('assets/cartier/Screenshot*.png')
    if not image_files:
        print("No image files found in assets/cartier/")
        return
    
    # Use the first image file found
    input_file = image_files[0]
    basename = os.path.basename(input_file).replace(' ', '_')
    output_file = f'assets/cartier/extracted/{basename.split(".")[0]}_extracted.png'
    
    print("\n\nDEMO 1: Single Image Extraction")
    print(f"Processing image: {input_file}")
    
    # Extract the ring with improved parameters based on recommendations
    extract_object_from_photo(
        input_file=input_file,
        output_file=output_file,
        # Use recommended parameters for reflective gold rings
        shadow_threshold=20,          # Balance between shadow detection and preserving ring details
        color_distance_threshold=30,  # Higher threshold to include reflective areas
        blur_size=5,                  # Good balance for edge smoothing
        metallic_reflection_fix=True, # Enable enhanced metallic reflection handling
        hole_size=5,                  # Appropriate for preserving ring details
        advanced_hole_filling=True,   # Enable advanced hole filling for rings
        debug_color=True              # Enable color debugging
    )
    
    # Enhanced version with shadow removal
    enhanced_basename = os.path.basename(output_file).replace('.png', '_enhanced.png')
    enhanced_output = f'assets/cartier/enhanced/{enhanced_basename}'
    
    print(f"Enhancing extracted image")
    enhance_extracted_object(
        input_file=output_file,
        output_file=enhanced_output,
        shadow_removal_strength=0.4,  # Reduced from 0.6 to better preserve metallic reflections
        brightness=1.05,              # Slight brightness increase
        contrast=1.1,                 # Moderate contrast increase for ring details
        sharpen=0.3                   # Reduced from 0.4 to avoid artifacts on metallic edges
    )
    
    print("Single image processing complete.\n")
    print(f"- Extracted: {output_file}")
    print(f"- Enhanced:  {enhanced_output}")
    print(f"- Mask overlay: {os.path.splitext(output_file)[0]}_mask_overlay.png")

def demo_batch_extraction(input_dir, output_dir, enhanced_dir):
    """Demonstrate batch processing of all ring images"""
    print("\n\nDEMO 2: Batch Processing All Images")
    
    # Get all image files directly instead of using a pattern with glob
    # This helps with spaces in filenames
    image_files = glob.glob(os.path.join(input_dir, "Screenshot*.png"))
    print(f"Found {len(image_files)} images to process")
    
    if not image_files:
        print("No matching images found!")
        return
    
    # Process each image individually for better handling of filenames with spaces
    output_files = []
    for img_path in image_files:
        # Create more filename-friendly output path by preserving unique parts of the filename
        # Extract the time portion (e.g., 1.15.34) to create unique filenames
        basename = os.path.basename(img_path)
        # Extract time part from filename (e.g., extract "1.15.34" from "Screenshot 2025-05-09 at 1.15.34 PM.png")
        time_part = basename.split("at ")[1].split(" PM")[0].replace(".", "_")
        output_path = os.path.join(output_dir, f"ring_{time_part}_png_extracted.png")
        output_files.append(output_path)
        
        # Process the image
        print(f"Processing: {os.path.basename(img_path)}")
        extract_object_from_photo(
            input_file=img_path,
            output_file=output_path,
            shadow_threshold=15,               # Reduced to be less aggressive with shadows
            color_distance_threshold=18,       # Slightly increased to better detect gold rings
            blur_size=3,                       # Smaller blur for sharper edges
            metallic_reflection_fix=True,      # Enable special handling for metallic reflective surfaces
            fill_holes=True,                   # Fill small holes in the extraction
            hole_size=10,                      # Size of holes to fill (adjusted for ring details)
            advanced_hole_filling=True,        # Enable the new advanced hole filling algorithm for rings
            debug_color=True                   # Enable color debugging to verify color preservation
        )
    
    print(f"Extracted {len(output_files)} images")
    
    # Enhance the extracted objects with various parameters
    print("Enhancing extracted images...")
    enhanced_files = []
    
    for i, file_path in enumerate(output_files):
        basename = os.path.basename(file_path)
        base_name = os.path.splitext(basename)[0]
        
        # Output path for enhanced image
        enhanced_path = os.path.join(enhanced_dir, f"{base_name}_enhanced.png")
        enhanced_files.append(enhanced_path)
        
        # Use slightly different enhancement parameters for demonstration
        shadow_strength = 0.3 + (i % 3) * 0.1  # Reduce shadow removal strength (0.3-0.5 range)
        brightness = 1.0 + (i % 5) * 0.02      # Vary between 1.0, 1.02, 1.04, 1.06, 1.08
        contrast = 1.0 + (i % 4) * 0.05        # Vary between 1.0, 1.05, 1.1, 1.15
        
        # Enhance the extracted image
        enhance_extracted_object(
            input_file=file_path,
            output_file=enhanced_path,
            shadow_removal_strength=shadow_strength,
            brightness=brightness,
            contrast=contrast,
            sharpen=0.15                        # Reduced sharpening to preserve metallic appearance
        )
    
    print("\nBatch processing complete!")
    print(f"- Original images:  {len(image_files)}")
    print(f"- Extracted images: {len(output_files)}")
    print(f"- Enhanced images:  {len(enhanced_files)}")
    print("\nOutputs saved to:")
    print(f"- Extracted: {output_dir}")
    print(f"- Enhanced: {enhanced_dir}")
    print("\nYou can now use these extracted images with transparency in your projects!")

def demo_parameter_tuning_extraction():
    """Test a reduced set of parameter combinations and save results in organized folders."""
    print("\n\nDEMO 3: Parameter Tuning Extraction Tests")
    input_dir = 'assets/cartier'
    base_output_dir = 'assets/cartier/extracted'
    base_enhanced_dir = 'assets/cartier/enhanced'
    os.makedirs(base_output_dir, exist_ok=True)
    os.makedirs(base_enhanced_dir, exist_ok=True)

    # Reduced parameter grids - just testing cdt15 and cdt20
    color_distance_thresholds = [15, 20]
    shadow_thresholds = [10, 20]
    blur_sizes = [3, 5]

    # Use the first image for tuning
    image_files = glob.glob(os.path.join(input_dir, 'Screenshot*.png'))
    if not image_files:
        print("No image files found for parameter tuning.")
        return
    input_file = image_files[0]
    basename = os.path.basename(input_file).replace(' ', '_').split('.')[0]

    for cdt in color_distance_thresholds:
        for st in shadow_thresholds:
            for blur in blur_sizes:
                test_dir = os.path.join(base_output_dir, f'test_cdt{cdt}_st{st}_blur{blur}')
                os.makedirs(test_dir, exist_ok=True)
                out_file = os.path.join(test_dir, f'{basename}_extracted.png')
                print(f"Extracting with color_distance_threshold={cdt}, shadow_threshold={st}, blur_size={blur}")
                extract_object_from_photo(
                    input_file=input_file,
                    output_file=out_file,
                    shadow_threshold=st,
                    color_distance_threshold=cdt,
                    blur_size=blur,
                    debug_color=True
                )
                # Enhance and save in the same test dir
                enhanced_file = os.path.join(test_dir, f'{basename}_enhanced.png')
                enhance_extracted_object(
                    input_file=out_file,
                    output_file=enhanced_file,
                    shadow_removal_strength=0.6,
                    brightness=1.05,
                    contrast=1.1,
                    sharpen=0.4
                )
                print(f"  Saved extracted: {out_file}")
                print(f"  Saved enhanced:  {enhanced_file}")
    print("\nParameter tuning extraction complete. Check the extracted/ subfolders for results.")

def demo_compare_old_new_extraction():
    """Compare old and new extraction logic with a broad sweep of parameters."""
    print("\n\nDEMO 4: Old vs New Extraction Comparison")
    input_dir = 'assets/cartier'
    base_output_dir = 'assets/cartier/extracted/compare_old_new'
    os.makedirs(base_output_dir, exist_ok=True)

    # Use the first image for comparison
    image_files = glob.glob(os.path.join(input_dir, 'Screenshot*.png'))
    if not image_files:
        print("No image files found for comparison.")
        return
    input_file = image_files[0]
    basename = os.path.basename(input_file).replace(' ', '_').split('.')[0]

    # Parameter sweeps
    color_distance_thresholds = [10, 15, 20, 25, 30, 40]
    shadow_thresholds = [5, 10, 15, 20, 25, 30]
    blur_sizes = [3, 5, 7]

    # Import old extraction logic
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
    from spritetools_old_working_version import extract_object_from_photo as old_extract

    # Run old extraction logic (fixed params)
    old_dir = os.path.join(base_output_dir, 'old_logic')
    os.makedirs(old_dir, exist_ok=True)
    old_out = os.path.join(old_dir, f'{basename}_old_extracted.png')
    old_mask_out = os.path.join(old_dir, f'{basename}_old_mask_debug.png')
    print("Running old extraction logic...")
    old_img, old_mask = old_extract(
        input_file,
        output_file=old_out,
        color_distance_threshold=15,
        shadow_threshold=10,
        blur_size=3,
        debug_color=True
    )
    # Save mask for old logic
    from PIL import Image
    Image.fromarray(old_mask).save(old_mask_out)
    print(f"  Old logic extracted: {old_out}")
    print(f"  Old logic mask:      {old_mask_out}")

    # Run new extraction logic with a sweep of parameters
    from spritetools import extract_object_from_photo as new_extract
    for cdt in color_distance_thresholds:
        for st in shadow_thresholds:
            for blur in blur_sizes:
                test_dir = os.path.join(base_output_dir, f'new_cdt{cdt}_st{st}_blur{blur}')
                os.makedirs(test_dir, exist_ok=True)
                out_file = os.path.join(test_dir, f'{basename}_new_extracted.png')
                mask_file = os.path.join(test_dir, f'{basename}_new_mask_debug.png')
                print(f"New extraction: cdt={cdt}, st={st}, blur={blur}")
                img, mask = new_extract(
                    input_file=input_file,
                    output_file=out_file,
                    color_distance_threshold=cdt,
                    shadow_threshold=st,
                    blur_size=blur,
                    debug_color=True
                )
                Image.fromarray(mask).save(mask_file)
                print(f"  Saved: {out_file}")
                print(f"  Mask:  {mask_file}")
    print("\nOld vs new extraction comparison complete. Check the compare_old_new/ subfolders for results.")

def demo_rembg_extraction():
    """Demonstrate background removal using the rembg library"""
    print("\n\nDEMO 5: Background Removal with rembg")
    
    # Get the first image file from the directory
    image_files = glob.glob('assets/cartier/Screenshot*.png')
    if not image_files:
        print("No image files found in assets/cartier/")
        return
    
    # Use the first image file found
    input_file = image_files[0]
    basename = os.path.basename(input_file).replace(' ', '_')
    output_dir = 'assets/cartier/rembg'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/{basename.split(".")[0]}_extracted.png'
    
    print(f"Processing image: {input_file}")
    
    model_name = 'birefnet-general'
    
    # Process with standard settings
    result = remove_background_with_rembg_simple(
        input_file=input_file,
        output_file=output_file,
        model_name=model_name,
        process_ring_hole=True  # Apply ring hole detection
    )
    
    # Try with alpha matting for better edges
    alpha_matting_output = f'{output_dir}/{basename.split(".")[0]}_alpha_matting.png'
    result_alpha = remove_background_with_rembg_simple(
        input_file=input_file,
        output_file=alpha_matting_output,
        model_name=model_name,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        process_ring_hole=True  # Apply ring hole detection
    )
    
    # Try without ring hole processing for comparison
    without_hole_processing = f'{output_dir}/{basename.split(".")[0]}_no_hole_processing.png'
    result_no_hole = remove_background_with_rembg_simple(
        input_file=input_file,
        output_file=without_hole_processing,
        model_name=model_name,
        process_ring_hole=False  # Disable ring hole detection
    )
    
    # Enhanced version 
    enhanced_output = f'{output_dir}/{basename.split(".")[0]}_enhanced.png'
    print("Enhancing extracted image")
    enhance_extracted_object(
        input_file=output_file,
        output_file=enhanced_output,
        shadow_removal_strength=0.4,
        brightness=1.05,
        contrast=1.1,
        sharpen=0.3
    )
    
    print("rembg processing complete.\n")
    print(f"- Standard with hole processing: {output_file}")
    print(f"- Without hole processing: {without_hole_processing}")
    print(f"- Alpha matting with hole processing: {alpha_matting_output}")
    print(f"- Enhanced: {enhanced_output}")

def demo_pixellib_extraction():
    """Demonstrate background removal using the pixellib library"""
    print("\n\nDEMO 6: Background Removal with pixellib")
    
    # Get the first image file from the directory
    image_files = glob.glob('assets/cartier/Screenshot*.png')
    if not image_files:
        print("No image files found in assets/cartier/")
        return
    
    # Use the first image file found
    input_file = image_files[0]
    basename = os.path.basename(input_file).replace(' ', '_')
    output_dir = 'assets/cartier/pixellib'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/{basename.split(".")[0]}_extracted.png'
    
    print(f"Processing image: {input_file}")
    
    # Process with standard settings (deeplabv3plus)
    result = remove_background_with_pixellib_simple(
        input_file=input_file,
        output_file=output_file,
        model_type="deeplabv3plus"
    )
    
    # Try with PascalVOC model which might give different results
    pascalvoc_output = f'{output_dir}/{basename.split(".")[0]}_pascalvoc.png'
    result_pascal = remove_background_with_pixellib_simple(
        input_file=input_file,
        output_file=pascalvoc_output,
        model_type="pascalvoc"
    )
    
    # Enhanced version of the first result
    enhanced_output = f'{output_dir}/{basename.split(".")[0]}_enhanced.png'
    print("Enhancing extracted image")
    enhance_extracted_object(
        input_file=output_file,
        output_file=enhanced_output,
        shadow_removal_strength=0.4,
        brightness=1.05,
        contrast=1.1,
        sharpen=0.3
    )
    
    print("pixellib processing complete.\n")
    print(f"- DeepLabV3+: {output_file}")
    print(f"- PascalVOC: {pascalvoc_output}")
    print(f"- Enhanced: {enhanced_output}")

def demo_segment_anything_extraction():
    """Demonstrate background removal using Meta's Segment Anything Model (SAM)"""
    print("\n\nDEMO 7: Background Removal with Segment Anything Model")
    
    # Get the first image file from the directory
    image_files = glob.glob('assets/cartier/Screenshot*.png')
    if not image_files:
        print("No image files found in assets/cartier/")
        return
    
    # Use the first image file found
    input_file = image_files[0]
    basename = os.path.basename(input_file).replace(' ', '_')
    output_dir = 'assets/cartier/segment_anything'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/{basename.split(".")[0]}_extracted.png'
    
    print(f"Processing image: {input_file}")
    
    # Process with standard settings
    result = remove_background_with_segment_anything_simple(
        input_file=input_file,
        output_file=output_file,
        confidence_threshold=0.3,
        points_per_side=64
    )
    
    if result is not None:
        # Try with higher detail settings
        high_detail_output = f'{output_dir}/{basename.split(".")[0]}_high_detail.png'
        result_high_detail = remove_background_with_segment_anything_simple(
            input_file=input_file,
            output_file=high_detail_output,
            confidence_threshold=0.3,  # Lower threshold to catch more details
            points_per_side=64         # More points for higher detail
        )
        
        # Enhanced version
        if os.path.exists(output_file):
            enhanced_output = f'{output_dir}/{basename.split(".")[0]}_enhanced.png'
            print("Enhancing extracted image")
            enhance_extracted_object(
                input_file=output_file,
                output_file=enhanced_output,
                shadow_removal_strength=0.4,
                brightness=1.05,
                contrast=1.1,
                sharpen=0.3
            )
            print(f"- Enhanced: {enhanced_output}")
        
        print("Segment Anything processing complete.\n")
        print(f"- Standard: {output_file}")
        if os.path.exists(high_detail_output):
            print(f"- High detail: {high_detail_output}")
    else:
        print("Segment Anything processing failed. Check the error messages above.")

def demo_dis_extraction():
    """Demonstrate background removal using the dis_bg_remover library"""
    print("\n\nDEMO 8: Background Removal with dis_bg_remover")
    
    # Get the first image file from the directory
    image_files = glob.glob('assets/cartier/Screenshot*.png')
    if not image_files:
        print("No image files found in assets/cartier/")
        return
    
    # Use the first image file found
    input_file = image_files[0]
    basename = os.path.basename(input_file).replace(' ', '_')
    output_dir = 'assets/cartier/dis_bg'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/{basename.split(".")[0]}_extracted.png'
    
    print(f"Processing image: {input_file}")
    
    # Check for model file
    model_path = "scripts/extraction/isnet_dis.onnx"
    if not os.path.exists(model_path):
        # Model might be included in the site-packages after installation
        possible_model_dirs = [".", "models", os.path.join(os.path.expanduser("~"), ".dis_bg_remover")]
        
        found = False
        for model_dir in possible_model_dirs:
            possible_path = os.path.join(model_dir, "isnet_dis.onnx")
            if os.path.exists(possible_path):
                model_path = possible_path
                found = True
                break
        
        if not found:
            print("DIS model not found. Please download it from:")
            print("https://github.com/xuebinqin/DIS/tree/main/IS-Net/models")
            print("and save it as 'isnet_dis.onnx' in the current directory")
            model_path = input("Or enter the path to the model file: ")
            if not os.path.exists(model_path):
                print(f"Model not found at {model_path}. Aborting.")
                return
    
    # Process with standard settings
    result = remove_background_with_dis_simple(
        input_file=input_file,
        output_file=output_file,
        model_path=model_path
    )
    
    if result is not None:
        # Enhanced version of the standard extraction
        if os.path.exists(output_file):
            enhanced_output = f'{output_dir}/{basename.split(".")[0]}_enhanced.png'
            print("Enhancing extracted image")
            enhance_extracted_object(
                input_file=output_file,
                output_file=enhanced_output,
                shadow_removal_strength=0.4,
                brightness=1.05,
                contrast=1.1,
                sharpen=0.3
            )
            print(f"- Enhanced: {enhanced_output}")
        
        print("dis_bg_remover processing complete.\n")
        print(f"- Standard extraction: {output_file}")
        print(f"- Enhanced extraction: {enhanced_output}")
    else:
        print("dis_bg_remover processing failed. Check the error messages above.")

def demo_batch_rembg_extraction():
    """Demonstrate batch background removal using the rembg library with birefnet-general model"""
    print("\n\nDEMO 9: Batch Background Removal with rembg and birefnet-general")
    
    input_dir = 'assets/cartier'
    
    # Create output directories
    output_dir_with_holes = 'assets/cartier/rembg/batch_with_holes'
    output_dir_no_holes = 'assets/cartier/rembg/batch_no_holes'
    os.makedirs(output_dir_with_holes, exist_ok=True)
    os.makedirs(output_dir_no_holes, exist_ok=True)
    
    # Process with ring holes preserved
    print("\nProcessing all images with ring holes preserved:")
    with_holes_files = batch_remove_background_with_rembg(
        input_dir=input_dir,
        output_dir=output_dir_with_holes,
        model_name="birefnet-general",
        keep_ring_hole=True,
        file_pattern="Screenshot*.png"
    )
    
    # Process without ring holes (filling them)
    print("\nProcessing all images with ring holes filled:")
    no_holes_files = batch_remove_background_with_rembg(
        input_dir=input_dir,
        output_dir=output_dir_no_holes,
        model_name="birefnet-general",
        keep_ring_hole=False,
        file_pattern="Screenshot*.png"
    )
    
    print("\nBatch processing complete!")
    print(f"- Images with ring holes preserved: {len(with_holes_files)} (saved to {output_dir_with_holes})")
    print(f"- Images with ring holes filled: {len(no_holes_files)} (saved to {output_dir_no_holes})")
    
    # Process a single example with alpha matting for comparison
    if with_holes_files:
        # Choose the first processed file as an example for alpha matting
        example_file = os.path.join(input_dir, os.path.basename(with_holes_files[0]).replace('_rembg.png', '.png'))
        
        if os.path.exists(example_file):
            alpha_matting_output = f'{output_dir_with_holes}/example_alpha_matting.png'
            print("\nCreating an example with alpha matting for better edges...")
            result_alpha = remove_background_with_rembg_simple(
                input_file=example_file,
                output_file=alpha_matting_output,
                model_name="birefnet-general",
                alpha_matting=True,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                process_ring_hole=True
            )
            
            if result_alpha:
                print(f"- Alpha matting example: {alpha_matting_output}")

def demo_cartier_batch_birefnet():
    """Demonstrate batch background removal specifically optimized for Cartier rings using the birefnet model"""
    print("\n\nDEMO 10: Cartier Ring Batch Processing with BirefNet")
    
    # input_dir = 'assets/cartier'
    input_dir = 'assets/logo'
    
    # Create specialized output directory for Cartier rings
    # output_dir = 'assets/cartier/rembg/cartier_birefnet'
    # enhanced_dir = 'assets/cartier/rembg/cartier_birefnet_enhanced'
    output_dir = 'assets/logo/regular'
    enhanced_dir = 'assets/logo/enhanced'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(enhanced_dir, exist_ok=True)
    
    # Process Cartier ring photos with specialized settings
    print("\nProcessing all Cartier ring photos with optimized settings:")
    processed_files = batch_process_cartier_with_birefnet(
        input_dir=input_dir,
        input_pattern="photo*.jpg",
        output_dir=output_dir,
    )
    
    # Count the number of processed files
    num_processed = len(processed_files) if processed_files else 0
    print(f"\nProcessed {num_processed} Cartier ring photos with BirefNet model")
    
    # Enhance the extracted images
    if processed_files:
        print("\nEnhancing extracted ring images:")
        enhanced_files = []
        
        for i, file_path in enumerate(processed_files):
            # Create name for enhanced file
            basename = os.path.basename(file_path)
            base_name = os.path.splitext(basename)[0]
            enhanced_path = os.path.join(enhanced_dir, f"{base_name}_enhanced.png")
            enhanced_files.append(enhanced_path)
            
            # Apply enhancement with parameters optimized for metallic rings
            print(f"Enhancing: {basename}")
            enhance_extracted_object(
                input_file=file_path,
                output_file=enhanced_path,
                shadow_removal_strength=0.35,  # Reduced shadow removal to preserve metallic details
                brightness=1.03,              # Slight brightness increase
                contrast=1.08,                # Moderate contrast for gold rings
                sharpen=0.2                   # Subtle sharpening to avoid artifacts on reflective surfaces
            )
        
        print("\nCartier BirefNet batch processing complete!")
        print(f"- Original images:  {num_processed}")
        print(f"- Extracted images: {num_processed}")
        print(f"- Enhanced images:  {len(enhanced_files)}")
        print("\nOutputs saved to:")
        print(f"- Extracted: {output_dir}")
        print(f"- Enhanced: {enhanced_dir}")
    else:
        print("No images were processed. Check that the input directory contains matching files.")

if __name__ == "__main__":
    main()
