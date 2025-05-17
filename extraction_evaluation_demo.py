\
import os
import sys
import argparse
import glob
from pathlib import Path

# Add scripts directory to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

# Import the extraction modules
from scripts.extraction import (
    extract_object_from_photo,
    # enhance_extracted_object, # Not used in this evaluation script directly
    remove_background_with_rembg_simple,
    remove_background_with_pixellib_simple,
    remove_background_with_segment_anything_simple,
    remove_background_with_dis_simple
)
from scripts.extraction.background_remover_tool import remove_background_br

def get_output_path(base_dir, input_filename, method_name, params_suffix_with_ext):
    # Ensure the params_suffix_with_ext already contains the desired extension
    # and potentially the original base filename if needed for uniqueness.
    # Example: output_filename = f"{base_fn}_{method_name}_{params_suffix}{ext}"
    # For simplicity here, params_suffix_with_ext is the full desired filename part after method_name.
    
    # Construct path: base_dir / method_name / filename_method_params.ext
    # Extract original filename without extension to prepend
    original_base_fn, _ = os.path.splitext(os.path.basename(input_filename))
    
    # Ensure params_suffix_with_ext includes the extension
    # e.g., params_suffix_with_ext = "u2net_alpha.png"
    output_filename = f"{original_base_fn}_{params_suffix_with_ext}"
    
    return os.path.join(base_dir, method_name, output_filename)

def ensure_dir(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)

def run_rembg_tests(input_file, base_output_dir):
    method_name = "rembg"
    output_dir_method = os.path.join(base_output_dir, method_name)
    ensure_dir(output_dir_method)
    print(f"--- Running {method_name} tests ---")
    _, ext = os.path.splitext(input_file)

    tests = [
        {"model": "u2net", "alpha_matting": False, "suffix": f"u2net_default{ext}"},
        {"model": "u2net", "alpha_matting": True, "suffix": f"u2net_alpha{ext}"},
        {"model": "u2net_human_seg", "alpha_matting": False, "suffix": f"u2nethumanseg_default{ext}"},
        {"model": "birefnet-general", "alpha_matting": False, "suffix": f"birefnet_default{ext}"},
        {"model": "birefnet-general", "alpha_matting": True, "suffix": f"birefnet_alpha{ext}"},
    ]

    for test_params in tests:
        output_path = get_output_path(base_output_dir, input_file, method_name, test_params["suffix"])
        print(f"  Processing with {test_params['suffix'].split('.')[0]} -> {os.path.basename(output_path)}")
        try:
            remove_background_with_rembg_simple(
                input_file=input_file,
                output_file=output_path,
                model_name=test_params["model"],
                alpha_matting=test_params["alpha_matting"],
            )
        except Exception as e:
            print(f"    Error during rembg ({test_params['suffix'].split('.')[0]}): {e}")

def run_pixellib_tests(input_file, base_output_dir):
    method_name = "pixellib"
    output_dir_method = os.path.join(base_output_dir, method_name)
    ensure_dir(output_dir_method)
    print(f"--- Running {method_name} tests ---")
    _, ext = os.path.splitext(input_file)

    tests = [
        {"model": "deeplabv3plus", "suffix": f"deeplab{ext}"},
        {"model": "pascalvoc", "suffix": f"pascalvoc{ext}"},
    ]
    for test_params in tests:
        output_path = get_output_path(base_output_dir, input_file, method_name, test_params["suffix"])
        print(f"  Processing with {test_params['suffix'].split('.')[0]} -> {os.path.basename(output_path)}")
        try:
            remove_background_with_pixellib_simple(
                input_file=input_file,
                output_file=output_path,
                model_type=test_params["model"]
            )
        except Exception as e:
            print(f"    Error during pixellib ({test_params['suffix'].split('.')[0]}): {e}")

def run_sam_tests(input_file, base_output_dir):
    method_name = "segment_anything"
    output_dir_method = os.path.join(base_output_dir, method_name)
    ensure_dir(output_dir_method)
    print(f"--- Running {method_name} tests ---")
    _, ext = os.path.splitext(input_file)

    tests = [
        {"conf": 0.3, "pps": 32, "suffix": f"conf0.3_pps32{ext}"},
        {"conf": 0.5, "pps": 32, "suffix": f"conf0.5_pps32{ext}"},
        {"conf": 0.3, "pps": 64, "suffix": f"conf0.3_pps64{ext}"},
    ]
    for test_params in tests:
        output_path = get_output_path(base_output_dir, input_file, method_name, test_params["suffix"])
        print(f"  Processing with {test_params['suffix'].split('.')[0]} -> {os.path.basename(output_path)}")
        try:
            remove_background_with_segment_anything_simple(
                input_file=input_file,
                output_file=output_path,
                confidence_threshold=test_params["conf"],
                points_per_side=test_params["pps"]
            )
        except Exception as e:
            print(f"    Error during SAM ({test_params['suffix'].split('.')[0]}): {e}")

def run_dis_tests(input_file, base_output_dir):
    method_name = "dis_bg"
    output_dir_method = os.path.join(base_output_dir, method_name)
    ensure_dir(output_dir_method)
    print(f"--- Running {method_name} tests ---")
    _, ext = os.path.splitext(input_file)
    
    model_path_to_try = "scripts/extraction/isnet_dis.onnx" 
    # Attempt to locate the model if not at the default path
    if not os.path.exists(model_path_to_try):
        print(f"  DIS model not found at {model_path_to_try}, searching common locations...")
        possible_model_dirs = [".", "models", os.path.join(os.path.expanduser("~"), ".dis_bg_remover"), "scripts/extraction"]
        found_model = False
        for model_dir_search in possible_model_dirs:
            potential_path = os.path.join(model_dir_search, "isnet_dis.onnx")
            if os.path.exists(potential_path):
                model_path_to_try = potential_path
                found_model = True
                print(f"  Found DIS model at: {model_path_to_try}")
                break
        if not found_model:
            print("  DIS model 'isnet_dis.onnx' not found. Please ensure it's available. Skipping DIS tests.")
            return

    output_path = get_output_path(base_output_dir, input_file, method_name, f"default{ext}")
    print(f"  Processing with default DIS model -> {os.path.basename(output_path)}")
    try:
        remove_background_with_dis_simple(
            input_file=input_file,
            output_file=output_path,
            model_path=model_path_to_try
        )
    except Exception as e:
        print(f"    Error during DIS: {e}")

def run_backgroundremover_tests(input_file, base_output_dir):
    method_name = "backgroundremover_pkg"
    output_dir_method = os.path.join(base_output_dir, method_name)
    ensure_dir(output_dir_method)
    print(f"--- Running {method_name} tests ---")
    _, ext = os.path.splitext(input_file)

    tests = [
        {"model": "u2net", "alpha_matting": True, "fg": 240, "bg": 10, "erode": 10, "suffix": f"u2net_alpha_default{ext}"},
        {"model": "u2net", "alpha_matting": False, "suffix": f"u2net_no_alpha{ext}"},
        {"model": "u2net", "alpha_matting": True, "fg": 200, "bg": 20, "erode": 5, "suffix": f"u2net_alpha_tuned1{ext}"},
        {"model": "u2net_human_seg", "alpha_matting": True, "fg": 240, "bg": 10, "erode": 10, "suffix": f"u2nethuman_alpha{ext}"},
        {"model": "u2netp", "alpha_matting": True, "fg": 240, "bg": 10, "erode": 10, "suffix": f"u2netp_alpha{ext}"},
    ]

    for test_params in tests:
        output_path = get_output_path(base_output_dir, input_file, method_name, test_params["suffix"])
        print(f"  Processing with {test_params['suffix'].split('.')[0]} -> {os.path.basename(output_path)}")
        try:
            remove_background_br(
                input_path=input_file,
                output_path=output_path,
                model_name=test_params["model"],
                alpha_matting=test_params["alpha_matting"],
                fg_threshold=test_params.get("fg", 240),
                bg_threshold=test_params.get("bg", 10),
                erode_size=test_params.get("erode", 10)
            )
        except Exception as e:
            print(f"    Error during backgroundremover ({test_params['suffix'].split('.')[0]}): {e}")

def run_custom_extraction_tests(input_file, base_output_dir):
    method_name = "custom_spritetools"
    output_dir_method = os.path.join(base_output_dir, method_name)
    ensure_dir(output_dir_method)
    print(f"--- Running {method_name} tests ---")
    _, ext = os.path.splitext(input_file)

    tests = [
        {"st": 20, "cdt": 15, "blur": 5, "suffix": f"st20_cdt15_blur5{ext}"},
        {"st": 10, "cdt": 25, "blur": 3, "suffix": f"st10_cdt25_blur3{ext}"},
        {"st": 20, "cdt": 30, "blur": 5, "mfix": True, "ahf": True, "suffix": f"st20_cdt30_blur5_mfix_ahf{ext}"},
    ]

    for test_params in tests:
        output_path = get_output_path(base_output_dir, input_file, method_name, test_params["suffix"])
        print(f"  Processing with {test_params['suffix'].split('.')[0]} -> {os.path.basename(output_path)}")
        try:
            extract_object_from_photo(
                input_file=input_file,
                output_file=output_path,
                shadow_threshold=test_params["st"],
                color_distance_threshold=test_params["cdt"],
                blur_size=test_params["blur"],
                metallic_reflection_fix=test_params.get("mfix", False),
                advanced_hole_filling=test_params.get("ahf", False),
                debug_color=True 
            )
        except Exception as e:
            print(f"    Error during custom_spritetools ({test_params['suffix'].split('.')[0]}): {e}")

def main():
    parser = argparse.ArgumentParser(description="Run a suite of background extraction tests on a single image.")
    parser.add_argument("-i", "--input", required=True, help="Input image file.")
    parser.add_argument("-o", "--output_dir", required=True, help="Base directory to save output images.")
    
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    ensure_dir(args.output_dir)
    print(f"Starting extraction suite for: {args.input}")
    print(f"Outputs will be saved in subdirectories of: {args.output_dir}")

    run_rembg_tests(args.input, args.output_dir)
    run_pixellib_tests(args.input, args.output_dir)
    run_sam_tests(args.input, args.output_dir)
    run_dis_tests(args.input, args.output_dir)
    run_backgroundremover_tests(args.input, args.output_dir)
    run_custom_extraction_tests(args.input, args.output_dir)
    
    print("\\n--- Extraction Suite Complete ---")
    print(f"All outputs saved under: {args.output_dir}")

if __name__ == "__main__":
    main()
