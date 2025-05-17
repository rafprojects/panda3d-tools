# filepath: /home/user/projects/py_projects/pandas_learn/scripts/extraction/background_remover_tool.py
try:
    from backgroundremover.bg import remove as br_remove
    HAS_BACKGROUNDREMOVER = True
except ImportError:
    HAS_BACKGROUNDREMOVER = False
import os

def remove_background_br(input_path, output_path, model_name="u2net", 
                         alpha_matting=True, fg_threshold=240, bg_threshold=10, 
                         erode_size=10, base_size=1000):
    """
    Removes the background from an image using the backgroundremover package.

    Args:
        input_path (str): Path to the input image.
        output_path (str): Path to save the output image with background removed.
        model_name (str): Model to use ("u2net", "u2net_human_seg", "u2netp").
        alpha_matting (bool): Whether to use alpha matting.
        fg_threshold (int): Alpha matting foreground threshold.
        bg_threshold (int): Alpha matting background threshold.
        erode_size (int): Alpha matting erode structure size.
        base_size (int): Alpha matting base size.
    """
    if not HAS_BACKGROUNDREMOVER:
        raise ImportError("backgroundremover package is not installed. Please install it with 'pip install backgroundremover'")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(input_path, 'rb') as i_file:
            input_data = i_file.read()
        
        output_data = br_remove(
            input_data, 
            model_name=model_name,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=fg_threshold,
            alpha_matting_background_threshold=bg_threshold,
            alpha_matting_erode_structure_size=erode_size,
            alpha_matting_base_size=base_size
        )
        
        with open(output_path, 'wb') as o_file:
            o_file.write(output_data)
        
        print(f"Background removal complete using '{model_name}'. Output saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error during background removal with backgroundremover: {e}")
        raise
