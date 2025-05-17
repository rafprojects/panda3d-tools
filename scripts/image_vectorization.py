import os
import vtracer

def vectorize_image_vtracer(
    input_path: str,
    output_path: str,
    mode: str = "spline", # "color" or "binary"
    colormode: str = "color", # "color" or "binary" (for vtracer library)
    hierarchical: str = "stacked", # "stacked" or "cutout"
    filter_speckle: int = 4, # Filter speckle
    color_precision: int = 6, # Color precision
    layer_difference: int = 16, # Layer difference
    corner_threshold: int = 60, # Corner threshold
    length_threshold: int = 4.0, # Length threshold
    max_iterations: int = 10, # Max iterations
    splice_threshold: int = 45, # Splice threshold
    path_precision: int | None = 3  # Path precision, None for optimal
):
    """
    Vectorizes an image using the vtracer library.

    Args:
        input_path: Path to the input raster image.
        output_path: Path to save the output SVG file.
        mode: "color" for color tracing, "binary" for monochrome.
        colormode: "color" or "binary" - vtracer's parameter for color mode.
        hierarchical: "stacked" (layers) or "cutout" (paths).
        filter_speckle: Filter speckles smaller than this size (pixels).
        color_precision: Number of bits for color quantization.
        layer_difference: Difference between layers (pixels).
        corner_threshold: Threshold for corner detection (degrees).
        length_threshold: Minimum length of segments (pixels).
        max_iterations: Maximum iterations for tracing.
        splice_threshold: Threshold for splicing segments (degrees).
        path_precision: Number of decimal places for path coordinates (None for auto).
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Vectorizing {input_path} to {output_path} with vtracer...")
    print(f"Parameters: mode='{mode}', colormode='{colormode}', hierarchical='{hierarchical}', "
          f"filter_speckle={filter_speckle}, color_precision={color_precision}, "
          f"layer_difference={layer_difference}, corner_threshold={corner_threshold}, "
          f"length_threshold={length_threshold}, max_iterations={max_iterations}, "
          f"splice_threshold={splice_threshold}, path_precision={path_precision}"
        )

    try:
        vtracer.convert_image_to_svg_py(
            input_path,
            output_path,
            mode=mode,
            colormode=colormode,
            hierarchical=hierarchical,
            filter_speckle=filter_speckle,
            color_precision=color_precision,
            layer_difference=layer_difference,
            corner_threshold=corner_threshold,
            length_threshold=length_threshold,
            max_iterations=max_iterations,
            splice_threshold=splice_threshold,
            path_precision=path_precision
        )
        print(f"Successfully vectorized image and saved to {output_path}")
    except Exception as e:
        print(f"Error during vtracer processing: {e}")
        raise

if __name__ == '__main__':
    # Create a dummy image for testing if it doesn't exist
    # Requires Pillow for this test part: pip install Pillow
    from PIL import Image, ImageDraw

    dummy_input = "dummy_raster_for_vector_test.png"
    dummy_output_color = "dummy_vector_output_color.svg"
    dummy_output_binary = "dummy_vector_output_binary.svg"

    if not os.path.exists(dummy_input):
        img = Image.new('RGB', (100, 100), color = 'red')
        draw = ImageDraw.Draw(img)
        draw.ellipse((20, 20, 80, 80), fill='blue', outline='green', width=5)
        draw.rectangle((30, 5, 70, 35), fill='yellow')
        img.save(dummy_input)
        print(f"Created dummy input: {dummy_input}")

    print("\nTesting color vectorization...")
    try:
        vectorize_image_vtracer(
            dummy_input,
            dummy_output_color,
            mode="color",
            colormode="color",
            path_precision=3
        )
    except Exception as e:
        print(f"Test failed for color vectorization: {e}")


    print("\nTesting binary vectorization...")
    try:
        vectorize_image_vtracer(
            dummy_input,
            dummy_output_binary,
            mode="binary",
            colormode="binary", # Important for binary mode
            filter_speckle=2,
            path_precision=3
        )
    except Exception as e:
        print(f"Test failed for binary vectorization: {e}")

    # Example of how to call from another script:
    # from image_vectorization import vectorize_image_vtracer
    # vectorize_image_vtracer("input.png", "output.svg", mode="color")
