"""
PixelLib extraction module for removing backgrounds from images using PixelLib's segmentation models.
PixelLib provides semantic segmentation capabilities for identifying objects in images.
"""
import os
import numpy as np
import cv2
from PIL import Image

def remove_background_with_pixellib(input_file, output_file=None, model_type="deeplabv3plus",
                                   model_path=None, post_process=True):
    """Extract object from a photo using the pixellib library.
    
    Args:
        input_file (str): Path to the input image file
        output_file (str, optional): Path to save the output image
        model_type (str, optional): Type of model to use
        model_path (str, optional): Path to custom model weights
        post_process (bool, optional): Whether to smooth the edges of the mask
    
    Returns:
        PIL.Image or None: Processed image with transparent background, or None if processing failed
    """
    try:
        # Try to import the required libraries (may fail if not installed)
        import pixellib
        from pixellib.semantic import semantic_segmentation
        import cv2
        import numpy as np
        import os
        
        print(f"Processing image with pixellib (model: {model_type})...")
        
        # Set up the segmentation model
        segment = semantic_segmentation()
        
        try:
            if model_path:
                # Use custom model if provided
                segment.load_pascalvoc_model(model_path)
            else:
                # Use default model
                if model_type == "pascalvoc":
                    # Check if model file exists
                    if not os.path.exists("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"):
                        print("Downloading pascalvoc model (this may take a few minutes)...")
                        try:
                            import gdown
                            url = "https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
                            output = "deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
                            gdown.download(url, output, quiet=False)
                        except Exception as e:
                            print(f"Error downloading model: {e}")
                            print("Please download the model manually from: https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
                            return None
                    segment.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
                else:  # deeplabv3plus by default
                    # Check if model file exists
                    if not os.path.exists("deeplabv3plus_xception65_ade20k.h5"):
                        print("Downloading deeplabv3plus model (this may take a few minutes)...")
                        try:
                            import gdown
                            url = "https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.1/deeplabv3plus_xception65_ade20k.h5"
                            output = "deeplabv3plus_xception65_ade20k.h5"
                            gdown.download(url, output, quiet=False)
                        except Exception as e:
                            print(f"Error downloading model: {e}")
                            print("Please download the model manually from: https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.1/deeplabv3plus_xception65_ade20k.h5")
                            return None
                    segment.load_ade20k_model("deeplabv3plus_xception65_ade20k.h5")
        
            # Process the image
            segvalues, output = segment.segmentAsAde20k(input_file, process_frame=True)
            
            # Read the original image to extract the objects
            img = cv2.imread(input_file)
            height, width, _ = img.shape
            
            # Create a blank image with transparency
            result = np.zeros((height, width, 4), dtype=np.uint8)
            
            # Look for likely object categories - focusing on objects, not backgrounds
            object_classes = []
            for segclass in segvalues["class_names"]:
                # Exclude common background classes
                if segclass.lower() not in ["wall", "sky", "floor", "background", "ceiling", "ground"]:
                    object_classes.append(segclass)
                    
            # If no clear object was found, default to the non-background class with largest area
            if len(object_classes) == 0:
                areas = {}
                for i, segclass in enumerate(segvalues["class_names"]):
                    if segclass.lower() not in ["wall", "sky", "floor", "background", "ceiling", "ground"]:
                        segmap = output == i
                        areas[segclass] = np.sum(segmap)
                
                if areas:
                    object_classes = [max(areas, key=areas.get)]
                else:
                    # If still no objects, use the most prominent segment
                    areas = {}
                    for i, segclass in enumerate(segvalues["class_names"]):
                        segmap = output == i
                        areas[segclass] = np.sum(segmap)
                    object_classes = [max(areas, key=areas.get)]
            
            # Create the combined mask for all found objects
            mask = np.zeros((height, width), dtype=np.uint8)
            for obj_class in object_classes:
                class_idx = segvalues["class_names"].index(obj_class)
                class_mask = output == class_idx
                mask = np.logical_or(mask, class_mask)
            
            # Post-process the mask if requested
            if post_process:
                # Apply morphological operations to clean up the mask
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                # Apply Gaussian blur for smoother edges
                mask = cv2.GaussianBlur(mask, (5, 5), 0)
                
                # Threshold again to get binary mask
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            else:
                mask = mask.astype(np.uint8) * 255
            
            # Copy RGB channels from original image
            result[:, :, 0:3] = img[:, :, ::-1]  # Convert BGR to RGB
            
            # Set alpha channel from mask
            result[:, :, 3] = mask
            
            # Convert to PIL image
            pil_img = Image.fromarray(result)
            
            # Save the result if output path is provided
            if output_file:
                # Ensure directory exists
                os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
                pil_img.save(output_file)
                print(f"Result saved to {output_file}")
            else:
                # Create default output path
                base_name = os.path.splitext(input_file)[0]
                default_output = f"{base_name}_pixellib.png"
                pil_img.save(default_output)
                print(f"Result saved to {default_output}")
                output_file = default_output
            
            # Save mask debug image
            mask_debug_path = os.path.splitext(output_file)[0] + "_mask_debug.png"
            Image.fromarray(mask).save(mask_debug_path)
            print(f"Mask debug image saved to {mask_debug_path}")
            
            return pil_img
        
        except Exception as e:
            print(f"Error during pixellib processing: {e}")
            print("This might be due to version incompatibilities. Consider creating a separate virtual environment.")
            return None
            
    except ImportError as e:
        print(f"Error: Could not import pixellib or its dependencies. Please install with 'pip install pixellib gdown': {e}")
        print("If you're getting numpy incompatibility errors, try creating a fresh virtual environment.")
        return None

def remove_background_with_pixellib_simple(input_file, output_file=None, model_type="deeplabv3plus",
                                         model_path=None, post_process=True):
    """Extract object from a photo using the pixellib library (simplified version).
    
    This is a simplified version that focuses on the core functionality without debug outputs.
    
    Args:
        input_file (str): Path to the input image file
        output_file (str, optional): Path to save the output image
        model_type (str, optional): Type of model to use
        model_path (str, optional): Path to custom model weights
        post_process (bool, optional): Whether to smooth the edges of the mask
    
    Returns:
        PIL.Image or None: Processed image with transparent background, or None if processing failed
    """
    try:
        # Try to import the required libraries (may fail if not installed)
        import pixellib
        from pixellib.semantic import semantic_segmentation
        
        print(f"Processing image with pixellib (model: {model_type})...")
        
        # Set up the segmentation model
        segment = semantic_segmentation()
        
        try:
            if model_path:
                # Use custom model if provided
                segment.load_pascalvoc_model(model_path)
            else:
                # Use default model
                if model_type == "pascalvoc":
                    # Check if model file exists
                    if not os.path.exists("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"):
                        print("Downloading pascalvoc model (this may take a few minutes)...")
                        try:
                            import gdown
                            url = "https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
                            output = "deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
                            gdown.download(url, output, quiet=False)
                        except Exception as e:
                            print(f"Error downloading model: {e}")
                            print("Please download the model manually from: https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
                            return None
                    segment.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
                else:  # deeplabv3plus by default
                    # Check if model file exists
                    if not os.path.exists("deeplabv3plus_xception65_ade20k.h5"):
                        print("Downloading deeplabv3plus model (this may take a few minutes)...")
                        try:
                            import gdown
                            url = "https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.1/deeplabv3plus_xception65_ade20k.h5"
                            output = "deeplabv3plus_xception65_ade20k.h5"
                            gdown.download(url, output, quiet=False)
                        except Exception as e:
                            print(f"Error downloading model: {e}")
                            print("Please download the model manually from: https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.1/deeplabv3plus_xception65_ade20k.h5")
                            return None
                    segment.load_ade20k_model("deeplabv3plus_xception65_ade20k.h5")
        
            # Process the image
            segvalues, output = segment.segmentAsAde20k(input_file, process_frame=True)
            
            # Read the original image to extract the objects
            img = cv2.imread(input_file)
            height, width, _ = img.shape
            
            # Create a mask for object
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Look for likely object categories - focusing on objects, not backgrounds
            object_classes = []
            for segclass in segvalues["class_names"]:
                # Exclude common background classes
                if segclass.lower() not in ["wall", "sky", "floor", "background", "ceiling", "ground"]:
                    object_classes.append(segclass)
                    
            # If no clear object was found, default to the non-background class with largest area
            if len(object_classes) == 0:
                areas = {}
                for i, segclass in enumerate(segvalues["class_names"]):
                    if segclass.lower() not in ["wall", "sky", "floor", "background", "ceiling", "ground"]:
                        segmap = output == i
                        areas[segclass] = np.sum(segmap)
                
                if areas:
                    object_classes = [max(areas, key=areas.get)]
                else:
                    # If still no objects, use the most prominent segment
                    areas = {}
                    for i, segclass in enumerate(segvalues["class_names"]):
                        segmap = output == i
                        areas[segclass] = np.sum(segmap)
                    object_classes = [max(areas, key=areas.get)]
            
            # Create the combined mask for all found objects
            for obj_class in object_classes:
                class_idx = segvalues["class_names"].index(obj_class)
                class_mask = output == class_idx
                mask = np.logical_or(mask, class_mask)
            
            # Post-process the mask if requested
            if post_process:
                # Apply morphological operations to clean up the mask
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                # Apply Gaussian blur for smoother edges
                mask = cv2.GaussianBlur(mask, (5, 5), 0)
                
                # Threshold again to get binary mask
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            else:
                mask = mask.astype(np.uint8) * 255
            
            # Load the original image in PIL for transparency
            orig_img = Image.open(input_file).convert("RGBA")
            orig_np = np.array(orig_img)
            
            # Apply the mask to the alpha channel
            orig_np[..., 3] = mask
            
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
                default_output = f"{base_name}_pixellib.png"
                result_img.save(default_output)
                print(f"Result saved to {default_output}")
                output_file = default_output
            
            return result_img
        
        except Exception as e:
            print(f"Error during pixellib processing: {e}")
            print("This might be due to version incompatibilities. Consider creating a separate virtual environment.")
            return None
            
    except ImportError as e:
        print(f"Error: Could not import pixellib or its dependencies. Please install with 'pip install pixellib gdown': {e}")
        print("If you're getting numpy incompatibility errors, try creating a fresh virtual environment.")
        return None