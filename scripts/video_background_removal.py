import subprocess
import os
import shutil
import tempfile
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

def find_executable(executable_name):
    """Helper function to find an executable in PATH or common virtualenv locations."""
    if shutil.which(executable_name):
        return executable_name
    
    # Check common venv paths if not in global PATH
    # This is a simplified check; a more robust solution might be needed
    # if Python environments are complex.
    try:
        import sys
        venv_path = sys.prefix
        possible_path = os.path.join(venv_path, 'bin', executable_name)
        if os.path.exists(possible_path) and os.access(possible_path, os.X_OK):
            return possible_path
    except Exception:
        pass # Ignore errors if sys.prefix is not what we expect
        
    return None # Executable not found

def despill_color_channel(image_bgra, spill_color_name: str = "green", spill_threshold_factor=1.05, correction_intensity=0.7):
    """
    Reduces color spill on the foreground of an image with an alpha channel.

    Args:
        image_bgra: Input image in BGRA format.
        spill_color_name (str): The name of the spill color to target ("green", "blue", "red").
        spill_threshold_factor (float): How much stronger the target spill channel needs to be
                                        compared to the other two primary color channels.
        correction_intensity (float): How strongly to correct the spill channel.
    Returns:
        Despilled image in BGRA format.
    """
    if not isinstance(image_bgra, np.ndarray) or image_bgra.ndim != 3 or image_bgra.shape[2] != 4:
        return image_bgra

    b, g, r, a = cv2.split(image_bgra)
    b_f, g_f, r_f = b.astype(np.float32), g.astype(np.float32), r.astype(np.float32)

    foreground_mask = (a > 0)
    
    target_channel_f = None
    other_channel1_f = None
    other_channel2_f = None
    is_valid_color = True

    if spill_color_name == "green":
        target_channel_f = g_f
        other_channel1_f = r_f
        other_channel2_f = b_f
    elif spill_color_name == "blue":
        target_channel_f = b_f
        other_channel1_f = r_f
        other_channel2_f = g_f
    elif spill_color_name == "red":
        target_channel_f = r_f
        other_channel1_f = g_f
        other_channel2_f = b_f
    else:
        print(f"Warning: Unsupported spill_color_name: {spill_color_name}. No despill applied.")
        is_valid_color = False

    if not is_valid_color:
        return image_bgra

    is_spill = (target_channel_f > other_channel1_f * spill_threshold_factor) & \
               (target_channel_f > other_channel2_f * spill_threshold_factor) & \
               foreground_mask

    target_val_for_spill = np.maximum(other_channel1_f, other_channel2_f)
    corrected_target_channel_f = target_channel_f - (target_channel_f - target_val_for_spill) * correction_intensity
    corrected_target_channel_f = np.maximum(corrected_target_channel_f, 0) # Ensure non-negative

    final_target_channel_f = np.where(is_spill, corrected_target_channel_f, target_channel_f)
    final_target_channel_uint8 = np.clip(final_target_channel_f, 0, 255).astype(np.uint8)

    if spill_color_name == "green":
        return cv2.merge((b, final_target_channel_uint8, r, a))
    elif spill_color_name == "blue":
        return cv2.merge((final_target_channel_uint8, g, r, a))
    elif spill_color_name == "red":
        return cv2.merge((b, g, final_target_channel_uint8, a))
    
    return image_bgra # Should not reach here if spill_color_name is valid

def remove_background_video_rembg(
    input_video_path: str,
    output_video_path: str,
    model: str = "u2net",
    extra_args: list = None,
    max_workers: int = 4,
    quality: int = 95,
    despill: bool = False, # New parameter to enable/disable despill
    despill_color: str = "green", # New parameter for despill color
    despill_threshold: float = 1.05, # New parameter
    despill_intensity: float = 0.7    # New parameter
):
    """
    Removes background from a video using rembg in a frame-by-frame approach.
    Includes optional color despill functionality.

    Args:
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to save the output video file.
        model (str): Model to use (e.g., "u2net", "u2net_human_seg").
        extra_args (list, optional): List of additional arguments to pass to rembg CLI for each frame.
        max_workers (int): Maximum number of parallel workers for frame processing.
        quality (int): JPEG quality for temporary frame storage (1-100).
        despill (bool): Whether to apply color despill.
        despill_color (str): Target color for despill ("green", "blue", "red").
        despill_threshold (float): Threshold factor for despill detection.
        despill_intensity (float): Intensity of the despill correction.
    Returns:
        bool: True if successful, False otherwise.
    """
    if not os.path.exists(input_video_path):
        print(f"Error: Input video not found at {input_video_path}")
        return False

    rembg_executable = find_executable("rembg")
    if not rembg_executable:
        print("Error: 'rembg' executable not found. Please ensure it's installed and in your PATH.")
        print("You can install it with: pip install rembg[cli]")
        return False

    # Create a temporary directory for storing frames
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory for frames: {temp_dir}")

        # Step 1: Extract frames from input video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Error: Could not open input video: {input_video_path}")
            return False

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {frame_width}x{frame_height}, {fps} fps, {total_frames} frames")

        # Function to process a single frame
        def process_frame(frame_idx):
            input_frame_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.png")
            output_frame_path = os.path.join(temp_dir, f"processed_{frame_idx:06d}.png")
            
            command = [
                rembg_executable,
                "i",  # image processing command
                "-m", model
            ]
            
            # Add extra arguments if provided
            if extra_args:
                command.extend(extra_args)
                
            # Add input and output paths
            command.extend([input_frame_path, output_frame_path])
            
            try:
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return True
            except subprocess.CalledProcessError as e:
                print(f"Error processing frame {frame_idx}: {e}")
                if e.stderr:
                    print(f"stderr: {e.stderr.decode()}")
                return False

        # Extract and save all frames
        print(f"Extracting {total_frames} frames from video...")
        frame_idx = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Save frame as PNG for transparency support
            frame_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(frame_path, frame)
            
            frame_idx += 1
            
            # Show progress every 100 frames
            if frame_idx % 100 == 0:
                elapsed = time.time() - start_time
                frames_per_second = frame_idx / elapsed if elapsed > 0 else 0
                print(f"Extracted {frame_idx}/{total_frames} frames ({frames_per_second:.2f} frames/sec)")
        
        cap.release()
        print(f"Extracted {frame_idx} frames in {time.time() - start_time:.2f} seconds")

        # Step 2: Process frames in parallel using rembg
        print(f"Processing frames with rembg using model '{model}'...")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_frame, range(frame_idx)))
            
        if not all(results):
            print("Error: Some frames failed to process with rembg")
            return False
            
        print(f"All frames processed in {time.time() - start_time:.2f} seconds")

        # Step 3: Create output video from processed frames
        print(f"Creating output video at {output_video_path}...")
        output_dir = os.path.dirname(output_video_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1' for better compatibility
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), True)
        
        if not out.isOpened():
            print(f"Error: Could not create output video file: {output_video_path}")
            return False
            
        # Add processed frames to the video
        start_time = time.time()
        for i in range(frame_idx):
            processed_frame_path = os.path.join(temp_dir, f"processed_{i:06d}.png")
            if os.path.exists(processed_frame_path):
                processed_frame_bgra = cv2.imread(processed_frame_path, cv2.IMREAD_UNCHANGED)

                if processed_frame_bgra is None:
                    print(f"Warning: Could not read processed frame {i}")
                    continue

                # --- Apply color despill if enabled ---
                if despill and processed_frame_bgra.shape[2] == 4: # Ensure it has an alpha channel
                    processed_frame_bgra = despill_color_channel(
                        processed_frame_bgra,
                        spill_color_name=despill_color,
                        spill_threshold_factor=despill_threshold,
                        correction_intensity=despill_intensity
                    )
                # --- End of despill ---

                # Handle transparency for output format (using the potentially despilled frame)
                if processed_frame_bgra.shape[2] == 4:
                    alpha = processed_frame_bgra[:, :, 3]
                    bg = np.ones_like(processed_frame_bgra[:, :, :3]) * 255
                    fg = processed_frame_bgra[:, :, :3]

                    blended = np.zeros_like(fg)
                    for c_idx in range(3): # Iterate B, G, R channels
                        blended[:, :, c_idx] = fg[:, :, c_idx] * (alpha / 255.0) + \
                                               bg[:, :, c_idx] * (1 - alpha / 255.0)
                    frame_to_write = blended.astype(np.uint8)
                else:
                    frame_to_write = processed_frame_bgra # Should be BGR if no alpha
                
                out.write(frame_to_write)
                
                # Show progress every 100 frames
                if i % 100 == 0 and i > 0:
                    elapsed = time.time() - start_time
                    frames_per_second = i / elapsed if elapsed > 0 else 0
                    print(f"Encoded {i}/{frame_idx} frames ({frames_per_second:.2f} frames/sec)")
            else:
                print(f"Warning: Processed frame not found: {processed_frame_path}")
        
        out.release()
        print(f"Video creation complete in {time.time() - start_time:.2f} seconds")
        print(f"Output video saved to: {output_video_path}")
        
    return True

def remove_background_video_backgroundremover(
    input_video_path: str,
    output_video_path: str,
    model_name: str = "u2net",
    alpha_matting: bool = False,
    alpha_matting_foreground_threshold: int = 240,
    alpha_matting_background_threshold: int = 10,
    alpha_matting_erode_size: int = 10
):
    """
    Removes background from a video using the backgroundremover CLI.

    Args:
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to save the output video file.
        model_name (str): Model to use ("u2net", "u2net_human_seg", "u2netp").
        alpha_matting (bool): Enable alpha matting.
        alpha_matting_foreground_threshold (int): Foreground threshold for alpha matting.
        alpha_matting_background_threshold (int): Background threshold for alpha matting.
        alpha_matting_erode_size (int): Erode size for alpha matting.

    Returns:
        bool: True if successful, False otherwise.
    """
    if not os.path.exists(input_video_path):
        print(f"Error: Input video not found at {input_video_path}")
        return False

    br_executable = find_executable("backgroundremover")
    if not br_executable:
        print("Error: 'backgroundremover' executable not found. Please ensure it's installed and in your PATH.")
        print("You can install it with: pip install backgroundremover")
        return False
        
    command = [
        br_executable,
        "-i", input_video_path,
        "-o", output_video_path,
        "-m", model_name
    ]

    if alpha_matting:
        command.extend([
            "-a", # Enable alpha matting
            "-afg", str(alpha_matting_foreground_threshold),
            "-abg", str(alpha_matting_background_threshold),
            "-ae", str(alpha_matting_erode_size)
        ])
    
    print(f"Executing backgroundremover command: {' '.join(command)}")
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            print(f"backgroundremover video processing successful. Output saved to {output_video_path}")
            if stdout:
                print("backgroundremover stdout:\n", stdout.decode())
            return True
        else:
            print(f"Error during backgroundremover video processing (return code: {process.returncode}):")
            if stdout:
                print("backgroundremover stdout:\n", stdout.decode())
            if stderr:
                print("backgroundremover stderr:\n", stderr.decode())
            return False
    except FileNotFoundError:
        print(f"Error: backgroundremover command '{br_executable}' not found. Make sure backgroundremover is installed and in your PATH.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred with backgroundremover: {e}")
        return False

if __name__ == '__main__':
    # Basic test (requires dummy video files and executables in PATH)
    print("Running basic tests for video_background_removal.py...")
    
    # Create dummy input video file if it doesn't exist
    dummy_input_video = "dummy_input_video.mp4"
    dummy_rembg_output = "dummy_rembg_output.mp4"
    dummy_br_output = "dummy_br_output.mp4"

    if not os.path.exists(dummy_input_video):
        try:
            # Create a very small, short dummy MP4 video using opencv if possible
            # This requires opencv-python to be installed.
            fourcc_dummy = cv2.VideoWriter_fourcc(*'mp4v')
            dummy_out = cv2.VideoWriter(dummy_input_video, fourcc_dummy, 1, (10, 10))
            if dummy_out.isOpened():
                for _ in range(5): # 5 frames
                    dummy_frame = np.zeros((10, 10, 3), dtype=np.uint8)
                    dummy_out.write(dummy_frame)
                dummy_out.release()
                print(f"Created dummy input video: {dummy_input_video}")
            else:
                print(f"Failed to create dummy input video with OpenCV.")
        except Exception as e:
            print(f"Could not create dummy input video: {e}")

    if os.path.exists(dummy_input_video):
        print("\nTesting rembg video processing...")
        remove_background_video_rembg(dummy_input_video, dummy_rembg_output, model="u2net_lite")
                                      # Using u2net_lite for faster testing if available

        print("\nTesting backgroundremover video processing...")
        remove_background_video_backgroundremover(dummy_input_video, dummy_br_output, model_name="u2netp")
                                                  # Using u2netp for faster testing
    else:
        print(f"Skipping tests as dummy input video '{dummy_input_video}' does not exist.")
