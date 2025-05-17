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

def remove_background_video_rembg(
    input_video_path: str,
    output_video_path: str,
    model: str = "u2net",
    extra_args: list = None,
    max_workers: int = 4,
    quality: int = 95
):
    """
    Removes background from a video using rembg in a frame-by-frame approach.

    Args:
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to save the output video file.
        model (str): Model to use (e.g., "u2net", "u2net_human_seg").
        extra_args (list, optional): List of additional arguments to pass to rembg CLI for each frame.
        max_workers (int): Maximum number of parallel workers for frame processing.
        quality (int): JPEG quality for temporary frame storage (1-100).

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
                processed_frame = cv2.imread(processed_frame_path, cv2.IMREAD_UNCHANGED)
                
                # Handle transparency for output format
                if processed_frame is None:
                    print(f"Warning: Could not read processed frame {i}")
                    continue
                    
                # If the processed image has an alpha channel (4 channels)
                if processed_frame.shape[2] == 4:
                    # Extract alpha channel
                    alpha = processed_frame[:, :, 3]
                    
                    # Create a white background
                    bg = np.ones_like(processed_frame[:, :, :3]) * 255
                    
                    # Calculate foreground based on alpha
                    fg = processed_frame[:, :, :3]
                    
                    # Blend foreground and background
                    blended = np.zeros_like(fg)
                    for c in range(3):
                        blended[:, :, c] = fg[:, :, c] * (alpha / 255.0) + bg[:, :, c] * (1 - alpha / 255.0)
                        
                    # Convert to valid CV format
                    frame_to_write = blended.astype(np.uint8)
                else:
                    # Just use the BGR image if no alpha channel
                    frame_to_write = processed_frame
                    
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
            # Create a very short, small dummy mp4 file using ffmpeg if available
            ffmpeg_exe = find_executable("ffmpeg")
            if ffmpeg_exe:
                subprocess.run([
                    ffmpeg_exe, "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=128x72:rate=10",
                    "-c:v", "libx264", "-t", "1", dummy_input_video
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"Created dummy video: {dummy_input_video}")
            else:
                print(f"ffmpeg not found. Cannot create dummy video. Please create '{dummy_input_video}' manually for testing.")
        except Exception as e:
            print(f"Could not create dummy video '{dummy_input_video}': {e}. Please create it manually for testing.")

    if os.path.exists(dummy_input_video):
        print("\nTesting rembg video processing...")
        remove_background_video_rembg(dummy_input_video, dummy_rembg_output, model="u2net_lite")
                                      # Using u2net_lite for faster testing if available

        print("\nTesting backgroundremover video processing...")
        remove_background_video_backgroundremover(dummy_input_video, dummy_br_output, model_name="u2netp")
                                                  # Using u2netp for faster testing
    else:
        print(f"Skipping tests as dummy input video '{dummy_input_video}' does not exist.")
