import cv2
import numpy as np
import os
from skimage import io as skimage_io, img_as_float, img_as_ubyte
from skimage.restoration import denoise_tv_chambolle, denoise_nl_means, denoise_wavelet
from skimage.util import random_noise

# Initialize a random number generator for use in dummy image creation
rng = np.random.default_rng(seed=42)  # Added seed for reproducibility

def denoise_image_opencv(
    input_path: str, 
    output_path: str, 
    h_luminance: float = 10.0, 
    h_color: float = 10.0, 
    template_window_size: int = 7, 
    search_window_size: int = 21
):
    """
    Denoises an image using OpenCV's fastNlMeansDenoising algorithms.

    Automatically detects if the image is grayscale or color and uses the
    appropriate denoising function.

    Args:
        input_path: Path to the input image file.
        output_path: Path to save the denoised image file.
        h_luminance: Parameter regulating filter strength for luminance component.
                     Higher h_luminance value removes more noise but also removes 
                     image details. (Recommended: 10).
                     For grayscale images, this is the main strength parameter.
        h_color: Parameter regulating filter strength for color components.
                 (Recommended: 10, same as h_luminance for most cases).
                 Only used for color images.
        template_window_size: Size in pixels of the template patch that is used 
                              to compute weights. Should be odd. (Recommended: 7).
        search_window_size: Size in pixels of the window that is used to compute 
                            weighted average for a given pixel. Should be odd. 
                            Affects performance linearly. (Recommended: 21).

    Returns:
        str: The path to the saved denoised image.

    Raises:
        FileNotFoundError: If the input_path does not point to a valid file.
        ValueError: If template_window_size or search_window_size are not odd.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")

    if template_window_size % 2 == 0:
        raise ValueError("template_window_size must be an odd number.")
    if search_window_size % 2 == 0:
        raise ValueError("search_window_size must be an odd number.")

    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Could not read image from path: {input_path}")

    denoised_img = None
    if len(img.shape) == 2 or img.shape[2] == 1:  # Grayscale image
        print(f"Denoising grayscale image: {input_path}")
        denoised_img = cv2.fastNlMeansDenoising(
            img, 
            None, 
            h_luminance, 
            template_window_size, 
            search_window_size
        )
    elif img.shape[2] == 3:  # Color image without alpha
        print(f"Denoising BGR color image: {input_path}")
        denoised_img = cv2.fastNlMeansDenoisingColored(
            img, 
            None, 
            h_luminance, 
            h_color, 
            template_window_size, 
            search_window_size
        )
    elif img.shape[2] == 4:  # Color image with alpha
        print(f"Denoising BGRA color image (alpha channel will be preserved): {input_path}")
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        
        denoised_bgr = cv2.fastNlMeansDenoisingColored(
            bgr, 
            None, 
            h_luminance, 
            h_color, 
            template_window_size, 
            search_window_size
        )
        denoised_img = cv2.merge((denoised_bgr, alpha))
    else:
        raise ValueError(
            f"Unsupported image format. Expected grayscale, BGR, or BGRA. Got shape: {img.shape}"
        )

    cv2.imwrite(output_path, denoised_img)
    print(f"Denoised image saved to: {output_path}")
    return output_path

def denoise_tv_chambolle_skimage(
    input_path: str,
    output_path: str,
    weight: float = 0.1,
    multichannel: bool = True
):
    """
    Denoises an image using Total Variation (TV) Chambolle algorithm from skimage.

    Args:
        input_path: Path to the input image file.
        output_path: Path to save the denoised image file.
        weight: Denoising weight. The greater the weight, the more denoising.
        multichannel: If True, denoise each channel of a color image separately.
                      If False, convert color image to grayscale before denoising.
                      Alpha channel is preserved if present and multichannel is True.

    Returns:
        str: The path to the saved denoised image.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")

    img = skimage_io.imread(input_path)
    float_img = img_as_float(img)
    
    denoised_img_float = None
    original_alpha = None

    if float_img.ndim == 3 and float_img.shape[2] == 4:  # RGBA
        print(f"Denoising RGBA image with TV Chambolle (skimage): {input_path}")
        rgb_part = float_img[:, :, :3]
        original_alpha = float_img[:, :, 3]
        if multichannel:
            denoised_rgb_float = denoise_tv_chambolle(rgb_part, weight=weight, channel_axis=-1)
            denoised_img_float = np.dstack((denoised_rgb_float, original_alpha))
        else:
            from skimage.color import rgb2gray
            gray_part = rgb2gray(rgb_part)
            denoised_gray_float = denoise_tv_chambolle(gray_part, weight=weight, channel_axis=None)
            denoised_rgb_float = np.stack([denoised_gray_float] * 3, axis=-1)
            denoised_img_float = np.dstack((denoised_rgb_float, original_alpha))
    elif float_img.ndim == 3 and float_img.shape[2] == 3:  # RGB
        print(f"Denoising RGB image with TV Chambolle (skimage): {input_path}")
        denoised_img_float = denoise_tv_chambolle(float_img, weight=weight, channel_axis=-1 if multichannel else None)
    elif float_img.ndim == 2:  # Grayscale
        print(f"Denoising Grayscale image with TV Chambolle (skimage): {input_path}")
        denoised_img_float = denoise_tv_chambolle(float_img, weight=weight, channel_axis=None)
    else:
        raise ValueError(f"Unsupported image format. Shape: {img.shape}")

    denoised_img_ubyte = img_as_ubyte(np.clip(denoised_img_float, 0, 1))
    skimage_io.imsave(output_path, denoised_img_ubyte)
    print(f"TV Chambolle (skimage) denoised image saved to: {output_path}")
    return output_path

def denoise_nl_means_skimage(
    input_path: str,
    output_path: str,
    patch_size: int = 7,
    patch_distance: int = 11,
    h_parameter: float = 0.08,
    multichannel: bool = True,
    fast_mode: bool = True
):
    """
    Denoises an image using Non-Local Means (NL-Means) algorithm from skimage.

    Args:
        input_path: Path to the input image file.
        output_path: Path to save the denoised image file.
        patch_size: Size of patches used for comparison.
        patch_distance: Maximal distance in pixels where to search patches.
        h_parameter: Cut-off distance (in intensity) for patches similarity.
        multichannel: If True, denoise each channel of a color image separately.
        fast_mode: If True, use a faster version of the algorithm.
                   Alpha channel is preserved if present and multichannel is True.

    Returns:
        str: The path to the saved denoised image.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")

    img = skimage_io.imread(input_path)
    float_img = img_as_float(img)
    
    denoised_img_float = None
    original_alpha = None

    if float_img.ndim == 3 and float_img.shape[2] == 4:  # RGBA
        print(f"Denoising RGBA image with NL-Means (skimage): {input_path}")
        rgb_part = float_img[:, :, :3]
        original_alpha = float_img[:, :, 3]
        if multichannel:
            denoised_rgb_float = denoise_nl_means(
                rgb_part, 
                h=h_parameter, 
                patch_size=patch_size, 
                patch_distance=patch_distance, 
                channel_axis=-1, 
                fast_mode=fast_mode
            )
            denoised_img_float = np.dstack((denoised_rgb_float, original_alpha))
        else:
            from skimage.color import rgb2gray
            gray_part = rgb2gray(rgb_part)
            denoised_gray_float = denoise_nl_means(
                gray_part, h=h_parameter, patch_size=patch_size, patch_distance=patch_distance, channel_axis=None, fast_mode=fast_mode
            )
            denoised_rgb_float = np.stack([denoised_gray_float] * 3, axis=-1)
            denoised_img_float = np.dstack((denoised_rgb_float, original_alpha))
    elif float_img.ndim == 3 and float_img.shape[2] == 3:  # RGB
        print(f"Denoising RGB image with NL-Means (skimage): {input_path}")
        denoised_img_float = denoise_nl_means(
            float_img, 
            h=h_parameter, 
            patch_size=patch_size, 
            patch_distance=patch_distance, 
            channel_axis=-1 if multichannel else None, 
            fast_mode=fast_mode
        )
    elif float_img.ndim == 2:  # Grayscale
        print(f"Denoising Grayscale image with NL-Means (skimage): {input_path}")
        denoised_img_float = denoise_nl_means(
            float_img, h=h_parameter, patch_size=patch_size, patch_distance=patch_distance, channel_axis=None, fast_mode=fast_mode
        )
    else:
        raise ValueError(f"Unsupported image format. Shape: {img.shape}")

    denoised_img_ubyte = img_as_ubyte(np.clip(denoised_img_float, 0, 1))
    skimage_io.imsave(output_path, denoised_img_ubyte)
    print(f"NL-Means (skimage) denoised image saved to: {output_path}")
    return output_path

def denoise_wavelet_skimage(
    input_path: str,
    output_path: str,
    method: str = 'BayesShrink',
    mode: str = 'soft',
    wavelet_levels: int = None,
    wavelet: str = 'db1',
    multichannel: bool = True,
    rescale_sigma: bool = True
):
    """
    Denoises an image using Wavelet denoising from skimage.

    Args:
        input_path: Path to the input image file.
        output_path: Path to save the denoised image file.
        method: Thresholding method ('BayesShrink' or 'VisuShrink').
        mode: Thresholding mode ('soft' or 'hard').
        wavelet_levels: Number of wavelet decomposition levels.
        wavelet: Type of wavelet to use (e.g., 'db1', 'sym2').
        multichannel: If True, denoise each channel of a color image separately.
        rescale_sigma: If True, rescale image intensities to have noise std dev of 1.
                       Alpha channel is preserved if present and multichannel is True.

    Returns:
        str: The path to the saved denoised image.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")

    img = skimage_io.imread(input_path)
    float_img = img_as_float(img)

    denoised_img_float = None
    original_alpha = None

    if float_img.ndim == 3 and float_img.shape[2] == 4:  # RGBA
        print(f"Denoising RGBA image with Wavelet (skimage): {input_path}")
        rgb_part = float_img[:, :, :3]
        original_alpha = float_img[:, :, 3]
        if multichannel:
            denoised_rgb_float = denoise_wavelet(
                rgb_part, 
                channel_axis=-1, 
                method=method, 
                mode=mode, 
                wavelet_levels=wavelet_levels, 
                wavelet=wavelet, 
                rescale_sigma=rescale_sigma
            )
            denoised_img_float = np.dstack((denoised_rgb_float, original_alpha))
        else:
            from skimage.color import rgb2gray
            gray_part = rgb2gray(rgb_part)
            denoised_gray_float = denoise_wavelet(
                gray_part, channel_axis=None, method=method, mode=mode, wavelet_levels=wavelet_levels, wavelet=wavelet, rescale_sigma=rescale_sigma
            )
            denoised_rgb_float = np.stack([denoised_gray_float] * 3, axis=-1)
            denoised_img_float = np.dstack((denoised_rgb_float, original_alpha))
    elif float_img.ndim == 3 and float_img.shape[2] == 3:  # RGB
        print(f"Denoising RGB image with Wavelet (skimage): {input_path}")
        denoised_img_float = denoise_wavelet(
            float_img, 
            channel_axis=-1 if multichannel else None, 
            method=method, 
            mode=mode, 
            wavelet_levels=wavelet_levels, 
            wavelet=wavelet, 
            rescale_sigma=rescale_sigma
        )
    elif float_img.ndim == 2:  # Grayscale
        print(f"Denoising Grayscale image with Wavelet (skimage): {input_path}")
        denoised_img_float = denoise_wavelet(
            float_img, channel_axis=None, method=method, mode=mode, wavelet_levels=wavelet_levels, wavelet=wavelet, rescale_sigma=rescale_sigma
        )
    else:
        raise ValueError(f"Unsupported image format. Shape: {img.shape}")

    denoised_img_ubyte = img_as_ubyte(np.clip(denoised_img_float, 0, 1))
    skimage_io.imsave(output_path, denoised_img_ubyte)
    print(f"Wavelet (skimage) denoised image saved to: {output_path}")
    return output_path

def inpaint_image_opencv(
    input_path: str,
    mask_path: str,
    output_path: str,
    radius: float = 3.0,
    method_flag: int = cv2.INPAINT_TELEA
):
    """
    Inpaints an image using OpenCV's inpainting algorithms.

    Args:
        input_path: Path to the input image file (color or grayscale).
        mask_path: Path to the mask file (grayscale, 8-bit). 
                   Non-zero pixels indicate areas to be inpainted.
        output_path: Path to save the inpainted image file.
        radius: Radius of a circular neighborhood of each point inpainted.
        method_flag: Inpainting method. Can be cv2.INPAINT_TELEA or cv2.INPAINT_NS.
                     cv2.INPAINT_TELEA is generally faster and gives smooth results.
                     cv2.INPAINT_NS (Navier-Stokes) can be better for thin structures.

    Returns:
        str: The path to the saved inpainted image.

    Raises:
        FileNotFoundError: If input_path or mask_path does not point to a valid file.
        IOError: If images cannot be read.
        ValueError: If mask is not single-channel or input image format is unsupported.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask image not found: {mask_path}")

    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Could not read image from path: {input_path}")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise IOError(f"Could not read mask from path: {mask_path}")

    # Ensure mask is 8-bit single channel
    if len(mask.shape) != 2:
        raise ValueError("Mask must be a single-channel grayscale image.")
    if mask.dtype != np.uint8:
        original_mask_dtype = mask.dtype
        # Attempt to convert common mask representations (e.g., boolean, float 0-1)
        if np.issubdtype(mask.dtype, np.floating) and mask.max() <= 1.0 and mask.min() >= 0.0:
            print(f"Converting float mask (max: {mask.max()}) to uint8.")
            mask = (mask * 255).astype(np.uint8)
        elif np.issubdtype(mask.dtype, bool):
            print("Converting boolean mask to uint8.")
            mask = (mask * 255).astype(np.uint8)
        else:
            # For other types, try a direct conversion, clipping if necessary
            print(f"Attempting to convert mask of type {original_mask_dtype} to uint8.")
            if np.issubdtype(mask.dtype, np.integer) and mask.max() > 255:
                 mask = np.clip(mask, 0, 255).astype(np.uint8)
            else:
                 mask = mask.astype(np.uint8) # Fallback, might not be ideal for all types

        print(f"Converted mask to uint8. Original dtype: {original_mask_dtype}, New dtype: {mask.dtype}")


    original_alpha = None
    img_to_inpaint = None

    if len(img.shape) == 3 and img.shape[2] == 4:  # BGRA
        print(f"Inpainting BGRA image: {input_path}")
        original_alpha = img[:, :, 3]
        img_to_inpaint = img[:, :, :3]  # Work with BGR
    elif len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):  # Grayscale
        print(f"Inpainting grayscale image: {input_path}")
        img_to_inpaint = img
        if img_to_inpaint.ndim == 3 and img_to_inpaint.shape[2] == 1: # Convert (H, W, 1) to (H, W)
            img_to_inpaint = img_to_inpaint.reshape(img_to_inpaint.shape[0], img_to_inpaint.shape[1])
    elif len(img.shape) == 3 and img.shape[2] == 3:  # BGR
        print(f"Inpainting BGR color image: {input_path}")
        img_to_inpaint = img
    else:
        raise ValueError(f"Unsupported input image format. Shape: {img.shape}")

    # Ensure img_to_inpaint and mask have the same dimensions (H, W)
    if img_to_inpaint.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Image and mask dimensions do not match. "
            f"Image: {img_to_inpaint.shape[:2]}, Mask: {mask.shape[:2]}"
        )

    inpainted_img_base = cv2.inpaint(img_to_inpaint, mask, radius, method_flag)

    final_output_img = None
    if original_alpha is not None and inpainted_img_base.ndim == 3: # Re-attach alpha if it was BGRA
        final_output_img = cv2.merge((inpainted_img_base, original_alpha))
    else:
        final_output_img = inpainted_img_base
        
    cv2.imwrite(output_path, final_output_img)
    print(f"Inpainted image saved to: {output_path}")
    return output_path

if __name__ == '__main__':
    # Create dummy images for testing
    def create_dummy_image(filename, channels, add_noise=True, size=(100, 100)):
        h, w = size
        if channels == 1:
            img_data = np.full((h, w), 128, dtype=np.uint8)
        elif channels == 3:
            img_data = np.full((h, w, 3), (128, 128, 128), dtype=np.uint8)
        elif channels == 4:
            img_data = np.full((h, w, 4), (128, 128, 128, 255), dtype=np.uint8)
        else:
            print(f"Unsupported channel count: {channels} for dummy image.")
            return

        if add_noise:
            noise = rng.integers(0, 50, img_data.shape, dtype=np.uint8, endpoint=False)
            img_data = cv2.add(img_data, noise)
            noise2 = rng.integers(0, 50, img_data.shape, dtype=np.uint8, endpoint=False)
            img_data = cv2.subtract(img_data, noise2)
            img_data = np.clip(img_data, 0, 255)

        cv2.imwrite(filename, img_data)
        print(f"Created dummy image: {filename}")

    test_dir = "denoise_test_images"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    dummy_gray = os.path.join(test_dir, "dummy_gray_noisy.png")
    dummy_color = os.path.join(test_dir, "dummy_color_noisy.png")
    dummy_color_alpha = os.path.join(test_dir, "dummy_color_alpha_noisy.png")

    create_dummy_image(dummy_gray, 1)
    create_dummy_image(dummy_color, 3)
    create_dummy_image(dummy_color_alpha, 4)

    dummy_gray_sk = os.path.join(test_dir, "dummy_gray_noisy_sk.png")
    dummy_color_sk = os.path.join(test_dir, "dummy_color_noisy_sk.png")
    dummy_color_alpha_sk = os.path.join(test_dir, "dummy_color_alpha_noisy_sk.png")
    
    if os.path.exists(dummy_gray): skimage_io.imsave(dummy_gray_sk, skimage_io.imread(dummy_gray))
    else: create_dummy_image(dummy_gray_sk, 1)
        
    if os.path.exists(dummy_color): skimage_io.imsave(dummy_color_sk, skimage_io.imread(dummy_color))
    else: create_dummy_image(dummy_color_sk, 3)

    if os.path.exists(dummy_color_alpha): skimage_io.imsave(dummy_color_alpha_sk, skimage_io.imread(dummy_color_alpha))
    else: create_dummy_image(dummy_color_alpha_sk, 4)

    print("\\n--- Testing denoise_image_opencv ---")
    
    output_gray = os.path.join(test_dir, "dummy_gray_denoised.png")
    try:
        denoise_image_opencv(dummy_gray, output_gray, h_luminance=15)
    except Exception as e:
        print(f"Error denoising grayscale: {e}")

    output_color = os.path.join(test_dir, "dummy_color_denoised.png")
    try:
        denoise_image_opencv(dummy_color, output_color, h_luminance=12, h_color=12)
    except Exception as e:
        print(f"Error denoising color: {e}")

    output_color_alpha = os.path.join(test_dir, "dummy_color_alpha_denoised.png")
    try:
        denoise_image_opencv(dummy_color_alpha, output_color_alpha, h_luminance=10, h_color=10)
    except Exception as e:
        print(f"Error denoising color with alpha: {e}")
        
    print("\\n--- Testing denoise_tv_chambolle_skimage ---")
    output_gray_tv = os.path.join(test_dir, "dummy_gray_denoised_tv_sk.png")
    try:
        if os.path.exists(dummy_gray_sk): denoise_tv_chambolle_skimage(dummy_gray_sk, output_gray_tv, weight=0.05)
    except Exception as e:
        print(f"Error denoising grayscale with TV (skimage): {e}")

    output_color_tv = os.path.join(test_dir, "dummy_color_denoised_tv_sk.png")
    try:
        if os.path.exists(dummy_color_sk): denoise_tv_chambolle_skimage(dummy_color_sk, output_color_tv, weight=0.05)
    except Exception as e:
        print(f"Error denoising color with TV (skimage): {e}")
    
    output_color_alpha_tv = os.path.join(test_dir, "dummy_color_alpha_denoised_tv_sk.png")
    try:
        if os.path.exists(dummy_color_alpha_sk): denoise_tv_chambolle_skimage(dummy_color_alpha_sk, output_color_alpha_tv, weight=0.05)
    except Exception as e:
        print(f"Error denoising color alpha with TV (skimage): {e}")

    print("\\n--- Testing denoise_nl_means_skimage ---")
    output_gray_nlm = os.path.join(test_dir, "dummy_gray_denoised_nlm_sk.png")
    try:
        if os.path.exists(dummy_gray_sk): denoise_nl_means_skimage(dummy_gray_sk, output_gray_nlm, h_parameter=0.05)
    except Exception as e:
        print(f"Error denoising grayscale with NL-Means (skimage): {e}")

    output_color_nlm = os.path.join(test_dir, "dummy_color_denoised_nlm_sk.png")
    try:
        if os.path.exists(dummy_color_sk): denoise_nl_means_skimage(dummy_color_sk, output_color_nlm, h_parameter=0.05)
    except Exception as e:
        print(f"Error denoising color with NL-Means (skimage): {e}")

    output_color_alpha_nlm = os.path.join(test_dir, "dummy_color_alpha_denoised_nlm_sk.png")
    try:
        if os.path.exists(dummy_color_alpha_sk): denoise_nl_means_skimage(dummy_color_alpha_sk, output_color_alpha_nlm, h_parameter=0.05)
    except Exception as e:
        print(f"Error denoising color alpha with NL-Means (skimage): {e}")

    print("\\n--- Testing denoise_wavelet_skimage ---")
    output_gray_wavelet = os.path.join(test_dir, "dummy_gray_denoised_wavelet_sk.png")
    try:
        if os.path.exists(dummy_gray_sk): denoise_wavelet_skimage(dummy_gray_sk, output_gray_wavelet)
    except Exception as e:
        print(f"Error denoising grayscale with Wavelet (skimage): {e}")

    output_color_wavelet = os.path.join(test_dir, "dummy_color_denoised_wavelet_sk.png")
    try:
        if os.path.exists(dummy_color_sk): denoise_wavelet_skimage(dummy_color_sk, output_color_wavelet)
    except Exception as e:
        print(f"Error denoising color with Wavelet (skimage): {e}")
        
    output_color_alpha_wavelet = os.path.join(test_dir, "dummy_color_alpha_denoised_wavelet_sk.png")
    try:
        if os.path.exists(dummy_color_alpha_sk): denoise_wavelet_skimage(dummy_color_alpha_sk, output_color_alpha_wavelet)
    except Exception as e:
        print(f"Error denoising color alpha with Wavelet (skimage): {e}")

    print("\\n--- Testing inpaint_image_opencv ---")

    dummy_inpaint_src_fn = "dummy_inpaint_src.png"
    dummy_mask_fn = "dummy_mask.png"
    dummy_inpaint_src = os.path.join(test_dir, dummy_inpaint_src_fn)
    dummy_mask = os.path.join(test_dir, dummy_mask_fn)
    
    create_dummy_image(dummy_inpaint_src, 3, add_noise=False)

    mask_img_data = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask_img_data, (30, 30), (70, 70), 255, -1) 
    cv2.imwrite(dummy_mask, mask_img_data)
    print(f"Created dummy mask: {dummy_mask}")
    
    src_img_for_inpaint_data = cv2.imread(dummy_inpaint_src)
    if src_img_for_inpaint_data is not None:
        damaged_src_for_inpaint = src_img_for_inpaint_data.copy()
        cv2.rectangle(damaged_src_for_inpaint, (30, 30), (70, 70), (0,0,255), -1) 
        cv2.imwrite(dummy_inpaint_src, damaged_src_for_inpaint) 
        print(f"Created dummy damaged source for inpainting: {dummy_inpaint_src}")
    else:
        print(f"Failed to read {dummy_inpaint_src} to create damaged version.")

    output_inpaint_telea = os.path.join(test_dir, "dummy_inpainted_telea.png")
    try:
        inpaint_image_opencv(dummy_inpaint_src, dummy_mask, output_inpaint_telea, radius=5, method_flag=cv2.INPAINT_TELEA)
    except Exception as e:
        print(f"Error inpainting with TELEA: {e}")

    src_img_for_inpaint_data_clean = cv2.imread(os.path.join(test_dir, "dummy_inpaint_src.png"))
    if src_img_for_inpaint_data_clean is not None:
        cv2.imwrite(os.path.join(test_dir, "dummy_inpaint_src_clean_temp.png"), src_img_for_inpaint_data_clean)
        create_dummy_image(dummy_inpaint_src, 3, add_noise=False)
        src_img_for_inpaint_data = cv2.imread(dummy_inpaint_src)
        if src_img_for_inpaint_data is not None:
            damaged_src_for_inpaint = src_img_for_inpaint_data.copy()
            cv2.rectangle(damaged_src_for_inpaint, (30, 30), (70, 70), (0,0,255), -1) 
            cv2.imwrite(dummy_inpaint_src, damaged_src_for_inpaint) 
    
    output_inpaint_ns = os.path.join(test_dir, "dummy_inpainted_ns.png")
    try:
        inpaint_image_opencv(dummy_inpaint_src, dummy_mask, output_inpaint_ns, radius=5, method_flag=cv2.INPAINT_NS)
    except Exception as e:
        print(f"Error inpainting with NS: {e}")
        
    dummy_inpaint_src_alpha_fn = "dummy_inpaint_src_alpha.png"
    dummy_inpaint_src_alpha = os.path.join(test_dir, dummy_inpaint_src_alpha_fn)
    create_dummy_image(dummy_inpaint_src_alpha, 4, add_noise=False)
    
    src_img_alpha_for_inpaint_data = cv2.imread(dummy_inpaint_src_alpha, cv2.IMREAD_UNCHANGED)
    if src_img_alpha_for_inpaint_data is not None:
        bgr_part = src_img_alpha_for_inpaint_data[:,:,:3].copy()
        cv2.rectangle(bgr_part, (30,30), (70,70), (0,255,0), -1) 
        src_img_alpha_for_inpaint_data[:,:,:3] = bgr_part
        cv2.imwrite(dummy_inpaint_src_alpha, src_img_alpha_for_inpaint_data)
        print(f"Created dummy damaged BGRA source for inpainting: {dummy_inpaint_src_alpha}")
    else:
        print(f"Failed to read {dummy_inpaint_src_alpha} to create damaged BGRA version.")

    output_inpaint_alpha = os.path.join(test_dir, "dummy_inpainted_alpha.png")
    try:
        inpaint_image_opencv(dummy_inpaint_src_alpha, dummy_mask, output_inpaint_alpha, radius=5, method_flag=cv2.INPAINT_TELEA)
    except Exception as e:
        print(f"Error inpainting BGRA image: {e}")

    print("\\nTo see results, check the 'denoise_test_images' directory.")
