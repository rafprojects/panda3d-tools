# filepath: /home/user/projects/py_projects/pandas_learn/scripts/image_upscale.py
import cv2
import numpy as np
import os
import re
import torch

# Add optional imports for upscalers
try:
    from realesrgan import RealESRGANer
    HAS_REALESRGAN = True
except ImportError:
    HAS_REALESRGAN = False

# try:
#     import super_image
#     HAS_SUPER_IMAGE = True
# except ImportError:
#     HAS_SUPER_IMAGE = False
import super_image

def upscale_with_cv2(image_path, output_path, scale_factor=2, method=cv2.INTER_NEAREST):
    """Upscale an image using OpenCV. Works for any image type.
    Args:
        image_path: Path to the source image
        output_path: Path to save the upscaled image
        scale_factor: Scaling factor (2, 3, 4, etc.)
        method: OpenCV interpolation method
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    h, w = img.shape[:2]
    new_h, new_w = h * scale_factor, w * scale_factor
    upscaled = cv2.resize(img, (new_w, new_h), interpolation=method)
    cv2.imwrite(output_path, upscaled)
    return upscaled.shape[:2]

def pixel_art_upscale(image_path, output_path, scale_factor=2, edge_sharpness=1.0):
    """Upscale pixel art or regular images, preserving sharp edges if desired.
    Args:
        image_path: Path to the source image
        output_path: Path to save the upscaled image
        scale_factor: Scaling factor (2, 3, 4, etc.)
        edge_sharpness: Higher values preserve sharper pixel edges
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    h, w = img.shape[:2]
    new_h, new_w = h * scale_factor, w * scale_factor
    upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    if edge_sharpness < 1.0:
        smooth = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        alpha = edge_sharpness
        beta = 1.0 - alpha
        upscaled = cv2.addWeighted(upscaled, alpha, smooth, beta, 0)
    elif edge_sharpness > 1.0:
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]]) * (edge_sharpness - 1.0) / 8.0 + np.eye(3)
        upscaled = cv2.filter2D(upscaled, -1, kernel)
    cv2.imwrite(output_path, upscaled)
    return upscaled.shape[:2]

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    if lv == 6:
        return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))  # BGR order
    raise ValueError('Hex color must be 6 digits')

def clean_background(image_path, output_path, mode='white', color_hex='#FFFFFF', threshold=220):
    """Clean background around text/logo, replacing with transparency, white, or a color.
    Args:
        image_path: Path to input image
        output_path: Path to save cleaned image
        mode: 'transparent', 'white', or 'color'
        color_hex: Hex color string for custom color
        threshold: Intensity threshold for background detection
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    if img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
    else:
        bgr = img
        alpha = None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > threshold).astype(np.uint8) * 255
    mask_inv = cv2.bitwise_not(mask)
    if mode == 'transparent':
        if alpha is None:
            alpha = np.ones_like(gray, dtype=np.uint8) * 255
        alpha = cv2.bitwise_and(alpha, mask_inv)
        result = np.dstack([bgr, alpha])
    elif mode == 'white':
        white = np.ones_like(bgr, dtype=np.uint8) * 255
        result = cv2.bitwise_and(bgr, bgr, mask=mask_inv)
        result += cv2.bitwise_and(white, white, mask=mask)
    elif mode == 'color':
        color = np.array(hex_to_bgr(color_hex), dtype=np.uint8)
        color_img = np.ones_like(bgr, dtype=np.uint8) * color
        result = cv2.bitwise_and(bgr, bgr, mask=mask_inv)
        result += cv2.bitwise_and(color_img, color_img, mask=mask)
    else:
        raise ValueError('Unknown mode for clean_background')
    cv2.imwrite(output_path, result)
    return output_path

def ai_upscale_realesrgan(image_path, output_path, scale=2, model_name='RealESRGAN_x4plus'):
    """
    AI upscaling using Real-ESRGAN.
    Args:
        image_path: Path to input image
        output_path: Path to save upscaled image
        scale: Upscale factor (2, 4, etc.)
        model_name: Model to use ('RealESRGAN_x4plus', 'RealESRGAN_x2plus', etc.)
    """
    import cv2
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    # device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_path = 'models/weights/RealESRGAN_x4plus.pth'
    model = RealESRGANer(scale=scale, device=device, model_path=model_path)
    # model.load_weights(model_path)
    upscaled = model.predict(img)
    cv2.imwrite(output_path, upscaled)
    return upscaled.shape[:2]

def ai_upscale_superimage(image_path, output_path, scale=2, model_name='edsr-base', denoise_level=0.0, sharpen_level=0.0):
    """
    AI upscaling using super-image package's models.
    Args:
        image_path: Path to input image
        output_path: Path to save upscaled image
        scale: Upscale factor (2, 3, 4)
        model_name: Model to use ('edsr-base', 'edsr', 'mdsr', 'a2n', 'han', 'real-world-sr', etc.)
        denoise_level: Amount of denoising to apply (0.0 to 1.0)
        sharpen_level: Amount of sharpening to apply after upscaling (0.0 to 1.0)
    """
    # if not HAS_SUPER_IMAGE:
    #     raise ImportError("super-image package is not installed. Please install it with 'pip install super-image'")
    
    from PIL import Image
    import numpy as np
    
    # Check if scale is valid
    if scale not in [2, 3, 4]:
        raise ValueError(f"Scale factor {scale} not supported by super-image. Use 2, 3, or 4.")
    
    # Load image
    image = Image.open(image_path)
    
    # Save alpha channel if present
    has_alpha = image.mode == 'RGBA'
    if has_alpha:
        alpha_channel = image.split()[3]
        image = image.convert('RGB')
    
    # Apply denoising if requested
    if denoise_level > 0.0:
        # Convert to OpenCV for denoising
        import cv2
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h_param = min(20 * denoise_level, 20)  # Filter strength
        denoised = cv2.fastNlMeansDenoisingColored(cv_img, None, h_param, h_param, 7, 21)
        image = Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
    
    # Select the model
    if model_name == 'edsr-base':
        model = super_image.EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=scale)
    elif model_name == 'edsr':
        model = super_image.EdsrModel.from_pretrained('eugenesiow/edsr', scale=scale)
    elif model_name == 'mdsr':
        model = super_image.MdsrModel.from_pretrained('eugenesiow/mdsr', scale=scale)
    elif model_name == 'a2n':
        model = super_image.A2nModel.from_pretrained('eugenesiow/a2n', scale=scale)
    elif model_name == 'han':
        model = super_image.HanModel.from_pretrained('eugenesiow/han', scale=scale)
    elif model_name == 'real-world-sr':
        model = super_image.EdsrModel.from_pretrained('eugenesiow/real-world-sr', scale=scale)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Process the image using the simplified API
    # inputs = super_image.ImageLoader.load_image(image)
    img_array = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
    img_array = img_array.transpose(2, 0, 1)               # [H, W, C] -> [C, H, W]
    inputs = torch.from_numpy(img_array).unsqueeze(0)
    
    preds = model(inputs)
    # output_img = super_image.ImageLoader.tensor_to_image(preds)
    preds = preds.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()  # [C, H, W] -> [H, W, C]
    preds = np.clip(preds * 255.0, 0, 255).astype(np.uint8)   # Scale and clip
    output_img = Image.fromarray(preds)
    
    # Apply sharpening if requested
    if sharpen_level > 0.0:
        import cv2
        img_array = np.array(output_img)
        bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Create sharpening kernel
        kernel_strength = min(1.0, sharpen_level) * 0.8
        kernel = np.array([[-kernel_strength, -kernel_strength, -kernel_strength],
                          [-kernel_strength, 1 + 8 * kernel_strength, -kernel_strength],
                          [-kernel_strength, -kernel_strength, -kernel_strength]])
        
        # Apply sharpening
        sharpened = cv2.filter2D(bgr, -1, kernel)
        output_img = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    
    # Handle alpha channel if present
    if has_alpha:
        # Upscale alpha channel with bicubic interpolation
        alpha_size = (output_img.width, output_img.height)
        alpha_upscaled = alpha_channel.resize(alpha_size, Image.BICUBIC)
        output_img.putalpha(alpha_upscaled)
    
    # Save the output
    output_img.save(output_path)
    
    return output_img.height, output_img.width
