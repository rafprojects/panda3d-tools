# Image Upscaling Tool

A versatile command-line tool for upscaling images with various methods, from simple pixel-art friendly resizing to AI-powered super-resolution.

## Features

- **Basic Upscaling**: Nearest, bilinear, bicubic, and Lanczos interpolation
- **Pixel Art Upscaling**: Preserves sharp edges ideal for pixel art
- **AI Upscaling Options**:
  - **RealESRGAN**: High-quality upscaling with pre-trained models
  - **SuperImage**: Multiple models with enhanced features:
    - Denoising before upscaling to reduce artifacts
    - Sharpening after upscaling to enhance details
    - Support for various models (edsr, mdsr, a2n, han, real-world-sr)
- **Background Cleaning**: Remove or replace image backgrounds
- **Batch Processing**: Process multiple images in one command

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- Torch (for AI upscaling)
- RealESRGAN (optional)
- Super-Image (optional)

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install opencv-python numpy torch
   pip install realesrgan  # Optional, for RealESRGAN support
   pip install super-image  # Optional, for SuperImage support
   ```

## Usage

### Basic Upscaling

```bash
python scripts/image_upscale_cli.py -i input.png -o output.png -m bilinear -s 4
```

### Pixel Art Upscaling

```bash
python scripts/image_upscale_cli.py -i pixel_art.png -o upscaled.png -m pixel_art -s 4 --edge-sharpness 1.5
```

### AI Upscaling with SuperImage (New Features)

```bash
# Basic SuperImage upscaling - simplified implementation
python scripts/image_upscale_cli.py -i image.png -o upscaled.png -m superimage -s 4 --superimage-model edsr-base

# With denoising (reduces noise before upscaling)
python scripts/image_upscale_cli.py -i noisy_image.png -o upscaled.png -m superimage -s 4 --denoise 0.5

# With sharpening (enhances details after upscaling)
python scripts/image_upscale_cli.py -i image.png -o upscaled.png -m superimage -s 4 --sharpen 0.7

# Try the minimal implementation (follows GitHub example exactly)
python super_image_minimal.py input.png 4
```

### SuperImage Demo

Try the demo script to generate multiple versions for comparison:

```bash
python superimage_demo.py -i your_image.png -s 4
```

### RealESRGAN Upscaling

```bash
python scripts/image_upscale_cli.py -i image.png -o upscaled.png -m realesrgan -s 4
```

### Batch Processing

```bash
python scripts/image_upscale_cli.py -i input_folder/ -o output_folder/ -m superimage -s 4 --batch
```

## Benchmarks

Compare the quality and performance of different upscaling methods:

| Method | Speed | Quality for Pixel Art | Quality for Photos |
|--------|-------|------------------------|-------------------|
| Nearest | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
| Bicubic | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Pixel Art | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| SuperImage | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| RealESRGAN | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## License

MIT
