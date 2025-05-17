import cv2
import numpy as np
import os
import pandas as pd
from PIL import Image


def detect_sprites(spritesheet_file, output_file, background_color):
    """Uses opencv to detect sprites in a spritesheet image (preferably PNG)
    \nBackground color must be specified as a tuple of (b, g, r) values, and is 
    used to remove the background from the sprites.
    \nOutputs a CSV file with the coordinates and dimensions of each sprite."""
    # imread() returns a 3D array if the image has an alpha channel
    # cv2.IMREAD_UNCHANGED returns the image as is
    image = cv2.imread(spritesheet_file, cv2.IMREAD_UNCHANGED)
    # np.array() converts the background color to an array
    bg_color = np.array(background_color)
    # Create a mask for the background color
    mask = cv2.inRange(image, bg_color, bg_color)
    # Invert the mask because we want to keep the sprites
    mask_inv = cv2.bitwise_not(mask)
    # Apply the mask to the image in order to remove the background
    image_masked = cv2.bitwise_and(image, image, mask=mask_inv)
    # findContours() returns a list of contours and a hierarchy
    contours, _ = cv2.findContours(
        # findContours() modifies the image, so we pass in a copy
        # RETR_EXTERNAL returns only the outermost contours
        # CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments
        mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rows = []
    for contour in contours:
        # boundingRect() returns the bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(contour)
        # Ignore very small sprites
        if w > 5 and h > 5:
            rows.append({"x": x, "y": y, "width": w, "height": h})
    # pd.DataFrame() creates a dataframe from a list of dictionaries
    # a DataFrame is a 2D array with labeled rows and columns
    df = pd.DataFrame(rows, columns=["x", "y", "width", "height"])
    df.to_csv(output_file, index=False)


def remove_background(sprite, background_color, tolerance=0):
    """Remove background from sprite with optional color tolerance.
    
    Args:
        sprite (PIL.Image): The sprite image
        background_color (tuple): RGB(A) color tuple to treat as background
        tolerance (int, optional): Color similarity tolerance (0-255). Defaults to 0.
    
    Returns:
        PIL.Image: Alpha mask for the sprite
    """
    # Convert to NumPy array for faster processing
    sprite_array = np.array(sprite)
    mask = Image.new("L", sprite.size, 0)
    mask_array = np.array(mask)
    
    # Handle both RGB and RGBA background colors
    bg_color = background_color[:3] if len(background_color) > 3 else background_color
    
    # Fast NumPy operations instead of pixel-by-pixel
    if sprite_array.shape[2] >= 3:  # Has RGB channels
        color_diff = np.abs(sprite_array[:,:,:3] - np.array(bg_color)).sum(axis=2)
        non_bg_pixels = color_diff > tolerance
        mask_array[non_bg_pixels] = 255
    
    return Image.fromarray(mask_array)


def sprite_look(spritesheet_file, spritesheet_csv, background_color):
    """Uses pillow to extract sprites from a spritesheet image and csv data
    and display them in a browser.
    \nAccepts background_color as a tuple of (r, g, b, a) values. "a" is optional."""
    # Load the spritesheet image
    spritesheet = Image.open(spritesheet_file)

    # Load the CSV file and iterate over the rows
    with open(spritesheet_csv) as f:
        next(f) # Skip the header row
        for line in f:
            # Parse the row data
            # map() applies the int() function to each element in the list
            x, y, width, height = map(int, line.split(","))

            # Extract the sprite image from the spritesheet
            # crop() returns a rectangular region from the image
            sprite = spritesheet.crop((x, y, x + width, y + height))

            mask = remove_background(sprite, background_color)
                        
            # Apply the mask to the sprite image
            sprite = sprite.copy()
            # copy() creates a copy of the image
            sprite.putalpha(mask)
            # putalpha() sets the alpha channel for the image
            # in this case, the alpha channel is the mask we created
            
            # Display the sprite image
            sprite.show()


def sprite_output(spritesheet_file, spritesheet_csv, name, output_dir, background_color):
    """Extracts sprites from a spritesheet image and csv data and saves them
    as PNG files to the specified directory.
    \n 'background_color' is a tuple of (r, g, b, a) values. "a" is optional.
    \n 'name' is the prefix for the output files, and each is numbered."""
    spritesheet = Image.open(spritesheet_file)
    
    with open(spritesheet_csv) as f:
        next(f)
        for i, line in enumerate(f):
            x, y, width, height = map(int, line.split(","))

            sprite = spritesheet.crop((x, y, x + width, y + height))
            
            mask = remove_background(sprite, background_color)
            
            sprite = sprite.copy()
            sprite.putalpha(mask)
            
            filename = f"{name}_{i}.png"
            sprite.save(os.path.join(output_dir, filename))


def batch_extract_sprites(spritesheet_files, output_dirs, background_colors, name_prefixes=None):
    """Batch process multiple spritesheets at once.
    
    Args:
        spritesheet_files (list): List of spritesheet file paths
        output_dirs (list): List of output directories (one per spritesheet)
        background_colors (list): List of background colors (one per spritesheet)
        name_prefixes (list, optional): List of prefixes for output files. Defaults to None.
    """
    if name_prefixes is None:
        name_prefixes = [f"sprite_{i}" for i in range(len(spritesheet_files))]
    
    for i, spritesheet_file in enumerate(spritesheet_files):
        # Generate CSV filename from spritesheet filename
        csv_file = f"{os.path.splitext(os.path.basename(spritesheet_file))[0]}.csv"
        
        # Ensure output directory exists
        os.makedirs(output_dirs[i], exist_ok=True)
        
        # Detect sprites and write to CSV
        detect_sprites(spritesheet_file, csv_file, background_colors[i])
        
        # Extract and save sprites
        sprite_output(
            spritesheet_file,
            csv_file,
            name_prefixes[i],
            output_dirs[i],
            background_colors[i][::-1] if len(background_colors[i]) == 3 else background_colors[i]  # Convert BGR to RGB
        )


def create_animation_sheet(sprite_files, output_file, rows=1, columns=None, spacing=0, background_color=(0, 0, 0, 0)):
    """Create an animation spritesheet from individual sprite images.
    
    Args:
        sprite_files (list): List of paths to individual sprite images
        output_file (str): Path to save the resulting spritesheet
        rows (int, optional): Number of rows in the sheet. Defaults to 1.
        columns (int, optional): Number of columns in the sheet. If None, calculated from len(sprite_files) and rows.
        spacing (int, optional): Spacing between sprites in pixels. Defaults to 0.
        background_color (tuple, optional): Background color of the sheet as RGBA. Defaults to transparent.
    
    Returns:
        PIL.Image: The generated spritesheet
    """
    if not sprite_files:
        raise ValueError("No sprite files provided")
    
    # Load the first sprite to get dimensions
    first_sprite = Image.open(sprite_files[0])
    sprite_width, sprite_height = first_sprite.size
    
    # Calculate columns if not specified
    if columns is None:
        columns = (len(sprite_files) + rows - 1) // rows  # Ceiling division
    
    # Calculate the dimensions of the spritesheet
    sheet_width = columns * sprite_width + (columns - 1) * spacing
    sheet_height = rows * sprite_height + (rows - 1) * spacing
    
    # Create a new image with an alpha channel
    spritesheet = Image.new('RGBA', (sheet_width, sheet_height), background_color)
    
    # Place each sprite in the sheet
    for i, sprite_file in enumerate(sprite_files):
        if i >= rows * columns:
            print(f"Warning: Only {rows * columns} slots available, skipping remaining sprites")
            break
            
        row = i // columns
        col = i % columns
        
        # Calculate position
        x = col * (sprite_width + spacing)
        y = row * (sprite_height + spacing)
        
        # Open and paste the sprite
        sprite = Image.open(sprite_file)
        spritesheet.paste(sprite, (x, y), sprite if sprite.mode == 'RGBA' else None)
    
    # Save the spritesheet
    spritesheet.save(output_file)
    return spritesheet


def transform_sprite(sprite_image, flip_h=False, flip_v=False, rotate=0, scale=1.0, color_adjust=None):
    """Apply various transformations to a sprite image.
    
    Args:
        sprite_image (PIL.Image): The source sprite image
        flip_h (bool, optional): Flip horizontally. Defaults to False.
        flip_v (bool, optional): Flip vertically. Defaults to False.
        rotate (int, optional): Rotation angle in degrees. Defaults to 0.
        scale (float, optional): Scale factor. Defaults to 1.0.
        color_adjust (dict, optional): Color adjustment parameters. Defaults to None.
            Example: {'brightness': 1.2, 'contrast': 0.8, 'saturation': 1.1}
            
    Returns:
        PIL.Image: Transformed sprite image
    """
    from PIL import ImageOps, ImageEnhance
    
    result = sprite_image.copy()
    
    # Apply flips
    if flip_h:
        result = result.transpose(Image.FLIP_LEFT_RIGHT)
    if flip_v:
        result = result.transpose(Image.FLIP_TOP_BOTTOM)
    
    # Apply rotation (expands canvas if needed)
    if rotate != 0:
        result = result.rotate(rotate, expand=True, resample=Image.BICUBIC)
    
    # Apply scaling
    if scale != 1.0:
        width, height = result.size
        new_width = int(width * scale)
        new_height = int(height * scale)
        result = result.resize((new_width, new_height), Image.BICUBIC)
    
    # Apply color adjustments
    if color_adjust:
        if 'brightness' in color_adjust:
            enhancer = ImageEnhance.Brightness(result)
            result = enhancer.enhance(color_adjust['brightness'])
        
        if 'contrast' in color_adjust:
            enhancer = ImageEnhance.Contrast(result)
            result = enhancer.enhance(color_adjust['contrast'])
        
        if 'saturation' in color_adjust:
            enhancer = ImageEnhance.Color(result)
            result = enhancer.enhance(color_adjust['saturation'])
    
    return result


def create_sprite_variations(sprite_image, output_dir, name_prefix, variations=None):
    """Create multiple variations of a sprite and save them.
    
    Args:
        sprite_image (PIL.Image): The source sprite image
        output_dir (str): Directory to save the variations
        name_prefix (str): Prefix for the output filenames
        variations (list, optional): List of variation specs. Defaults to creating standard variations.
            Each variation is a dict with any of the transform_sprite parameters.
            
    Returns:
        list: Paths to the created variation images
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Default variations if none specified
    if variations is None:
        variations = [
            {'name': 'original'},
            {'name': 'flipped', 'flip_h': True},
            {'name': 'rotate_90', 'rotate': 90},
            {'name': 'rotate_180', 'rotate': 180},
            {'name': 'rotate_270', 'rotate': 270},
            {'name': 'darker', 'color_adjust': {'brightness': 0.7}},
            {'name': 'lighter', 'color_adjust': {'brightness': 1.3}},
            {'name': 'larger', 'scale': 1.5},
            {'name': 'smaller', 'scale': 0.75},
        ]
    
    output_paths = []
    
    # Process each variation
    for i, var in enumerate(variations):
        # Get the variation name
        var_name = var.pop('name', f'variation_{i}')
        
        # Create the variation
        transformed = transform_sprite(sprite_image, **var)
        
        # Save the variation
        filename = f"{name_prefix}_{var_name}.png"
        filepath = os.path.join(output_dir, filename)
        transformed.save(filepath)
        output_paths.append(filepath)
        
    return output_paths


def generate_procedural_sprite(size, sprite_type='character', color_scheme=None, complexity=0.5, seed=None):
    """Generate a procedural sprite using algorithms.
    
    Args:
        size (tuple): Size of the sprite as (width, height)
        sprite_type (str, optional): Type of sprite to generate. Options:
            'character', 'item', 'weapon', 'environment'. Defaults to 'character'.
        color_scheme (list, optional): List of RGBA colors to use. Defaults to None (auto-generate).
        complexity (float, optional): How detailed the sprite should be (0.0-1.0). Defaults to 0.5.
        seed (int, optional): Seed for random generation. Defaults to None.
        
    Returns:
        PIL.Image: The generated sprite
    """
    from PIL import Image, ImageDraw
    import random
    import math
    
    # Set random seed for reproducibility if provided
    if seed is not None:
        random.seed(seed)
    
    # Create a transparent base image
    sprite = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(sprite)
    
    # Auto-generate color scheme if not provided
    if color_scheme is None:
        if sprite_type == 'character':
            # Character colors: skin, clothing, accent, detail
            hue = random.random()  # Base hue
            color_scheme = [
                # Skin tone (muted, lower saturation)
                _hsv_to_rgba(hue, 0.3, 0.8),
                # Main clothing (saturated, complementary to skin)
                _hsv_to_rgba((hue + 0.5) % 1.0, 0.7, 0.6),
                # Accent color (adjacent to clothing hue)
                _hsv_to_rgba((hue + 0.55) % 1.0, 0.8, 0.7),
                # Detail color (often darker)
                _hsv_to_rgba(hue, 0.2, 0.3),
            ]
        elif sprite_type == 'weapon':
            # Weapon colors: metal, handle, accent, glow
            color_scheme = [
                (180, 180, 190, 255),  # Metal
                (100, 60, 30, 255),    # Handle
                (220, 220, 230, 255),  # Highlight
                (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 200)  # Glow/magic
            ]
        else:
            # Default color scheme with some variety
            color_scheme = [
                (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200), 255),
                (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200), 255),
                (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200), 255),
                (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200), 255)
            ]
    
    width, height = size
    center_x, center_y = width // 2, height // 2
    
    # Generate based on sprite type
    if sprite_type == 'character':
        # Body
        body_width = int(width * 0.5)
        body_height = int(height * 0.6)
        body_top = center_y - body_height // 2
        
        # Draw body
        draw.ellipse(
            (center_x - body_width//2, body_top, 
             center_x + body_width//2, body_top + body_height), 
            fill=color_scheme[1]
        )
        
        # Head
        head_size = int(width * 0.3)
        head_top = body_top - head_size // 2
        draw.ellipse(
            (center_x - head_size//2, head_top,
             center_x + head_size//2, head_top + head_size),
            fill=color_scheme[0]
        )
        
        # Eyes
        eye_size = max(2, int(head_size * 0.2))
        eye_y = head_top + head_size // 2
        eye_spacing = eye_size * 1.5
        
        # Left eye
        draw.ellipse(
            (center_x - eye_spacing, eye_y - eye_size//2,
             center_x - eye_spacing + eye_size, eye_y + eye_size//2),
            fill=color_scheme[3]
        )
        
        # Right eye
        draw.ellipse(
            (center_x + eye_spacing - eye_size, eye_y - eye_size//2,
             center_x + eye_spacing, eye_y + eye_size//2),
            fill=color_scheme[3]
        )
        
        # Add details based on complexity
        if complexity > 0.3:
            # Arms
            arm_width = max(2, int(width * 0.12))
            arm_height = int(height * 0.4)
            arm_top = body_top + body_height // 4
            
            # Left arm
            draw.ellipse(
                (center_x - body_width//2 - arm_width//2, arm_top,
                 center_x - body_width//2 + arm_width//2, arm_top + arm_height),
                fill=color_scheme[1]
            )
            
            # Right arm
            draw.ellipse(
                (center_x + body_width//2 - arm_width//2, arm_top,
                 center_x + body_width//2 + arm_width//2, arm_top + arm_height),
                fill=color_scheme[1]
            )
        
        if complexity > 0.6:
            # Legs
            leg_width = max(2, int(width * 0.15))
            leg_height = int(height * 0.3)
            leg_top = body_top + body_height - leg_width // 2
            leg_spacing = body_width // 3
            
            # Left leg
            draw.ellipse(
                (center_x - leg_spacing - leg_width//2, leg_top,
                 center_x - leg_spacing + leg_width//2, leg_top + leg_height),
                fill=color_scheme[2]
            )
            
            # Right leg
            draw.ellipse(
                (center_x + leg_spacing - leg_width//2, leg_top,
                 center_x + leg_spacing + leg_width//2, leg_top + leg_height),
                fill=color_scheme[2]
            )
            
            # Mouth
            mouth_width = int(head_size * 0.6)
            mouth_height = max(1, int(head_size * 0.1))
            mouth_y = head_top + int(head_size * 0.7)
            
            draw.ellipse(
                (center_x - mouth_width//2, mouth_y,
                 center_x + mouth_width//2, mouth_y + mouth_height),
                fill=color_scheme[3]
            )
    
    elif sprite_type == 'weapon':
        # Blade
        blade_width = max(2, int(width * 0.2))
        blade_length = int(height * 0.7)
        
        # Draw the blade
        draw.rectangle(
            (center_x - blade_width//2, center_y - blade_length//2,
             center_x + blade_width//2, center_y + blade_length//2),
            fill=color_scheme[0]
        )
        
        # Handle
        handle_width = max(2, int(width * 0.12))
        handle_length = int(height * 0.3)
        
        # Draw the handle below the blade
        draw.rectangle(
            (center_x - handle_width//2, center_y + blade_length//2,
             center_x + handle_width//2, center_y + blade_length//2 + handle_length),
            fill=color_scheme[1]
        )
        
        # Cross guard
        guard_width = int(width * 0.5)
        guard_height = max(2, int(height * 0.05))
        
        # Draw the cross guard between blade and handle
        draw.rectangle(
            (center_x - guard_width//2, center_y + blade_length//2 - guard_height//2,
             center_x + guard_width//2, center_y + blade_length//2 + guard_height//2),
            fill=color_scheme[2]
        )
        
        # Add details based on complexity
        if complexity > 0.4:
            # Blade highlight
            highlight_width = max(1, int(blade_width * 0.4))
            
            draw.line(
                (center_x, center_y - blade_length//2,
                 center_x, center_y + blade_length//2),
                fill=color_scheme[2],
                width=highlight_width
            )
        
        if complexity > 0.7:
            # Glowing effect on the blade
            # We'll create a second image for the glow and then composite
            glow = Image.new('RGBA', size, (0, 0, 0, 0))
            glow_draw = ImageDraw.Draw(glow)
            
            glow_color = color_scheme[3]
            glow_width = blade_width * 2
            
            # Draw wider but semi-transparent blade
            glow_draw.rectangle(
                (center_x - glow_width//2, center_y - blade_length//2,
                 center_x + glow_width//2, center_y + blade_length//2),
                fill=glow_color
            )
            
            # Blur the glow (approximated with multiple drawings)
            for i in range(5):
                factor = (5 - i) / 5
                current_width = int(glow_width * factor)
                current_alpha = int(100 * (1 - factor))
                current_color = (glow_color[0], glow_color[1], glow_color[2], current_alpha)
                
                glow_draw.rectangle(
                    (center_x - current_width//2, center_y - blade_length//2,
                     center_x + current_width//2, center_y + blade_length//2),
                    fill=current_color
                )
            
            # Composite the glow under the main sprite
            sprite = Image.alpha_composite(glow, sprite)
            draw = ImageDraw.Draw(sprite)
    
    elif sprite_type == 'item':
        # Simple item (potion, scroll, etc.)
        item_size = min(width, height) * 0.8
        
        # Base shape
        shape_type = random.choice(['circle', 'square', 'diamond'])
        
        if shape_type == 'circle':
            draw.ellipse(
                (center_x - item_size//2, center_y - item_size//2,
                 center_x + item_size//2, center_y + item_size//2),
                fill=color_scheme[0]
            )
        elif shape_type == 'square':
            draw.rectangle(
                (center_x - item_size//2, center_y - item_size//2,
                 center_x + item_size//2, center_y + item_size//2),
                fill=color_scheme[0]
            )
        else:  # diamond
            draw.polygon(
                [(center_x, center_y - item_size//2),
                 (center_x + item_size//2, center_y),
                 (center_x, center_y + item_size//2),
                 (center_x - item_size//2, center_y)],
                fill=color_scheme[0]
            )
        
        # Inner detail
        inner_size = item_size * 0.6
        
        if shape_type == 'circle':
            draw.ellipse(
                (center_x - inner_size//2, center_y - inner_size//2,
                 center_x + inner_size//2, center_y + inner_size//2),
                fill=color_scheme[1]
            )
        elif shape_type == 'square':
            draw.rectangle(
                (center_x - inner_size//2, center_y - inner_size//2,
                 center_x + inner_size//2, center_y + inner_size//2),
                fill=color_scheme[1]
            )
        else:  # diamond
            draw.polygon(
                [(center_x, center_y - inner_size//2),
                 (center_x + inner_size//2, center_y),
                 (center_x, center_y + inner_size//2),
                 (center_x - inner_size//2, center_y)],
                fill=color_scheme[1]
            )
    
    elif sprite_type == 'environment':
        # Terrain or environmental element
        # We'll create a more complex shape with noise
        
        # Base shape covering most of the image
        draw.rectangle(
            (0, 0, width, height),
            fill=color_scheme[0]
        )
        
        # Add noise pattern
        noise_scale = int(10 / complexity)  # Higher complexity = finer noise
        for x in range(0, width, noise_scale):
            for y in range(0, height, noise_scale):
                if random.random() < 0.3:  # 30% chance of noise pixels
                    noise_color = random.choice(color_scheme[1:])
                    rect_size = noise_scale * random.uniform(0.5, 1.5)
                    
                    draw.rectangle(
                        (x, y, x + rect_size, y + rect_size),
                        fill=noise_color
                    )
    
    return sprite


def _hsv_to_rgba(h, s, v, a=255):
    """Convert HSV to RGBA.
    
    Args:
        h (float): Hue (0.0-1.0)
        s (float): Saturation (0.0-1.0)
        v (float): Value (0.0-1.0)
        a (int, optional): Alpha (0-255). Defaults to 255.
    
    Returns:
        tuple: RGBA color tuple
    """
    import colorsys
    
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255), a)


def upscale_sprite(sprite_image, scale_factor=2, method='bicubic'):
    """Upscale a sprite image using high-quality resampling techniques.
    
    Args:
        sprite_image (PIL.Image): The source sprite image
        scale_factor (int, optional): Scale multiplier (2x, 3x, 4x). Defaults to 2.
        method (str, optional): Resampling method. Options: 'nearest', 'box', 'bilinear', 
                               'bicubic', 'lanczos', 'enhanced'. Defaults to 'bicubic'.
        
    Returns:
        PIL.Image: Upscaled sprite image
    """
    from PIL import Image
    
    orig_width, orig_height = sprite_image.size
    new_width = int(orig_width * scale_factor)
    new_height = int(orig_height * scale_factor)
    
    # If method is 'enhanced', use a custom sequence of resamplings
    if method == 'enhanced':
        # First upscale with LANCZOS for smooth interpolation
        temp_image = sprite_image.resize((new_width, new_height), Image.LANCZOS)
        
        # Apply a subtle unsharp mask to enhance edges
        from PIL import ImageFilter
        temp_image = temp_image.filter(ImageFilter.UnsharpMask(radius=0.5, percent=150, threshold=3))
        
        # Slightly adjust contrast to make pixel art "pop"
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(temp_image)
        upscaled_image = enhancer.enhance(1.05)
        
        return upscaled_image
    
    # For other methods, map the string to PIL's resampling filters
    resampling_methods = {
        'nearest': Image.NEAREST,
        'box': Image.BOX,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'lanczos': Image.LANCZOS
    }
    
    # Default to BICUBIC if method not found
    resampling_filter = resampling_methods.get(method, Image.BICUBIC)
    
    # Perform the resize operation with the selected filter
    upscaled_image = sprite_image.resize((new_width, new_height), resampling_filter)
    
    return upscaled_image


def pixel_art_upscale(sprite_image, scale_factor=2, edge_preservation=1.0):
    """Upscale pixel art using an algorithm that preserves pixel art characteristics.
    This is a specialized algorithm that works well with pixel art and maintains crisp edges.
    
    Args:
        sprite_image (PIL.Image): The source sprite image
        scale_factor (int, optional): Scale multiplier (2x, 3x, 4x). Defaults to 2.
        edge_preservation (float, optional): How much to preserve pixel edges (0.0-2.0). 
                                           Higher values create crisper edges. Defaults to 1.0.
        
    Returns:
        PIL.Image: Upscaled sprite image with preserved pixel art characteristics
    """
    import numpy as np
    from PIL import Image, ImageFilter
    
    # Convert to numpy array for processing
    img_array = np.array(sprite_image)
    
    # Determine if we're dealing with an image that has an alpha channel
    has_alpha = sprite_image.mode == 'RGBA'
    
    # Get dimensions
    height, width = img_array.shape[:2]
    new_height = height * scale_factor
    new_width = width * scale_factor
    
    # Create a blank upscaled array
    if has_alpha:
        upscaled = np.zeros((new_height, new_width, 4), dtype=np.uint8)
    else:
        upscaled = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
    # Perform a simple nearest-neighbor upscale first
    for y in range(height):
        for x in range(width):
            for dy in range(scale_factor):
                for dx in range(scale_factor):
                    upscaled[y*scale_factor+dy, x*scale_factor+dx] = img_array[y, x]
    
    # Convert back to PIL Image
    upscaled_image = Image.fromarray(upscaled)
    
    # Apply edge-sensitive filtering
    if edge_preservation > 0:
        # Use a combination of filters to enhance edges while preserving the pixel art look
        # First apply a slight blur to reduce nearest-neighbor artifacts
        blurred = upscaled_image.filter(ImageFilter.GaussianBlur(radius=0.3 * edge_preservation))
        
        # Then apply an unsharp mask to recover and enhance edges
        edge_enhanced = upscaled_image.filter(
            ImageFilter.UnsharpMask(radius=0.5, percent=150 * edge_preservation, threshold=1)
        )
        
        # Blend the two images based on edge preservation factor
        if has_alpha:
            # Handle alpha channel separately
            r1, g1, b1, a1 = blurred.split()
            r2, g2, b2, a2 = edge_enhanced.split()
            
            # Convert to numpy arrays for blending
            r1_array = np.array(r1)
            g1_array = np.array(g1)
            b1_array = np.array(b1)
            r2_array = np.array(r2)
            g2_array = np.array(g2)
            b2_array = np.array(b2)
            
            # Calculate blend ratio based on edge preservation
            blend_ratio = min(max(0.0, 1.0 - (edge_preservation / 2.0)), 1.0)
            
            # Blend the RGB channels
            r_blend = (r1_array * blend_ratio + r2_array * (1 - blend_ratio)).astype(np.uint8)
            g_blend = (g1_array * blend_ratio + g2_array * (1 - blend_ratio)).astype(np.uint8)
            b_blend = (b1_array * blend_ratio + b2_array * (1 - blend_ratio)).astype(np.uint8)
            
            # Recreate image with blended channels
            blended = Image.merge('RGBA', (
                Image.fromarray(r_blend),
                Image.fromarray(g_blend),
                Image.fromarray(b_blend),
                a1
            ))
            
            return blended
        else:
            # For RGB images, use PIL's blend function
            blend_ratio = min(max(0.0, 1.0 - (edge_preservation / 2.0)), 1.0)
            from PIL import Image, ImageChops
            
            # Calculate difference and apply blending
            diff = ImageChops.difference(upscaled_image, blurred)
            return ImageChops.blend(blurred, upscaled_image, blend_ratio)
    
    return upscaled_image


def batch_upscale_sprites(sprite_files, output_dir, scale_factor=2, method='enhanced', name_suffix='_upscaled'):
    """Batch upscale multiple sprite files and save them to an output directory.
    
    Args:
        sprite_files (list): List of paths to sprite image files
        output_dir (str): Directory to save the upscaled images
        scale_factor (int, optional): Scale factor (2x, 3x, 4x). Defaults to 2.
        method (str, optional): Upscaling method ('enhanced', 'pixel_art', or any PIL method). Defaults to 'enhanced'.
        name_suffix (str, optional): Suffix to append to filenames. Defaults to '_upscaled'.
        
    Returns:
        list: Paths to the upscaled image files
    """
    import os
    from PIL import Image
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_paths = []
    
    for sprite_file in sprite_files:
        # Load the sprite image
        sprite = Image.open(sprite_file)
        
        # Get the base filename
        basename = os.path.basename(sprite_file)
        name, ext = os.path.splitext(basename)
        
        # Upscale based on the selected method
        if method == 'pixel_art':
            upscaled = pixel_art_upscale(sprite, scale_factor=scale_factor)
        else:
            upscaled = upscale_sprite(sprite, scale_factor=scale_factor, method=method)
        
        # Save the upscaled image
        out_path = os.path.join(output_dir, f"{name}{name_suffix}{ext}")
        upscaled.save(out_path)
        output_paths.append(out_path)
    
    return output_paths


def upscale_sprite_xbr(input_path, output_path=None, scale_factor=2):
    """Upscale sprite using an XBR-like algorithm optimized for pixel art.
    This implements a simplified version of the hqx algorithm, which is
    especially good for sprite and pixel art upscaling.
    
    Args:
        input_path (str): Path to input image or PIL Image
        output_path (str, optional): Path to save output. If None, returns the image without saving.
        scale_factor (int, optional): Scale factor (2 or 4). Defaults to 2.
    
    Returns:
        numpy.ndarray: The upscaled image array
    """
    import cv2
    import numpy as np
    
    # Support for PIL Image or file path
    if isinstance(input_path, str):
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    else:
        # Convert PIL Image to OpenCV format
        img = np.array(input_path)
        # Convert RGB to BGR for OpenCV
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    
    # Get dimensions and create output image
    h, w = img.shape[:2]
    channels = img.shape[2] if len(img.shape) > 2 else 1
    
    # For grayscale images
    if channels == 1:
        # Simple nearest-neighbor upscaling for grayscale
        upscaled = cv2.resize(img, (w * scale_factor, h * scale_factor), 
                             interpolation=cv2.INTER_NEAREST)
    else:
        # Alpha channel handling
        has_alpha = channels == 4
        
        if has_alpha:
            # Split the alpha channel
            bgr = img[:,:,0:3]
            alpha = img[:,:,3]
        else:
            bgr = img
        
        # Nearest-neighbor upscale first to maintain pixel boundaries
        upscaled_nn = cv2.resize(bgr, (w * scale_factor, h * scale_factor), 
                                interpolation=cv2.INTER_NEAREST)
        
        # Edge-aware processing
        # This simulates aspects of hq2x/xBR algorithms without requiring external libraries
        
        # 1. Detect edges in the original image (helps identify pixel boundaries)
        edges = cv2.Canny(bgr, 100, 200)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        
        # Upscale the edge map
        edges_upscaled = cv2.resize(edges, (w * scale_factor, h * scale_factor), 
                                    interpolation=cv2.INTER_NEAREST)
        
        # 2. Create a smoother version using Lanczos interpolation (good for diagonal edges)
        upscaled_smooth = cv2.resize(bgr, (w * scale_factor, h * scale_factor), 
                                    interpolation=cv2.INTER_LANCZOS4)
        
        # 3. Blend the nearest-neighbor and smooth versions based on edge map
        # Where edges exist, use nearest-neighbor to keep crisp boundaries
        # In other areas, use the smoother version for better gradients
        mask = edges_upscaled / 255.0
        mask = np.expand_dims(mask, axis=2)  # Make 3-channel
        upscaled_bgr = upscaled_nn * mask + upscaled_smooth * (1 - mask)
        
        # 4. Optional post-processing: add slight sharpening to enhance edges
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]]) * 0.3 + np.eye(3) * 0.7
        
        upscaled_bgr = cv2.filter2D(upscaled_bgr, -1, kernel)
        
        # Handle alpha channel if present
        if has_alpha:
            alpha_upscaled = cv2.resize(alpha, (w * scale_factor, h * scale_factor), 
                                        interpolation=cv2.INTER_NEAREST)
            upscaled = np.zeros((h * scale_factor, w * scale_factor, 4), dtype=np.uint8)
            upscaled[:,:,0:3] = upscaled_bgr
            upscaled[:,:,3] = alpha_upscaled
        else:
            upscaled = upscaled_bgr
    
    # Save output if path provided
    if output_path:
        cv2.imwrite(output_path, upscaled)
    
    return upscaled


def batch_upscale_directory(input_dir, output_dir, method='xbr', scale_factor=2, file_pattern='*.png'):
    """Upscale all matching sprite files in a directory.
    
    Args:
        input_dir (str): Directory containing sprite images
        output_dir (str): Directory to save upscaled images
        method (str, optional): Upscaling method ('nearest', 'bilinear', 'bicubic', 
                               'lanczos', 'xbr'). Defaults to 'xbr'.
        scale_factor (int, optional): Scale factor (2 or 4). Defaults to 2.
        file_pattern (str, optional): File pattern to match. Defaults to '*.png'.
        
    Returns:
        list: List of paths to upscaled images
    """
    import os
    import glob
    import cv2
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of files matching the pattern
    file_paths = glob.glob(os.path.join(input_dir, file_pattern))
    
    # Map method names to OpenCV interpolation methods
    cv2_methods = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    
    output_paths = []
    
    for file_path in file_paths:
        # Get base filename
        filename = os.path.basename(file_path)
        base_name, ext = os.path.splitext(filename)
        
        # Create output path
        output_path = os.path.join(output_dir, f"{base_name}_upscaled{ext}")
        output_paths.append(output_path)
        
        # Upscale based on selected method
        if method == 'xbr':
            upscale_sprite_xbr(file_path, output_path, scale_factor)
        else:
            # Use OpenCV's built-in methods
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            h, w = img.shape[:2]
            interpolation = cv2_methods.get(method, cv2.INTER_LANCZOS4)
            upscaled = cv2.resize(img, (w * scale_factor, h * scale_factor), interpolation=interpolation)
            cv2.imwrite(output_path, upscaled)
    
    return output_paths


# detect_sprites("sprites/The Guardian Alyssa.png", "TGA.csv", (0, 0, 38))
# 260000 - input must be in BGR values i.e. (0, 0, 38)

# sprite_look("sprites/The Guardian Alyssa.png", "TGA.csv", (38, 0, 0))

# sprite_output("sprites/The Guardian Alyssa.png", "TGA.csv", "TGA", "output", (38, 0, 0))

character = generate_procedural_sprite((64, 64), sprite_type='item', complexity=0.8)
character.save('character.png')