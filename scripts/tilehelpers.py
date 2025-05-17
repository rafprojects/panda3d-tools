import random

from opensimplex import OpenSimplex

palettesD = {
    'regular_grass': [(i, random.randint(100, 255), i) for i in range(50, 100)],
    'dirt': [(i, int(i / 1.5), int(i / 3)) for i in range(85, 160)],
    'dry_grass': [(i, i, int(i / 2)) for i in range(160, 230)],
    'green_grass': [(i, random.randint(100, 255), i) for i in range(50, 100)],
    'lava': [(random.randint(200, 255), i, int(i / 4)) for i in range(10, 100)],
    'ocean': [(i, i, random.randint(100, 255)) for i in range(50, 100)],
    'sand': [(i, i, int(i / 1.2)) for i in range(200, 255)],
    'rock': [(i, i, i) for i in range(50, 120)],
    'river': [(i, i, random.randint(150, 255)) for i in range(70, 100)],
    'swamp': [(i, random.randint(100, 150), i) for i in range(50, 80)],
    'pond': [(i, i, random.randint(120, 200)) for i in range(60, 90)]
}
tile_settings = {
    'regular_grass': {
        'variations': ['dirt', 'dry_grass']
    },
    'dirt': {
        'variations': ['dry_grass', 'rock']
    },
    'dry_grass': {
        'variations': ['sand', 'rock']
    },
    'green_grass': {
        'variations': ['dirt', 'regular_grass']
    }
}

light_directions = {
    'bottom-right': (1, 1),
    'top-left': (-1, -1),
    'top-right': (1, -1),
    'bottom-left': (-1, 1)
}

base_rock_shape = [
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1]
]


def save_img(img, save_location, img_name):
    if save_location:
        img.save(f'{save_location}/{img_name}.png')
    else:
        img.save(f'{img_name}.png')


def get_simplex_color(tile_type, i, j, scale, color_range, i_inc=0, j_inc=0, variance=False, palette_thresholds={}, simplex_gen=None):
    """
    Get the color based on simplex noise.

    Args:
        tile_type (str): Type of the tile.
        i (int): X-coordinate.
        j (int): Y-coordinate.
        scale (float): Scale for noise generation.
        color_range (list): List of colors.
        i_inc (int, optional): Increment for i. Defaults to 0.
        j_inc (int, optional): Increment for j. Defaults to 0.
        variance (bool, optional): Whether to use variance. Defaults to False.
        palette_thresholds (dict, optional): Dictionary with 'A' and 'B' thresholds. Defaults to empty dict.
        simplex_gen (OpenSimplex, optional): Existing noise generator. Defaults to None.

    Returns:
        color: Color from the palette.
    """
    # Use existing generator or create a new one - this prevents creating a new generator for each pixel
    if simplex_gen is None:
        simplex_gen = OpenSimplex(seed=random.randint(0, 1000))
        
    # Generate noise value between -1 to 1
    noise_value = simplex_gen.noise2((i / scale) + i_inc, (j / scale) + j_inc)

    if variance:
        palette, scaled_noise_val = palette_by_noise_val(
            tile_type=tile_type,
            noise_val=noise_value,
            palettesD=palettesD,
            threshold_a=palette_thresholds.get('A', 0.33),
            threshold_b=palette_thresholds.get('B', 0.66)
        )
        color_idx = int(scaled_noise_val * len(palette))
    else:
        color_idx = int((noise_value + 1) / 2 * len(color_range)) # scale noise val to range of palette
    
    # Ensure color_idx is within the valid range
    color_idx = min(max(color_idx, 0), len(color_range) - 1)
    
    return color_range[color_idx]
    
    
def get_other_pixel_colors(img, prev, up, down):
    '''Grab colors from surrounding pixels'''
    return img.getpixel(prev), img.getpixel(up), img.getpixel(down)


def palette_by_noise_val(tile_type, noise_val, palettesD, threshold_a, threshold_b):
    """
    Get the palette based on noise value.

    Args:
        tile_type (str): Type of the tile.
        noise_val (float): Noise value.
        palettesD (dict): Dictionary of palettes.
        threshold_a (float): Threshold A.
        threshold_b (float): Threshold B.

    Returns:
        tuple: Palette and scaled noise value.
    """
    scaled_noise_val = (noise_val + 1) / 2
    variations = tile_settings.get(tile_type, {})
    
    if variations:
        type_variationsL = tile_settings[tile_type]['variations']
        if scaled_noise_val < threshold_a:
            return palettesD[type_variationsL[0]], scaled_noise_val
        elif scaled_noise_val < threshold_b:
            return palettesD[type_variationsL[1]], scaled_noise_val
        else:
            # If above both thresholds, default to main tile_type palette
            return palettesD[tile_type], scaled_noise_val
    else:
        return palettesD[tile_type], scaled_noise_val


def draw_grass_blade(draw, x, y, direction, color, length_factor=1.0):
    # draw = ImageDraw object
    length = random.randint(5, 15) * length_factor
    if direction == 'left':
        end_x = x - random.randint(0, 5)
    elif direction == 'right':
        end_x = x + random.randint(0, 5)
    else:
        end_x = x
    end_y = y - length
    draw.line([(x, y), (end_x, end_y)], fill=color, width=1)


def generate_tileable_texture(width, height, tile_type=None, simplex_scale=2.0, border_blend=True, blend_amount=0.2):
    """Generate a seamlessly tileable texture using simplex noise.
    
    Args:
        width (int): Width of the texture
        height (int): Height of the texture
        tile_type (str, optional): Type of tile from palettesD. If None, uses rock. Defaults to None.
        simplex_scale (float, optional): Scale of the noise. Smaller = larger patterns. Defaults to 2.0.
        border_blend (bool, optional): Whether to blend the borders to ensure tileable edges. Defaults to True.
        blend_amount (float, optional): Amount of border blending. Defaults to 0.2.
        
    Returns:
        PIL.Image: The generated tileable texture
    """
    from opensimplex import OpenSimplex
    
    # Use rock palette by default if tile_type not specified
    tile_type = tile_type or 'rock'
    if tile_type not in palettesD:
        raise ValueError(f"Unknown tile type: {tile_type}")
        
    # Create image and get pixel access
    image = Image.new('RGB', (width, height))
    pixels = image.load()
    
    # Create noise generator
    noise_gen = OpenSimplex(seed=random.randint(0, 10000))
    
    # Generate base texture
    for x in range(width):
        for y in range(height):
            # Map coordinates to range [0, 1]
            nx = x / width
            ny = y / height
            
            # Use sin to make the noise wrap around at the edges
            if border_blend:
                x_val = nx * 2 * math.pi
                y_val = ny * 2 * math.pi
                # This creates a tileable noise pattern by using sine waves
                noise_val = noise_gen.noise2d(
                    math.sin(x_val) * simplex_scale,
                    math.sin(y_val) * simplex_scale
                )
            else:
                # Non-tileable version
                noise_val = noise_gen.noise2d(
                    nx * simplex_scale,
                    ny * simplex_scale
                )
            
            # Convert noise [-1, 1] to color index
            colors = palettesD[tile_type]
            color_idx = int((noise_val + 1) / 2 * len(colors))
            color_idx = min(max(color_idx, 0), len(colors) - 1)
            pixels[x, y] = colors[color_idx]
    
    # Apply edge blending if requested
    if border_blend and blend_amount > 0:
        # Create a copy of the image for blending
        blended = image.copy()
        blend_pixels = blended.load()
        
        # Calculate blend factor based on distance from edge
        for x in range(width):
            for y in range(height):
                # Calculate distance from the nearest edge (normalized to [0, 1])
                edge_dist = min(x, width - x - 1) / (width * blend_amount)
                edge_dist = min(edge_dist, min(y, height - y - 1) / (height * blend_amount))
                edge_dist = min(1.0, edge_dist)  # Clamp to 1.0
                
                # If pixel is near an edge, blend with the opposite side
                if edge_dist < 1.0:
                    # Calculate corresponding pixel on opposite side
                    x_opp = (x + width // 2) % width
                    y_opp = (y + height // 2) % height
                    
                    # Get colors
                    orig_color = pixels[x, y]
                    opp_color = pixels[x_opp, y_opp]
                    
                    # Blend colors based on edge distance
                    blend_factor = edge_dist
                    r = int(orig_color[0] * blend_factor + opp_color[0] * (1 - blend_factor))
                    g = int(orig_color[1] * blend_factor + opp_color[1] * (1 - blend_factor))
                    b = int(orig_color[2] * blend_factor + opp_color[2] * (1 - blend_factor))
                    
                    # Store blended color
                    blend_pixels[x, y] = (r, g, b)
        
        # Replace original image with blended version
        image = blended
    
    return image
