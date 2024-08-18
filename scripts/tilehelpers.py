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


def get_simplex_color(tile_type, i, j, scale, color_range, i_inc=0, j_inc=0, variance=False, palette_thresholds={}):
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
        palette_thresholds (dict, optional): Dictionary with 'A' and 'B' thresholds. Defaults to None.

    Returns:
        color: Color from the palette.
    """
    #scale determines "zoom level", smaller = larger patterns
    # Gen noise val btwn -1 to 1
    simplex_gen = OpenSimplex(seed=random.randint(0, 1000))
    noise_value = simplex_gen.noise2((i / scale) + i_inc, (j / scale) + j_inc)

    if variance:
        palette, scaled_noise_val = palette_by_noise_val(
            tile_type=tile_type,
            noise_val=noise_value,
            palettesD=palettesD,
            threshold_a=palette_thresholds['A'],
            threshold_b=palette_thresholds['B']
        )
        color_idx = int(scaled_noise_val * len(palette))
    else:
        color_idx = int((noise_value + 1) / 2 * len(color_range)) # scale noise val to range of palette

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
        palettes_dict (dict): Dictionary of palettes.
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