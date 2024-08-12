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


def get_simplex_color(type, i, j, scale, color_range, i_inc=0, j_inc=0, variance=False, palette_var_thresh_A=None, palette_var_thresh_B=None):
        """scale determines "zoom level", smaller = larger patterns"""
        # Gen noise val btwn -1 to 1
        simplex_gen = OpenSimplex(seed=random.randint(0, 1000))
        noise_value = simplex_gen.noise2((i / scale) + i_inc, (j / scale) + j_inc)
        color_idx = 0
        if variance:
            # palette = color_range
            palette, scaled_noise_val = palette_by_noise_val(
                type=type,
                noise_val=noise_value,
                palettesD=palettesD,
                threshold_A=palette_var_thresh_A,
                threshold_B=palette_var_thresh_B
            )
            color_idx = int(scaled_noise_val * len(palette))
        else:
            # scale noise val to range of palette
            color_idx = int((noise_value + 1) / 2 * len(color_range))
        # return color from palette
        return color_range[color_idx]
    
    
def get_other_pixel_colors(img, prev, up, down):
    '''Grab colors from surrounding pixels'''
    return img.getpixel(prev), img.getpixel(up), img.getpixel(down)


def palette_by_noise_val(type, noise_val, palettesD, threshold_A, threshold_B):
    scaled_noise_val = (noise_val + 1) / 2
    print(scaled_noise_val)
    variations = tile_settings.get(type, {})
    if variations:
        type_variationsL = tile_settings[type]['variations']
        print(type_variationsL)
        if scaled_noise_val < threshold_A:
            print(type_variationsL[0])
            return palettesD[type_variationsL[0]], scaled_noise_val
        elif scaled_noise_val < threshold_B:
            print(type_variationsL[1])
            return palettesD[type_variationsL[1]], scaled_noise_val
    else:
        return palettesD[type], scaled_noise_val


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