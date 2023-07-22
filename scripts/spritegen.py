from PIL import Image
import random

import numpy as np
import matplotlib.pyplot as plt

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

def generate_rand_template(x, y, show=False):
    """Generate numpy template 'tile' of on/off values"""
    # Define the size of the template
    template_size = (x, y)

    # Create an empty template
    template = np.zeros(template_size, dtype=int)

    # Procedurally generate the template
    for i in range(template_size[0]):
        for j in range(template_size[1]):
            # Randomly assign a value of 1 or 0
            template[i][j] = np.random.choice([0, 1])

    if show:
        # Display the template using matplotlib & QT
        # DEPENDS: sudo apt-get install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev
        plt.imshow(template, cmap='gray')
        plt.show()
    else:
        return template


class TileGenerator():
    def __init__(self, img_x, img_y, simplex_scale, palettesD, tile_settings, 
                 palette_var_thresh_A=0.3, palette_var_thresh_B=0.6, save_location=None, simplex=True):
        self.save_location = save_location
        self.tile_settings = tile_settings
        self.img_x = img_x
        self.img_y = img_y
        self.simplex_enabled = simplex
        self.simplex_scale = simplex_scale
        self.img = None  # the output image
        self.simplex_gen = OpenSimplex(seed=random.randint(0, 1000))
        self.color_range = None  # store range selected from palettesD
        self.palettesD = palettesD
        self.palette_var_thresh_A = palette_var_thresh_A 
        self.palette_var_thresh_B = palette_var_thresh_B

    def gen_simple_tile(self, type, variance=False):
        # Define your size and color range
        size = (self.img_x, self.img_y)
        self.color_range = palettesD[type]
        # Create a new image
        self.img = Image.new('RGB', size)
        # Get the pixels of the image
        pixels = self.img.load()
        # Pixel & color inits
        prev_pixel = [0, 0]
        prev_i = 0
        up_pixel = [0, 0]
        up_j = 0
        down_pixel = [0, 0]
        down_j = 0
        pixel_count = 0
        prev_color, up_color, down_color = None, None, None
        # Iterate over each pixel
        for i in range(self.img.size[0]):    # for every column
            if i > 0:
                prev_i = i - 1
            for j in range(self.img.size[1]):    # for every row
                if i > 0 and j > 0:
                    if j != 0:
                        up_j = j - 1
                    if j < self.img_y - 1:
                        down_j = j + 1
                    prev_pixel = (prev_i, j)
                    up_pixel = (i, up_j)
                    down_pixel = (i, down_j)
              
                    prev_color, up_color, down_color = self.get_other_pixel_colors(
                        self.img,
                        prev_pixel,
                        up_pixel,
                        down_pixel
                    )
          
                # Choose a random color from the color range
                if self.simplex_enabled:
                    color = self.get_simplex_color(
                        type=type,
                        i=i,
                        j=j,
                        scale=self.simplex_scale,
                        color_range=self.color_range,
                        variance=variance
                    )
                else:
                    color = random.choice(self.color_range)
                print(f"CURRENT: {[i, j]} | {color}")
                print(f'LEFT: {prev_pixel} | {prev_color}')
                print(f'UP: {up_pixel} | {up_color}')
                print(f'DOWN: {down_pixel} | {down_color}')
                print()
                
                # Set the pixel's color
                pixels[i, j] = color
                pixel_count += 1
        print(pixel_count)
        # Save the image
        if self.save_location:
            self.img.save(f'{self.save_location}/{type}2.png')
        else:
            self.img.save(f'{type}.png')
        
    def get_other_pixel_colors(self, img, prev, up, down):
        '''Grab colors from surrounding pixels'''
        return img.getpixel(prev), img.getpixel(up), img.getpixel(down)

    def get_simplex_color(self, type, i, j, scale, color_range, i_inc=0, j_inc=0, variance=False):
        """scale determines "zoom level", smaller = larger patterns"""
        # Gen noise val btwn -1 to 1
        noise_value = self.simplex_gen.noise2((i / scale) + i_inc, (j / scale) + j_inc)
        color_idx = 0
        if variance:
            # palette = color_range
            palette, scaled_noise_val = self.palette_by_noise_val(
                type=type,
                noise_val=noise_value,
                palettesD=self.palettesD,
                threshold_A=self.palette_var_thresh_A,
                threshold_B=self.palette_var_thresh_B
            )
            color_idx = int(scaled_noise_val * len(palette))
        else:
            # scale noise val to range of palette
            color_idx = int((noise_value + 1) / 2 * len(color_range))
        # return color from palette
        return color_range[color_idx]

    def palette_by_noise_val(self, type, noise_val, palettesD, threshold_A, threshold_B):
        scaled_noise_val = (noise_val + 1) / 2
        print(scaled_noise_val)
        type_variationsL = self.tile_settings[type]['variations']
        print(type_variationsL)
        if scaled_noise_val < threshold_A:
            print(type_variationsL[0])
            return palettesD[type_variationsL[0]], scaled_noise_val
        elif scaled_noise_val < threshold_B:
            print(type_variationsL[1])
            return palettesD[type_variationsL[1]], scaled_noise_val
        else:
            return palettesD[type], scaled_noise_val
  
tilegen = TileGenerator(
    img_x=64,
    img_y=64,
    simplex=True,
    simplex_scale=6,
    palettesD=palettesD,
    palette_var_thresh_A=0.4,
    palette_var_thresh_B=0.6,
    tile_settings=tile_settings
)
tilegen.gen_simple_tile(type='dirt', variance=True)

# generate_rand_template(64, 64, True)
