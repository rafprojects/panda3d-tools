from PIL import ImageDraw
import random

import numpy as np
import matplotlib.pyplot as plt

from tilehelpers import palettesD, save_img, get_simplex_color, get_other_pixel_colors
from tilegenerator import TileCanvasGenerator


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


def gen_simple_tile(img_obj, tile_type, variance=False, simplex_enabled=False, simplex_scale=None, palette_var_thresh_A=None, palette_var_thresh_B=None, save=False, save_location=None):
    # size = (img_x, img_y)
    color_range = palettesD[tile_type]
    # img = Image.new('RGB', size)

    # Get the pixels of the image
    pixels = img_obj.load()
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
    for i in range(img_obj.size[0]):    # for every column
        if i > 0:
            prev_i = i - 1
        for j in range(img_obj.size[1]):    # for every row
            if i > 0 and j > 0:
                if j != 0:
                    up_j = j - 1
                if j < img_obj.size[1] - 1:
                    down_j = j + 1
                prev_pixel = (prev_i, j)
                up_pixel = (i, up_j)
                down_pixel = (i, down_j)

                prev_color, up_color, down_color = get_other_pixel_colors(
                    img_obj,
                    prev_pixel,
                    up_pixel,
                    down_pixel
                )
            # Choose a random color from the color range
            if simplex_enabled:
                color = get_simplex_color(
                    type=tile_type,
                    i=i,
                    j=j,
                    scale=simplex_scale,
                    color_range=color_range,
                    variance=variance,
                    palette_var_thresh_A=palette_var_thresh_A,
                    palette_var_thresh_B=palette_var_thresh_B
                )
            else:
                color = random.choice(color_range)
            # print(f"CURRENT: {[i, j]} | {color}")
            # print(f'LEFT: {prev_pixel} | {prev_color}')
            # print(f'UP: {up_pixel} | {up_color}')
            # print(f'DOWN: {down_pixel} | {down_color}')
            print()
            # Set the pixel's color
            pixels[i, j] = color
            pixel_count += 1
    print(pixel_count)

    if save:
        save_img(img=img_obj, save_location=save_location, img_name=tile_type)
    else:
        return img_obj


def generate_brick_pattern(img_obj, size, brick_count_x, brick_count_y, brick_color, mortar_color, mortar_size=2, save=False, save_location=None):
    draw = ImageDraw.Draw(img_obj)
    # Calculate the size of bricks and mortar
    brick_width = (size[0] - (brick_count_x - 1) * mortar_size) // brick_count_x
    brick_height = (size[1] - (brick_count_y - 1) * mortar_size) // brick_count_y
    # Full and half brick width
    full_brick = (brick_width, brick_height)
    half_brick = (brick_width // 2, brick_height)
    # Draw each row
    for j in range(brick_count_y):
        y_top = j * (brick_height + mortar_size)
        y_bottom = y_top + brick_height
        # Determine the pattern for this row
        pattern = [full_brick] * brick_count_x if j % 2 == 0 else [half_brick] + [full_brick] * (brick_count_x - 1)
        x_left = 0
        for brick in pattern:
            x_right = x_left + brick[0]
            # If this brick goes beyond the edge, trim it
            if x_right > size[0]:
                x_right = size[0]
            # Draw the brick
            draw.rectangle([x_left, y_top, x_right, y_bottom], fill=brick_color)
            # Next brick position
            x_left = x_right + mortar_size
    # Draw the bottom row if there is space left
    if y_bottom < size[1]:
        draw.rectangle([0, y_bottom, size[0], size[1]], fill=mortar_color)
    if save:
        save_img(img=img_obj, save_location=save_location, img_name='brick_tessel2')
    else:
        return img_obj


def gen_grass_tile(draw_obj, direction='straight'):
    base_color_range = palettesD['regular_grass']



# def gen_rock_tile(base_shape):
#     tile = np.copy(base_shape)
#     for i in range(tile.shape[0]):
#         for j in range(tile.shape[1]):
#             # gen rand variations
#             if tile[i][j] == 1 and random.random() < 0.1:  # 10% chance to become a 0
#                 tile[i][j] = 0
#     return tile

# TESTING

tg = TileCanvasGenerator(64, 64)
tile = tg.make_empty_tile()

gen_simple_tile(tile, tile_type='ocean', variance=True, simplex_enabled=True, simplex_scale=2, palette_var_thresh_A=4, palette_var_thresh_B=6, save=True, save_location='.')
# tilegen.generate_brick_pattern(size=(128, 128), brick_count_x=12, brick_count_y=20, brick_color=(150, 75, 50), mortar_color=(200, 200, 200))
# generate_rand_template(64, 64, True)
