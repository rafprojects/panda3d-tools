from PIL import ImageDraw
import random

import numpy as np
import matplotlib.pyplot as plt

from tilehelpers import palettesD, save_img, get_simplex_color, get_other_pixel_colors
from tilegenerator import TileCanvasGenerator


# def generate_rand_template(x, y, show=False):
#     """Generate numpy template 'tile' of on/off values"""
#     # Define the size of the template
#     template_size = (x, y)

#     # Create an empty template
#     template = np.zeros(template_size, dtype=int)

#     # Procedurally generate the template
#     for i in range(template_size[0]):
#         for j in range(template_size[1]):
#             # Randomly assign a value of 1 or 0
#             template[i][j] = np.random.choice([0, 1])

#     if show:
#         # Display the template using matplotlib & QT
#         # DEPENDS: sudo apt-get install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev
#         plt.imshow(template, cmap='gray')
#         plt.show()
#     else:
#         return template

def generate_rand_template(x, y, show=False):
    """Generate numpy template 'tile' of on/off values"""
    rng = np.random.default_rng()
    template = rng.choice([0, 1], size=(x, y), dtype=int)
    
    if show:
        display_template(template)
    else:
        return template

def display_template(template):
    """Display the numpy template using matplotlib"""
    plt.imshow(template, cmap='gray')
    plt.show()


# def gen_simple_tile(img_obj, tile_type, variance=False, simplex_enabled=False, simplex_scale=None, palette_var_thresh_A=None, palette_var_thresh_B=None, save=False, save_location=None):
#     # size = (img_x, img_y)
#     color_range = palettesD[tile_type]
#     # img = Image.new('RGB', size)

#     # Get the pixels of the image
#     pixels = img_obj.load()
#     # Pixel & color inits
#     prev_pixel = [0, 0]
#     prev_i = 0
#     up_pixel = [0, 0]
#     up_j = 0
#     down_pixel = [0, 0]
#     down_j = 0
#     pixel_count = 0
#     prev_color, up_color, down_color = None, None, None
#     # Iterate over each pixel
#     for i in range(img_obj.size[0]):    # for every column
#         if i > 0:
#             prev_i = i - 1
#         for j in range(img_obj.size[1]):    # for every row
#             if i > 0 and j > 0:
#                 if j != 0:
#                     up_j = j - 1
#                 if j < img_obj.size[1] - 1:
#                     down_j = j + 1
#                 prev_pixel = (prev_i, j)
#                 up_pixel = (i, up_j)
#                 down_pixel = (i, down_j)

#                 prev_color, up_color, down_color = get_other_pixel_colors(
#                     img_obj,
#                     prev_pixel,
#                     up_pixel,
#                     down_pixel
#                 )
#             # Choose a random color from the color range
#             if simplex_enabled:
#                 color = get_simplex_color(
#                     type=tile_type,
#                     i=i,
#                     j=j,
#                     scale=simplex_scale,
#                     color_range=color_range,
#                     variance=variance,
#                     palette_var_thresh_A=palette_var_thresh_A,
#                     palette_var_thresh_B=palette_var_thresh_B
#                 )
#             else:
#                 color = random.choice(color_range)
#             # print(f"CURRENT: {[i, j]} | {color}")
#             # print(f'LEFT: {prev_pixel} | {prev_color}')
#             # print(f'UP: {up_pixel} | {up_color}')
#             # print(f'DOWN: {down_pixel} | {down_color}')
#             print()
#             # Set the pixel's color
#             pixels[i, j] = color
#             pixel_count += 1
#     print(pixel_count)

#     if save:
#         save_img(img=img_obj, save_location=save_location, img_name=tile_type)
#     else:
#         return img_obj

def gen_simple_tile(img_obj, tile_type, variance=False, simplex_enabled=False, simplex_scale=None, 
                    palette_var_thresh_A=None, palette_var_thresh_B=None, save=False, save_location=None):
    
    color_range = palettesD[tile_type]
    pixels = img_obj.load()
    
    width, height = img_obj.size
    for i in range(width):
        for j in range(height):
            if simplex_enabled:
                color = get_simplex_color(
                    tile_type=tile_type,
                    i=i,
                    j=j,
                    scale=simplex_scale,
                    color_range=color_range,
                    variance=variance,
                    palette_thresholds={'A': palette_var_thresh_A, 'B': palette_var_thresh_B}
                )
            else:
                color = random.choice(color_range)
            
            pixels[i, j] = color
    
    if save:
        save_img(img=img_obj, save_location=save_location, img_name=tile_type)
    else:
        return img_obj

# def generate_brick_pattern(img_obj, size, brick_count_x, brick_count_y, brick_color, mortar_color, mortar_size=2, save=False, save_location=None):
#     draw = ImageDraw.Draw(img_obj)
#     # Calculate the size of bricks and mortar
#     brick_width = (size[0] - (brick_count_x - 1) * mortar_size) // brick_count_x
#     brick_height = (size[1] - (brick_count_y - 1) * mortar_size) // brick_count_y
#     # Full and half brick width
#     full_brick = (brick_width, brick_height)
#     half_brick = (brick_width // 2, brick_height)
#     # Draw each row
#     for j in range(brick_count_y):
#         y_top = j * (brick_height + mortar_size)
#         y_bottom = y_top + brick_height
#         # Determine the pattern for this row
#         pattern = [full_brick] * brick_count_x if j % 2 == 0 else [half_brick] + [full_brick] * (brick_count_x - 1)
#         x_left = 0
#         for brick in pattern:
#             x_right = x_left + brick[0]
#             # If this brick goes beyond the edge, trim it
#             if x_right > size[0]:
#                 x_right = size[0]
#             # Draw the brick
#             draw.rectangle([x_left, y_top, x_right, y_bottom], fill=brick_color)
#             # Next brick position
#             x_left = x_right + mortar_size
#     # Draw the bottom row if there is space left
#     if y_bottom < size[1]:
#         draw.rectangle([0, y_bottom, size[0], size[1]], fill=mortar_color)
#     if save:
#         save_img(img=img_obj, save_location=save_location, img_name='brick_tessel2')
#     else:
#         return img_obj

# def generate_brick_pattern(img_obj, size, brick_count_x, brick_count_y, brick_color, 
#                            mortar_color, mortar_size=2, save=False, save_location=None):
#     draw = ImageDraw.Draw(img_obj)
#     # Calculate the size of bricks and mortar
#     brick_width = (size[0] - (brick_count_x - 1) * mortar_size) // brick_count_x
#     brick_height = (size[1] - (brick_count_y - 1) * mortar_size) // brick_count_y
    
#     for row in range(brick_count_y):
#         y_top = row * (brick_height + mortar_size)
#         y_bottom = y_top + brick_height
        
#         x_offset = brick_width // 2 if row % 2 else 0
        
#         for col in range(brick_count_x):
#             x_left = col * (brick_width + mortar_size) + x_offset
#             x_right = x_left + brick_width
            
#             if x_right > size[0]:
#                 x_right = size[0]
            
#             draw.rectangle([x_left, y_top, x_right, y_bottom], fill=brick_color)
    
#     if y_bottom < size[1]:
#         draw.rectangle([0, y_bottom, size[0], size[1]], fill=mortar_color)
    
#     if save:
#         save_img(img=img_obj, save_location=save_location, img_name='brick_tessel2')
#     else:
#         return img_obj

def generate_brick_pattern(img_obj, size, brick_count_x, brick_count_y, brick_color, 
                           mortar_color, mortar_size=2, top_mortar=True, 
                           light_direction='top-left', shadow_color=None, highlight_color=None, 
                           shadow_size=0.5, highlight_size=0.5,
                           save=False, save_location=None):
    draw = ImageDraw.Draw(img_obj)
    
    # Default shadow and highlight colors if not provided
    shadow_color = shadow_color or tuple(max(0, c - 30) for c in brick_color)
    highlight_color = highlight_color or tuple(min(255, c + 30) for c in brick_color)
    
    # Calculate the size of bricks and mortar
    brick_width = (size[0] - (brick_count_x - 1) * mortar_size) // brick_count_x
    brick_height = (size[1] - (brick_count_y - 1) * mortar_size) // brick_count_y
    
    for row in range(brick_count_y):
        y_top = row * (brick_height + mortar_size)
        y_bottom = y_top + brick_height
        
        # Draw the top mortar line only if it's the first row and top_mortar is True
        if top_mortar and row == 0:
            draw.rectangle([0, 0, size[0], mortar_size], fill=mortar_color)
        
        # Adjust the offset for alternating rows to create a staggered effect
        x_offset = -brick_width // 2 if row % 2 else 0
        
        for col in range(brick_count_x + 1):  # Include extra brick for the rightmost partial brick
            x_left = col * (brick_width + mortar_size) + x_offset
            
            # Ensure the left side of the brick is not negative
            if x_left < 0:
                x_left = 0
            
            x_right = x_left + brick_width
            
            # Draw vertical mortar between bricks
            if col > 0 or x_offset < 0:
                draw.rectangle([x_left - mortar_size, y_top, x_left, y_bottom], fill=mortar_color)
            
            # Ensure right side of brick does not exceed image width
            if x_right > size[0]:
                x_right = size[0]
            
            # Draw the brick
            draw.rectangle([x_left, y_top, x_right, y_bottom], fill=brick_color)
            
            # Draw shadows and highlights based on light direction
            if 'left' in light_direction:
                draw.rectangle([x_left, y_top, x_left + highlight_size, y_bottom], fill=highlight_color)
            if 'right' in light_direction:
                draw.rectangle([x_right - shadow_size, y_top, x_right, y_bottom], fill=shadow_color)
            if 'top' in light_direction:
                draw.rectangle([x_left, y_top, x_right, y_top + highlight_size], fill=highlight_color)
            if 'bottom' in light_direction:
                draw.rectangle([x_left, y_bottom - shadow_size, x_right, y_bottom], fill=shadow_color)
        
        # Draw the horizontal mortar line between rows
        if y_bottom + mortar_size < size[1]:
            draw.rectangle([0, y_bottom, size[0], y_bottom + mortar_size], fill=mortar_color)
    
    # Ensure the bottom mortar line is consistent
    if top_mortar and (brick_count_y * (brick_height + mortar_size) - mortar_size) < size[1]:
        draw.rectangle([0, size[1] - mortar_size, size[0], size[1]], fill=mortar_color)
    
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

gen_simple_tile(tile, tile_type='dirt', variance=True, simplex_enabled=True, simplex_scale=2, palette_var_thresh_A=4, palette_var_thresh_B=10, save=True, save_location='tile_gen')
# generate_brick_pattern(tile, size=(128, 128), brick_count_x=18, brick_count_y=30, brick_color=(150, 75, 50), mortar_color=(200, 200, 200), mortar_size=1, light_direction='top', top_mortar=True, save=True, save_location='tile_gen')
# generate_rand_template(64, 64, True)
