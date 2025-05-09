from PIL import ImageDraw
import random
import math

import numpy as np
import matplotlib.pyplot as plt

from tilehelpers import palettesD, save_img, get_simplex_color, get_other_pixel_colors, light_directions
from tilegenerator import TileCanvasGenerator


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

def gen_simple_tile(img_obj, tile_type, variance=False, simplex_enabled=False, simplex_scale=None, 
                    palette_var_thresh_A=None, palette_var_thresh_B=None, save=False, save_location=None, draw_grass_blades=False, blade_direction=None, blade_uniformity=1.0, blade_thickness=1.0, blade_tallness=1.0, light_direction=(1,1), root_skew_range=(2,4), blade_curve_intensity=1.0):
    
    color_range = palettesD[tile_type]
    # print("COLOR RANGE: ", color_range)
    # print("COLOR RANGE LEN: ", len(color_range))
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
            
    if draw_grass_blades and tile_type in ['regular_grass', 'green_grass']:
        draw_grass_blades_on_tile(img_obj, blade_direction, blade_uniformity, blade_thickness, blade_tallness, light_direction, root_skew_range, blade_curve_intensity)
    
    if save:
        save_img(img=img_obj, save_location=save_location, img_name=tile_type)
    else:
        return img_obj



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






def draw_grass_blades_on_tile(img_obj, blade_direction=None, blade_uniformity=1.0, blade_thickness=1, blade_tallness=1.0, light_direction=(1, 1), root_skew_range=(2, 4), blade_curve_intensity=1.0):
    draw = ImageDraw.Draw(img_obj)
    width, height = img_obj.size
    light_x, light_y = light_directions[light_direction]

    for i in range(0, width, int(width / 10)):  # Adjust step for the density of blades
        for j in range(height - int(height / 4), height):  # Draw blades starting from the bottom quarter
            # Apply random skew to root position
            root_skew_x = random.randint(-root_skew_range[0], root_skew_range[0])
            root_skew_y = random.randint(-root_skew_range[1], root_skew_range[1])

            root_x = i + root_skew_x
            root_y = j + root_skew_y

            # Adjust blade height by tallness factor
            base_blade_height = int(height / 8)
            blade_height = random.randint(base_blade_height, int(height / 4)) * blade_tallness
            max_blade_width = max(1, int(blade_thickness))  # Ensure thickness is at least 1
            
            if blade_direction:
                angle = blade_direction + random.uniform(-10, 10) * (1.0 - blade_uniformity)
            else:
                angle = random.uniform(-30, 30) * (1.0 - blade_uniformity)
            
            # Calculate curved blade endpoints
            curve_intensity = random.uniform(0.1, 0.3) * blade_curve_intensity  # Adjust the curvature intensity here
            mid_x = root_x + int(blade_height * 0.5 * math.sin(math.radians(angle + curve_intensity)))
            mid_y = root_y - int(blade_height * 0.5 * (1 - curve_intensity))

            end_x = root_x + int(blade_height * math.sin(math.radians(angle)))
            end_y = root_y - blade_height

            # Determine shadow side by comparing blade direction with light direction
            shadow_offset = max(1, int(blade_thickness / 2))
            shadow_x = root_x - shadow_offset * light_x
            shadow_y = root_y - shadow_offset * light_y
            shadow_mid_x = mid_x - shadow_offset * light_x
            shadow_mid_y = mid_y - shadow_offset * light_y
            shadow_end_x = end_x - shadow_offset * light_x
            shadow_end_y = end_y - shadow_offset * light_y

            # Draw the shadow first with an offset
            shadow_color = (max(0, 34 - 70), max(0, 139 - 70), max(0, 34 - 70))  # Darken color for shadow
            draw.line([(shadow_x, shadow_y), (shadow_mid_x, shadow_mid_y), (shadow_end_x, shadow_end_y)], fill=shadow_color, width=max_blade_width)

            # Draw the main blade with gradient and slight color variations
            for k in range(max_blade_width):
                gradient_factor = k / max_blade_width  # Ranges from 0 at the root to 1 at the tip
                color_variation = random.randint(-10, 10)
                blade_color = (
                    max(0, min(34 + color_variation, 255)) * (1 - gradient_factor),
                    max(0, min(139 + color_variation, 255)) * (1 - gradient_factor),
                    max(0, min(34 + color_variation, 255)) * (1 - gradient_factor)
                )
                draw.line([(root_x + k, root_y), (mid_x + k, mid_y), (end_x + k, end_y)], fill=blade_color, width=max_blade_width - k)








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

gen_simple_tile(tile, tile_type='regular_grass', variance=True, simplex_enabled=True, simplex_scale=2, palette_var_thresh_A=4, palette_var_thresh_B=10, save=True, save_location='tile_gen', draw_grass_blades=True, blade_direction=4, blade_uniformity=3, blade_thickness=2, blade_tallness=2, light_direction='top-left', root_skew_range=(2, 4), blade_curve_intensity=2.0)


# generate_brick_pattern(tile, size=(128, 128), brick_count_x=18, brick_count_y=30, brick_color=(150, 75, 50), mortar_color=(200, 200, 200), mortar_size=1, light_direction='top', top_mortar=True, save=True, save_location='tile_gen')
# generate_rand_template(64, 64, True)
