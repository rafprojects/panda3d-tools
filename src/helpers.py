import colorsys
import noise
import matplotlib.pyplot as plt
import numpy as np

def to_hsv(r, g, b):
    '''Returns h, s, v'''
    return colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)


def to_rgb(h, s, v):
    '''Returns r, g, b. Scaled back to 255 value.'''
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def adjust_hsv(hsv_value, percent_change):
    '''Alter a single hsv value by percent_change, in decimal'''
    return hsv_value * percent_change


def get_perlin_sprite():
    '''Generate a perlin noise tile based on 2 colors. TESTING function'''
    # Define a color palette with two colors
    palette = [(255, 50, 50), (25, 200, 80)]

    # Create an image
    width = 100
    height = 100
    img = np.zeros((height, width, 3))

    # Generate noise-based color for each pixel
    for i in range(height):
        for j in range(width):
            # Get Perlin noise value for this pixel (ranges from -1 to 1)
            n = noise.pnoise2(i/10, j/10)
            
            # Scale Perlin noise value to range from 0 to 1
            n = (n + 1) / 2.0

            # Use noise value to interpolate between colors in the palette
            r = (1 - n) * palette[0][0] + n * palette[1][0]
            g = (1 - n) * palette[0][1] + n * palette[1][1]
            b = (1 - n) * palette[0][2] + n * palette[1][2]

            # Set the pixel's color
            img[i, j] = [r, g, b]

    # Display the image
    plt.imshow(img/255)
    plt.show()
    
get_perlin_sprite()