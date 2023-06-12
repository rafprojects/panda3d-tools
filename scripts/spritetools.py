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


def remove_background(sprite, background_color):
    # Create a mask to remove the background color
    mask = Image.new("L", sprite.size, 0)
    # this creates a new image with the specified mode and size
    # the "L" mode is for 8-bit pixels, black and white
    for i in range(sprite.size[0]):
        for j in range(sprite.size[1]):
            if sprite.getpixel((i, j)) != background_color:
                # getpixel() returns the pixel value at the specified coordinates
                # in this case we are avoiding the background color
                # putpixel() sets the pixel value at the specified coordinates
                # 255 is the maximum value for a pixel, which is white
                mask.putpixel((i, j), 255)
    return mask


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


# detect_sprites("sprites/The Guardian Alyssa.png", "TGA.csv", (0, 0, 38))
# 260000 - input must be in BGR values i.e. (0, 0, 38)

# sprite_look("sprites/The Guardian Alyssa.png", "TGA.csv", (38, 0, 0))

sprite_output("sprites/The Guardian Alyssa.png", "TGA.csv", "TGA", "output", (38, 0, 0))