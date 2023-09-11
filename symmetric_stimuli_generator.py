import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage.transform import resize
import random
import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import sys
import functools
import scipy
from scipy.stats.stats import pearsonr   
import os
import random
import torch
from PIL import Image
from torchvision import transforms
import seaborn as sns
import matplotlib.pyplot as plt
import math
import time
from scipy.signal import convolve2d
import cProfile

def randomize_color_blocks(image, block_size):
    """
    Divide an image into blocks of a specified size and randomize the positions of the blocks.

    Parameters:
    - image: a 2D numpy array representing the image.
    - block_size: a tuple specifying the size of the blocks.

    Returns:
    - The randomized image.
    """
    image_size = image.shape

    num_blocks = (image_size[0] // block_size[0], image_size[1] // block_size[1])

    blocks = [np.hsplit(row, num_blocks[1]) for row in np.vsplit(image, num_blocks[0])]

    blocks = [block for row in blocks for block in row]

    np.random.shuffle(blocks)

    randomized_image = np.vstack([np.hstack(blocks[i:i+num_blocks[1]]) for i in range(0, len(blocks), num_blocks[1])])

    return randomized_image

def create_circular_kernel(size):
    """Create a circular kernel of a given size."""
    kernel = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            if (i - center)**2 + (j - center)**2 <= center**2:
                kernel[i, j] = 1
    return kernel

def apply_kernel(matrix, kernel, value, i, j):
    """Apply a kernel to a specific position in a matrix with a given value."""
    half_size = kernel.shape[0] // 2
    for ki in range(-half_size, half_size + 1):
        for kj in range(-half_size, half_size + 1):
            if 0 <= i + ki < matrix.shape[0] and 0 <= j + kj < matrix.shape[1] and kernel[ki + half_size, kj + half_size]:
                matrix[i + ki, j + kj] = value
    return matrix

def generate_symmetric_color_image(target_size, supersample_factor, density=0.00025, kernel_size=29, min_distance=3, adjustment_factor=1.0,
                            num_color_patterns=2, num_axes=1, lum_value=50, max_l_channel=25):
    """
    Generate a high-resolution image with given parameters.
    
    Parameters:
    - target_size: the target image size.
    - supersample_factor: factor to increase resolution before downsampling.
    - density: probability for placing a dot.
    - kernel_size: size of the kernel used for placing dots.
    - min_distance: minimum distance between dots.
    - adjustment_factor: factor to adjust the color.
    
    Returns:
    - Symmetric and Antisymmetric versions of the generated image.
    """
    
    def get_channel_colors(num_color_patterns):
        # Maximum and minimum possible values in LAB colorspace for A and B channels
        
        if num_color_patterns == 0:
            step_size = 0
            min_value = 0
            max_value = 0
            colors = [min_value, max_value]
        else:
            min_value = -128
            max_value = 127
            # Calculate the step size based on the number of color patterns
            step_size = (max_value - min_value) // (num_color_patterns // 2)
            
        
            # Generate the dynamic colors based on the step size
            colors = list(range(min_value, min_value + step_size * (num_color_patterns // 2), step_size))
        
        # Distribute these colors to the two channels
        possible_channel_a_colors = colors
        if num_color_patterns > 2:
            possible_channel_b_colors = colors
        else:
            possible_channel_b_colors = []
        
        return possible_channel_a_colors, possible_channel_b_colors
    
    possible_channel_a_colors, possible_channel_b_colors = get_channel_colors(num_color_patterns)
    
    supersample_size = (target_size[0]*supersample_factor, target_size[1]*supersample_factor)
    margin = kernel_size
    
    # Create the dot map for the right side
    if num_axes == 1:
        dot_map_height = supersample_size[0]
        dot_map_width = supersample_size[1] // 2
    elif num_axes == 2:
        dot_map_height = supersample_size[0] // 2
        dot_map_width = supersample_size[1] // 2
    elif num_axes == 4:
        dot_map_height = supersample_size[0] // 4
        dot_map_width = supersample_size[0] // 4

    dot_map = np.zeros((dot_map_height, dot_map_width))
    dot_map[margin:-margin, margin:-margin] = np.random.choice([0, 1], size=(dot_map_height-2*margin, (dot_map_width)-2*margin), p=[1-density, density])
    
    # Iterative method for high densities and low min_distance
    for i in range(dot_map_height):
        for j in range(dot_map_width):
            if dot_map[i, j] == 1:
                dot_map[max(0, i-min_distance):min(dot_map_height, i+min_distance+1), 
                        max(0, j-min_distance):min(dot_map_width, j+min_distance+1)] = 0
                dot_map[i, j] = 1
                
    l_channel = np.ones((dot_map_height, dot_map_width)) * lum_value
    a_channel = np.zeros((dot_map_height, dot_map_width))
    b_channel = np.zeros((dot_map_height, dot_map_width))
    kernel = create_circular_kernel(kernel_size)
    dots = []
    
    len_possible_channel_b_colors = len(possible_channel_b_colors)
    # Apply color values based on the dot map
    for i in range(dot_map_height):
        for j in range(dot_map_width):
            if dot_map[i, j] == 1:
                channel = 'a'
                if len_possible_channel_b_colors == 0: # Base case
                    color_val = possible_channel_a_colors
                else: # First choose randomly between the two possible channels
                    if np.random.rand() > 0.5:
                        color_val = possible_channel_a_colors
                    else:
                        color_val = possible_channel_b_colors
                        channel = 'b' 
                
                # Now based on the channel selected, choose a random subset of color values (this value will be negative or positive depending on extremum)
                color_val = random.choice(color_val)
                
                # Choose extremum
                color_val = -(color_val+1) if np.random.rand() > 0.5 else color_val

                if num_color_patterns == 0:
                    poss_lum_values = [0, max_l_channel]
                    poss_lum_value = random.choice(poss_lum_values)
                elif num_color_patterns >= 2:
                    poss_lum_value = lum_value
                    
                
                if channel == 'a':
                    a_channel = apply_kernel(a_channel, kernel, color_val, i, j)
                elif channel == 'b':
                    b_channel = apply_kernel(b_channel, kernel, color_val, i, j)
                    
                l_channel = apply_kernel(l_channel, kernel, poss_lum_value, i, j)
                dots.append({'channel':channel,'i':i, 'j':j})
    
    right_lab = np.stack([l_channel, a_channel, b_channel], axis=-1)


    
    # Create symmetric and antisymmetric versions of the left side, extend to num_axes = 1, 2 and 4
    def create_symm_and_anti_vers(a_channel, b_channel, l_channel, dots, adjustment_factor=1.0):
        left_a_sym = np.fliplr(a_channel)
        left_a_anti = -np.fliplr(a_channel)
        left_b_sym = np.fliplr(b_channel)
        left_b_anti = -np.fliplr(b_channel)
        left_l = np.fliplr(l_channel)
        
        # Adjust the colors based on the adjustment factor
        to_adjust = int(len(dots) * (1-adjustment_factor))
        random.shuffle(dots)
        
        for i in range(to_adjust):
            channel, x, y = dots[i]
            y_mirror = left_a_sym.shape[1] - y - 1
            
            if channel == 'a':
                left_a_sym = apply_kernel(left_a_sym, kernel, -a_channel[x, y], x, y_mirror)
                left_a_anti = apply_kernel(left_a_anti, kernel, a_channel[x, y], x, y_mirror)
            elif channel == 'b':
                left_b_sym = apply_kernel(left_b_sym, kernel, -b_channel[x, y], x, y_mirror)
                left_b_anti = apply_kernel(left_b_anti, kernel, b_channel[x, y], x, y_mirror)
        
        part_lab_symm = np.stack([left_l, left_a_sym, left_b_sym], axis=-1)
        part_lab_anti = np.stack([left_l, left_a_anti, left_b_anti], axis=-1)
        
        return part_lab_symm, part_lab_anti

    if num_axes == 1:
        left_lab_sym, left_lab_anti = create_symm_and_anti_vers(a_channel, b_channel, l_channel, dots, adjustment_factor=adjustment_factor)
        image_sym = np.concatenate([left_lab_sym, right_lab], axis=1)
        image_anti = np.concatenate([left_lab_anti, right_lab], axis=1)
    elif num_axes == 2:
        bottom_left_quarter_sym, bottom_left_quarter_anti = create_symm_and_anti_vers(a_channel, b_channel, l_channel, dots, adjustment_factor=adjustment_factor)
        top_right_quarter_sym, top_right_quarter_anti = create_symm_and_anti_vers(a_channel, b_channel, l_channel, dots, adjustment_factor=adjustment_factor)
        top_right_quarter_sym, top_right_quarter_anti = np.flipud(np.fliplr(top_right_quarter_sym)), np.flipud(np.fliplr(top_right_quarter_anti))
        
        a_channel_top_right = top_right_quarter_sym[:, :, 1]
        b_channel_top_right = top_right_quarter_sym[:, :, 2]
        l_channel_top_right = top_right_quarter_sym[:, :, 0]
        
        top_left_quarter_sym, top_left_quarter_anti = create_symm_and_anti_vers(a_channel_top_right, b_channel_top_right,
                                                                                 l_channel_top_right, dots, adjustment_factor=adjustment_factor)
        
        top_half_sym = np.concatenate((top_left_quarter_sym, top_right_quarter_sym), axis=1)
        bottom_half_sym = np.concatenate((bottom_left_quarter_sym, right_lab), axis=1)
        image_sym = np.concatenate((top_half_sym, bottom_half_sym), axis=0)
        
        top_half_anti = np.concatenate((top_left_quarter_anti, top_right_quarter_anti), axis=1)
        bottom_half_anti = np.concatenate((bottom_left_quarter_anti, right_lab), axis=1)
        image_anti = np.concatenate((top_half_anti, bottom_half_anti), axis=0)
    elif num_axes == 4:
        # Generate one sixteenth of the image with Gaussian noise
        # sixteenth is left_lab_sym, in this no antisymmetric filter is created. Maybe this is useful for other experiments
        # TODO - Add left_lab_anti to this case
        sixteenth, _ = create_symm_and_anti_vers(a_channel, b_channel, l_channel, dots, adjustment_factor=adjustment_factor)
        
        # Create the other sixteenths as the mirror symmetric of the first sixteenth
        eighths = [np.concatenate((np.fliplr(sixteenth), sixteenth), axis=1) for _ in range(2)]
        quarters = [np.concatenate((np.flipud(eighth), eighth), axis=0) for eighth in eighths]
        halves = [np.concatenate((np.fliplr(quarter), quarter), axis=1) for quarter in quarters]
        image_sym = np.concatenate((np.flipud(halves[0]), halves[1]), axis=0)
        
        # TODO - Change image_sym to image_anti when left_lab_anti is added
        image_anti = image_sym

    # Resize to the target size
    image_sym = resize(image_sym, (target_size[0], target_size[1]), anti_aliasing=False, mode='reflect', order=3)
    image_anti = resize(image_anti, (target_size[0], target_size[1]), anti_aliasing=False, mode='reflect', order=3)

    
    # Convert LAB to RGB
    image_sym = (color.lab2rgb(image_sym) * 255).astype(np.uint8)
    image_anti = (color.lab2rgb(image_anti) * 255).astype(np.uint8)

    return image_sym, image_anti


# generate_symmetric_color_image(target_size, supersample_factor, density=0.00025, kernel_size=29, min_distance=3, adjustment_factor=1.0,
#                             num_color_patterns=2, num_axes=1, lum_value=50, max_l_channel=25):

num_colors = 0
density = 1
max_lum = 50
min_distance = 3
supersample_factor = 3


image_sym_1,  _ = generate_symmetric_color_image((224, 224), supersample_factor=supersample_factor, density=density, kernel_size=1, min_distance=min_distance, adjustment_factor=1,
                                                num_color_patterns=num_colors, num_axes=4, lum_value=50, max_l_channel=max_lum)


### Show image_sym_1, image_sym_2 and image_sym_4 in subplots
image_sym_1 = Image.fromarray(image_sym_1)
image_sym_1.save(f'./symmetry_images/symmetry_1_axis_{num_colors}_colors_{max_lum}_\
    lum_{density}_density_{min_distance}_spacing_{supersample_factor}_supersample_factor.png')

sys.exit()


def save_symmetric_color_images(root_folder='./symmetry_images', num_axes=2, num_colors=2, num_images=1000):
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    
    for num_image in range(num_images):
        image_sym, image_anti = generate_symmetric_color_image((224, 224), 5, density=1, kernel_size=1, min_distance=0, adjustment_factor=1,
                                                        num_color_patterns=num_colors, num_axes=num_axes)
        null_block_randomized = randomize_color_blocks(image_sym, (1,1))
        # Convert the randomized image back to an Image object
        null_block_randomized_img = Image.fromarray(null_block_randomized)

        # Save on folder ./symmetry_images/symmetry image_sym, image_anti, and null_block_randomized
        image_sym_path = f'{num_image}_sym.jpg'
        image_anti_path = f'{num_image}_anti.jpg'
        randomized_path = f'{num_image}_randomized.jpg'

        image_sym = Image.fromarray(image_sym)
        image_anti = Image.fromarray(image_anti)

        image_sym.save(f'{root_folder}/{image_sym_path}')
        image_anti.save(f'{root_folder}/{image_anti_path}')
        null_block_randomized_img.save(f'{root_folder}/{randomized_path}')

        if num_image % 100 == 0:
            print(f"Saved {num_image} images")


# for num_lums in range(6, 127, 2):
#     save_symmetric_color_images(root_folder=f'./symmetry_images/symmetry_1_axis_{num_colors}_colors', num_axes=1, num_colors=num_colors, num_images=1000)
#     save_symmetric_color_images(root_folder=f'./symmetry_images/symmetry_2_axis_{num_colors}_colors', num_axes=2, num_colors=num_colors, num_images=1000)
#     save_symmetric_color_images(root_folder=f'./symmetry_images/symmetry_2_axis_{num_colors}_colors', num_axes=2, num_colors=num_colors, num_images=1000)
    
# sys.exit()