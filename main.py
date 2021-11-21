# main.py -- Entry point of the seam-carving algorithm

import os
import sys
import argparse
import numpy as np
from PIL import Image

# Import the algorithm functions and miscellaneous utility functions
from algorithms.seam_carving import seam_carving
from algorithms.column import column_removal
from algorithms.pixel import pixel_removal
from utils.misc import *


# Arguments to the CLI argument parser
# Each entry is of form (argument_name, type, required_flag, help_string)
CLI_ARGUMENTS = [
    ('--path', str, True, 'The path to the image file'),
    ('--dw', int, True, 'Amount of pixels by which image width will be changed'),
    ('--dh', int, True, 'Amount of pixels by which image height will be changed')
]

#################################################
# Define the ordering of seam-removal when image
# needs to be rescaled in both directions
# Default priority is to process the width of the
# image first
WIDTH_FIRST = True

# Result directory (Creates one if not created
# already)
RESULT_DIR = 'results'
#################################################


# Helper function to parse the CLI arguments and return them
def parse_args():
    arg_parser = argparse.ArgumentParser()

    # Add the arguments to the argument parser
    for arg, arg_type, req, help_str in CLI_ARGUMENTS:
        arg_parser.add_argument(arg, type=arg_type, required=req, help=help_str)

    parsed_args = arg_parser.parse_args(sys.argv[1:])           # Parse the arguments
    return parsed_args.path, parsed_args.dw, parsed_args.dh


if __name__ == '__main__':
    img_path, dw, dh = parse_args()

    # Try loading the image
    # Exit if the image cannot be loaded
    image = None
    image_name = img_path.split(os.path.sep)[-1].split('.')[0]

    try:
        image = Image.open(img_path)
        image = np.array(image)         # Image is of shape (height, width, channels)
    except FileNotFoundError:
        print(f'[ERROR]: Cannot open image-file {img_path}')
        sys.exit(-1)

    # Alright, image has been loaded
    # Apply each algorithm and save the results
    seam_carved_result = seam_carving(image, dw, dh, WIDTH_FIRST)
    save_image(result_dir=RESULT_DIR, img_matrix=seam_carved_result, img_name=image_name, algo='seam_carving', dw=dw, dh=dh)

    # NOTE: If there is UPSCALING involved, don't run the below algorithms
    #       as they have been used only for downscaling, as shown in the paper
    if not (dh > 0 or dw > 0):
        column_removed_result = column_removal(image, dw, dh, WIDTH_FIRST)
        pixel_removed_result = pixel_removal(image, dw, dh, WIDTH_FIRST)

        # Save the results
        save_image(result_dir=RESULT_DIR, img_matrix=column_removed_result, img_name=image_name, algo='column', dw=dw, dh=dh)
        save_image(result_dir=RESULT_DIR, img_matrix=pixel_removed_result, img_name=image_name, algo='pixel', dw=dw, dh=dh)

