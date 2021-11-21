# misc.py -- Module containing miscellaneous functions

import os
import pathlib
import numpy as np
from PIL import Image


def display_image(img_matrix):
    """ Displays the image on a new window from the given image matrix """

    # NOTE: Need to ensure that the matrix contains 1-byte elements
    Image._show(Image.fromarray(img_matrix.astype(np.uint8)))


def save_image(result_dir, img_matrix, img_name, algo, dw, dh):
    """ Saves the image to the result directory (creates one, if not created already) """

    # Create the result directory, if not present already
    # NOTE: Exceptions are not handled, assumption is made that directory path is accessible to user
    result_dir_path = pathlib.Path(result_dir)
    if result_dir not in os.listdir():
        os.mkdir(result_dir_path)

    # Create the image name, which is of format
    # {image_name}_{algorithm_used}_{change_in_width}_{change_in_height}
    img_name = f'{img_name}_{algo}_{dw}_{dh}.png'

    # Convert the image matrix into appropriate PIL image object
    pil_image = Image.fromarray(img_matrix.astype(np.uint8))
    pil_image.save(result_dir_path / img_name)