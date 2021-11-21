# column.py -- Module containing the algorithm to remove column with least energy

import tqdm
import numpy as np
from utils.grayscale import to_grayscale
from utils.energy import find_energies


def process_width(img_matrix, dw):
    """
    Removes the columns with the least energy
    NOTE: As done in the paper, this only REDUCES the scale of the image. Doesn't increase
          it, like Seam-carving algorithm
    """
    processed_img_matrix = img_matrix.copy()

    for _ in tqdm.tqdm(range(abs(dw))):
        gray_img_matrix = to_grayscale(processed_img_matrix)  # Convert to gray scale to calculate the energies
        energies = find_energies(gray_img_matrix)             # Find the energy matrix
        energies = energies.sum(axis=0)                       # Sum all rows

        # Pick the column with the least index
        least_col_idx = energies.argmin()

        # Now remove the column and update the image
        processed_img_matrix = np.delete(processed_img_matrix, least_col_idx, 1)

    return processed_img_matrix


def process_height(img_matrix, dh):
    """ Removes the rows with the least energy """
    processed_img_matrix = img_matrix.transpose(1, 0, 2)
    processed_img_matrix = process_width(processed_img_matrix, dh)
    return processed_img_matrix.transpose(1, 0, 2)


def column_removal(img_matrix, dw, dh, width_first):
    """ Algorithm that removes the column with the least energy """

    if width_first:
        result_img = process_width(img_matrix, dw)
        result_img = process_height(result_img, dh)
    else:
        result_img = process_height(img_matrix, dh)
        result_img = process_width(result_img, dw)

    return result_img
