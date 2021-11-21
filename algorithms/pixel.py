# column.py -- Module containing the algorithm to remove pixel in each row with least energy

import tqdm
import numpy as np
from utils.grayscale import to_grayscale
from utils.energy import find_energies


def process_width(img_matrix, dw):
    """ Removes the entries in rows with the least energy """
    processed_img_matrix = img_matrix.copy()

    for _ in tqdm.tqdm(range(abs(dw))):
        gray_img_matrix = to_grayscale(processed_img_matrix)  # Convert to gray scale to calculate the energies
        energies = find_energies(gray_img_matrix)             # Find the energy matrix

        # Create a mask to filter out the entries with least energy
        # Set the mask to false for those entries
        mask_matrix = np.ones_like(processed_img_matrix).astype(np.bool8)
        for i in range(mask_matrix.shape[0]):
            least_col_idx = energies[i].argmin()
            mask_matrix[i, least_col_idx, :] = False

        # Now remove the column and update the image
        # NOTE: Seems like masking with 3D mask leads to flattening the vector
        #       So need to reshape it back. Note that column will be one less because we have "removed" it
        rows, cols, channels = processed_img_matrix.shape
        processed_img_matrix = processed_img_matrix[mask_matrix].reshape(rows, cols-1, channels)

    return processed_img_matrix


def process_height(img_matrix, dh):
    """ Removes the entries in columns with the least energy """
    processed_img_matrix = img_matrix.transpose(1, 0, 2)
    processed_img_matrix = process_width(processed_img_matrix, dh)
    return processed_img_matrix.transpose(1, 0, 2)


def pixel_removal(img_matrix, dw, dh, width_first):
    """ Algorithm that removes the pixel in each row with the least energy """

    if width_first:
        result_img = process_width(img_matrix, dw)
        result_img = process_height(result_img, dh)
    else:
        result_img = process_height(img_matrix, dh)
        result_img = process_width(result_img, dw)

    return result_img


