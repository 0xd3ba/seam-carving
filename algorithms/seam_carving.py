# seam_carving.py -- Module containing the functions for seam-carving algorithm

import tqdm
import numpy as np
from utils.grayscale import to_grayscale
from utils.energy import find_energies

#################################################
# Seam insertion parameter -- Naive/Optimized
# Naive version causes artifacts
# Optimized version minimizes the artifacts
USE_OPTIMIZED_SEAM_INSERTION = True
SEAM_INSERTION_FACTOR = 4
#################################################


def find_cost_matrix(img_matrix):
    """
    img_matrix: 2D numpy array of shape (height, width, channels)

    Returns the cost and direction matrices for the given image matrix
    """
    # Extact the number of rows and columns of the image
    rows, cols, _ = img_matrix.shape

    gray_img_matrix = to_grayscale(img_matrix)
    energy_matrix = find_energies(gray_img_matrix)

    cost_matrix = np.zeros(shape=(rows, cols))                      # Define the cost matrix, initially with zeros
    dir_matrix = np.zeros(shape=(rows, cols), dtype=np.int32)       # Direction matrix to note the seam direction
    cost_matrix[0, :] = energy_matrix[0, :]                         # Base-case, initial cost

    # Direction vector for detecting which column to move
    # NOTE: If the ordering is changed, make sure to change the ordering in connected_costs as well
    col_dir_vector = [0, -1, 1]

    # Calculate the DP matrix of the cost
    # Start from the second row
    for i in range(1, rows):
        for j in range(0, cols):
            top_col_idx = j
            top_left_col_idx = max(0, j-1)
            top_right_col_idx = min(cols-1, j+1)

            connected_costs = [cost_matrix[i-1, top_col_idx],
                               cost_matrix[i-1, top_left_col_idx],
                               cost_matrix[i-1, top_right_col_idx]]

            min_cost = np.min(connected_costs)
            col_direction = col_dir_vector[np.argmin(connected_costs)]

            cost_matrix[i, j] = energy_matrix[i, j] + min_cost
            dir_matrix[i, j] = col_direction

    return cost_matrix, dir_matrix


def remove_seam(img_matrix, cost_matrix, dir_matrix):
    """
    img_matrix:   2D numpy array of shape (height, width, channels)
    cost_matrix:  2D numpy array of shape (height, width)
    dir_matrix:   2D numpy array of shape (height, width)

    Removes a SINGLE seam (top to bottom) and returns the new resulting matrix

    NOTE: Due to how the cost matrix is calculated, need to start from the bottom, i.e.
          from the last row
    """
    rows, cols, channels = img_matrix.shape
    new_img_matrix = np.zeros(shape=(rows, cols-1, channels))

    curr_col = np.argmin(cost_matrix[rows-1, :])    # The location of the first column to remove

    # Now remove corresponding element of each column specified
    # Start from the last row and move to the top
    for i in range(rows-1, -1, -1):

        # Copy all the entries from beginning to the column just before the entry to remove
        new_img_matrix[i, 0:curr_col, :] = img_matrix[i, 0:curr_col, :]

        # Copy all the entries from the column just next to the removed column
        if curr_col + 1 <= cols - 1:
            new_img_matrix[i, curr_col:, :] = img_matrix[i, curr_col+1:, :]

        # Move to the next column
        curr_col = curr_col + dir_matrix[i, curr_col]

    return new_img_matrix


def add_seam_naive(img_matrix, cost_matrix, dir_matrix):
    """
    img_matrix:   2D numpy array of shape (height, width, channels)
    cost_matrix:  2D numpy array of shape (height, width)
    dir_matrix:   2D numpy array of shape (height, width)

    Adds a SINGLE seam (top to bottom) and returns the new resulting matrix
    Here's the idea behind the algorithm, as proposed in the paper:
        If we need to add a seam to the image, think of it as the seam
        that was REMOVED by seam-carving in the past. So the optimal seam to remove
        at the moment, would have (most-likely) been the second optimal seam to remove
        in the past. So take this seam and average the intensity values between its left/right
        neighbours (on seam insertion to increase width) or top/bottom neighbouts (on seam insertion to
        increase height)

    NOTE: Due to how the cost matrix is calculated, need to start from the bottom, i.e.
          from the last row
    """
    rows, cols, channels = img_matrix.shape
    new_img_matrix = np.zeros(shape=(rows, cols+1, channels))

    curr_col = np.argmin(cost_matrix[rows-1, :])    # The location of the first column to remove

    # Now add the corresponding element next to each column specified
    # Start from the last row and move to the top
    for i in range(rows-1, -1, -1):

        # Copy all the entries from beginning to the column to the entry that would have been removed
        new_img_matrix[i, 0:curr_col+1, :] = img_matrix[i, 0:curr_col+1, :]

        # Copy all the entries from the column just next to the entry that would have been removed
        if curr_col + 1 <= cols - 1:
            new_img_matrix[i, curr_col+2:, :] = img_matrix[i, curr_col+1:, :]

        # Now add the intensity of the new seam
        # The position is at curr_col+1 in the new image
        new_img_matrix[i, curr_col+1, :] = img_matrix[i, curr_col, :]

        # Average the intensities, if the right neighbour is valid
        if curr_col+1 <= cols - 1:
            new_img_matrix[i, curr_col+1, :] += img_matrix[i, curr_col+1, :]
            new_img_matrix[i, curr_col+1, :] = new_img_matrix[i, curr_col+1, :] // 2

        # Move to the next column
        curr_col = curr_col + dir_matrix[i, curr_col]

    return new_img_matrix


def add_seam_optimized(img_matrix, cost_matrix, dir_matrix, process_seams):
    """
    img_matrix:   2D numpy array of shape (height, width, channels)
    cost_matrix:  2D numpy array of shape (height, width)
    dir_matrix:   2D numpy array of shape (height, width)

    Adds a SINGLE seam (top to bottom) and returns the new resulting matrix
    This is the optimized version that reduces the visible artifacts caused by naive addition
    It adds process_seams at once

    NOTE: Due to how the cost matrix is calculated, need to start from the bottom, i.e.
          from the last row
    """
    assert process_seams > 0, f"[BUG ALERT]: Detected {process_seams} in seam insertion"

    rows, cols, channels = img_matrix.shape
    final_new_img_matrix = None

    # Start processing the seams
    for _ in range(process_seams):
        new_img_matrix = np.zeros(shape=(rows, cols + 1, channels))

        curr_col = np.argmin(cost_matrix[rows - 1, :])  # The location of the first column to remove
        cost_matrix[rows-1, curr_col] = np.inf          # To ensure this will not be picked the next time

        # Now add the corresponding element next to each column specified
        # Start from the last row and move to the top
        for i in range(rows - 1, -1, -1):

            # Copy all the entries from beginning to the column to the entry that would have been removed
            new_img_matrix[i, 0:curr_col + 1, :] = img_matrix[i, 0:curr_col + 1, :]

            # Copy all the entries from the column just next to the entry that would have been removed
            if curr_col + 1 <= cols - 1:
                new_img_matrix[i, curr_col + 2:, :] = img_matrix[i, curr_col + 1:, :]

            # Now add the intensity of the new seam
            # The position is at curr_col+1 in the new image
            new_img_matrix[i, curr_col + 1, :] = img_matrix[i, curr_col, :]

            # Average the intensities, if the right neighbour is valid
            if curr_col + 1 <= cols - 1:
                new_img_matrix[i, curr_col + 1, :] += img_matrix[i, curr_col + 1, :]
                new_img_matrix[i, curr_col + 1, :] = new_img_matrix[i, curr_col + 1, :] // 2

            # Move to the next column
            cost_matrix[i, curr_col] = np.inf   # Ensure that this will not be picked next time
            curr_col = curr_col + dir_matrix[i, curr_col]

        rows, cols, channels = new_img_matrix.shape
        final_new_img_matrix = new_img_matrix
        img_matrix = new_img_matrix

    return final_new_img_matrix


def process_width(img_matrix, dw):
    """ Transforming the width of the image by seam-carving """

    processed_img_matrix = img_matrix.copy()
    seams_left_to_add = dw  # Ignored, if doing seam removal

    # Repeat for number of pixels to process
    for _ in tqdm.tqdm(range(abs(dw))):
        cost_matrix, dir_matrix = find_cost_matrix(processed_img_matrix)

        # Need to reduce the width of the image
        if dw < 0:
            processed_img_matrix = remove_seam(processed_img_matrix, cost_matrix, dir_matrix)

        # Need to increase the width of the image
        # Depending on the version to use, perform the seam addition accordingly
        else:
            if USE_OPTIMIZED_SEAM_INSERTION:
                process_seams = seams_left_to_add // SEAM_INSERTION_FACTOR   # How many seams to process at once
                process_seams = process_seams if process_seams > 0 else seams_left_to_add % SEAM_INSERTION_FACTOR

                processed_img_matrix = add_seam_optimized(processed_img_matrix, cost_matrix, dir_matrix, process_seams)

                seams_left_to_add -= process_seams
                if seams_left_to_add <= 0:
                    break

            # Use the naive seam-addition, that causes visible artifacts
            else:
                processed_img_matrix = add_seam_naive(processed_img_matrix, cost_matrix, dir_matrix)

    return processed_img_matrix


def process_height(img_matrix, dh):
    """ Transforms the height of the image by seam-carving """
    # Rotating the image by 90 degrees, transforming it width-wise and then transforming the result
    # will give the result we need
    processed_img_matrix = img_matrix.transpose(1, 0, 2)
    processed_img_matrix = process_width(processed_img_matrix, dh)
    return processed_img_matrix.transpose(1, 0, 2)


#############################################################
# The main seam-carving algorithm
#############################################################
def seam_carving(img_matrix, dw, dh, width_first):
    """ Applies seam-carving to the resulting image """

    if width_first:
        result_img = process_width(img_matrix, dw)
        result_img = process_height(result_img, dh)
    else:
        result_img = process_height(img_matrix, dh)
        result_img = process_width(result_img, dw)

    return result_img
