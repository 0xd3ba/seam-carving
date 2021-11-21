# energy.py -- Module containing the functions for calculating the energies

import numpy as np
from scipy.ndimage import sobel
from scipy.ndimage import laplace


def _apply_sobel(img_matrix):
    """
    Input: img_matrix(height, width) with type float32

    Convolves the image with sobel mask and returns the magnitude
    """
    dx = sobel(img_matrix, 1)
    dy = sobel(img_matrix, 0)

    grad_mag = np.hypot(dx, dy)         # Calculates sqrt(dx^2 + dy^2)
    grad_mag *= 255 / grad_mag.max()    # Normalize the gradient magnitudes

    return grad_mag


def _apply_laplacian(img_matrix):
    """
    Input: img_matrix(height, width) with type float32

    Convolves the image with Laplacian and returns the result
    """
    dx_dy = laplace(img_matrix)
    dx_dy *= 255 / dx_dy.max()          # Normalize the result

    return dx_dy


################################################################
# The energy function to use for calculating the "energies"
# of the given image. Change it accordingly
ENERGY_FUNCTION = _apply_sobel
################################################################


def find_energies(img_matrix):
    """
    img_matrix: 2D numpy array of shape (height, width), i.e. the image is grayscale

    Calculates the "energies", i.e. the digital gradients of the image (basically the edges)
    and returns the resulting matrix
    """
    energy_mat = ENERGY_FUNCTION(img_matrix)
    return energy_mat
