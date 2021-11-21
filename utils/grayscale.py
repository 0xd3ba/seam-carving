# grayscale.py -- Module containing the functions to convert a RGB image to grayscale

def to_grayscale(image_np):
    """
    image_np: 2D numpy array of shape (height, width, channels)
    Converts the image to grayscale image and returns it
    """
    assert len(image_np.shape) >= 2, f"Image must be 2D, provided {len(image_np.shape)}D instead"

    # If the number of dimensions are 2, then the image is already in grayscale
    if len(image_np.shape) == 2:
        return image_np

    # Convert it to grayscale using weighted sum of the channel intensities
    return (image_np[:, :, 0]*0.299 +
            image_np[:, :, 1]*0.587 +
            image_np[:, :, 2]*0.114
            )
