import numpy as np
import math


def image_to_vector(image):
    """
    Args:
    image: numpy array of shape (length, height, depth)

    Returns:
     v: a vector of shape (length x height x depth, 1)
    """
    length, height, depth = image.shape
    return image.reshape((length * height * depth, 1))
def SAM(s1, s2):
    ## https://pysptools.sourceforge.io/_modules/pysptools/distance/dist.html#SAM
    """
    Computes the spectral angle mapper between two vectors (in radians).

    Parameters:
        s1: `numpy array`
            The first vector.

        s2: `numpy array`
            The second vector.

    Returns: `float`
            The angle between vectors s1 and s2 in radians.
    """
    # image1 = _normalize(images[0])
    # image2= _normalize(images[1])
    # test = SAM(image_to_vector(image1),image_to_vector(image2))
    # SAMValuesForRGB=[]
    # s1= s1.flatten()
    # s2 = s2.flatten()
    s1_norm = math.sqrt(np.dot(s1, s1))
    s2_norm = math.sqrt(np.dot(s2, s2))
    sum_s1_s2 = np.dot(s1, s2)
    # test = sum_s1_s2 / (s1_norm * s2_norm)
    angle = math.acos(sum_s1_s2 / (s1_norm * s2_norm))
    return angle