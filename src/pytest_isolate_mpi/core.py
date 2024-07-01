"""Main module."""

import numpy as np


def sum_array(array: np.ndarray) -> float:
    """Adds all the numbers in an array.

    Args:
       array: The array to sum up.

    Returns:
       The sum of elements in the array.

    """
    return np.sum(array)
