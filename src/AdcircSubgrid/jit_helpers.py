import numpy as np
from numba import njit

"""
This file contains jitted versions of numpy functions
which can not be used within other functions that are
jitted themselves
"""


@njit
def nan_mean_jit(arr: np.ndarray) -> np.ndarray:
    """
    Jitted version of the numpy nanmean function that
    uses a three-dimensional array and reduces to a one
    dimensional array. The jitted version of the builtin
    numpy version cannot take the axis argument, hence the
    need for this one

    Args:
        arr: A three-dimensional numpy array

    Returns:
        A one dimensional numpy array with the mean of axis (1, 2)
    """
    out = np.zeros(arr.shape[0])
    for i in range(arr.shape[0]):
        out[i] = np.nanmean(arr[i, :, :])
    return out


@njit
def nan_sum_jit(arr: np.ndarray) -> np.ndarray:
    """
    Jitted version of the numpy nansum function that
    uses a three-dimensional array and reduces to a one
    dimensional array. The jitted version of the builtin
    numpy version cannot take the axis argument, hence the
    need for this one

    Args:
        arr: A three-dimensional numpy array

    Returns:
        A one dimensional numpy array with the sum of axis (1, 2)
    """
    out = np.zeros(arr.shape[0])
    for i in range(arr.shape[0]):
        out[i] = np.nansum(arr[i, :, :])
    return out


@njit
def reciprocal_jit(arr: np.ndarray) -> np.ndarray:
    """
    Jitted version of the numpy reciprocal function
    that tests for 0

    Args:
        arr: A one-dimensional numpy array

    Returns:
        A one dimensional numpy array with the reciprocal
    """
    out = np.zeros(arr.shape[0])
    for i in range(arr.shape[0]):
        if arr[i] != 0:
            out[i] = 1 / arr[i]
    return out
