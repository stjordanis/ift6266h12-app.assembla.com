"""
Input/Output functionality
"""
import numpy


def load_npy(path):
    """
    Function to load npy files. The single difference from `numpy.load` is
    that, if the loaded ndarray has dtype `object`, then the object is
    returned instead of the ndarray wrapped around it.
    """
    data = numpy.load(path)
    if str(data.dtype) == 'object':
        data = data.item()
    return data
