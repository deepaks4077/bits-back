"""
This file contains helper functions for interaction between numpy and torch
"""

import torch
import numpy as np

def tensor_to_array(tensor):
    """
    Converts a tuple or a single tensor object to a numpy array
    """
    if type(tensor) is tuple:
        return tuple(tensor_to_array(t) for t in tensor)
    elif type(tensor) is np.ndarray:
        return tensor
    else:
        return tensor.detach().numpy()

def array_to_tensor(array):
    """
    Convert a tuple or a single array (numpy) to a torch tensor.
    """
    if type(array) is tuple:
        return tuple(array_to_tensor(t) for t in array)
    elif type(array) is torch.Tensor:
        return array
    else:
        return torch.from_numpy(array)

def torch_to_numpy_function(torch_fn):
    """
    Takes a torch function and returns its numpy equivalent
    """
    def numpy_fn(*args, **kwargs):
        torch_arguments = array_to_tensor(args)
        return tensor_to_array(torch_fn(*torch_arguments, **kwargs))
    return numpy_fn
