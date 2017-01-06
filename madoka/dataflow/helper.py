# -*- coding: utf-8 -*-
import numpy as np

__all__ = []


def apply_function_on_array(f, input_data):
    """Apply a function on input data.

    This method will apply a function on the input data.  If the input data
    is 1-d, it will expand the data to 2-d before feeding into the function,
    and then squeeze the output data back to 1-d if possible.

    Parameters
    ----------
    f : (np.ndarray) -> np.ndarray
        The function that will be applied to input data.

    input_data : np.ndarray
        The input data.

    Returns
    -------
    np.ndarray
    """
    # expand the input data to 2-d if it is 1-d
    if len(input_data.shape) == 1:
        input_data = input_data.reshape([-1, 1])
        ret = f(input_data)
        # revert back to 1-d if necessary.
        if len(ret.shape) == 2 and ret.shape[1] == 1:
            ret = ret.reshape([-1])
    else:
        ret = f(input_data)
    return ret


def detect_output_shape_dtype(f, input_shape, input_dtype):
    """Detect the output shape and data type after a function is applied.

    This method will attempt to detect the output shape and data type after
    the specified function as been applied on input data.

    Parameters
    ----------
    f : (np.ndarray) -> np.ndarray
        The function that will be applied to input data.

    input_shape : tuple[int]
        Input data shape.

    input_dtype : np.dtype
        Input data type.

    Returns
    -------
    (tuple[int], np.dtype)
    """
    # Scikit-Learn transformers do not allow 1-dimensional input data
    # (i.e., each input data point be 0-dimensional).
    # Thus we need to expand such 1-d input data to 2-d.
    input_data = (np.random.random(size=(1,) + tuple(input_shape)).
                 astype(dtype=input_dtype))
    output_data = apply_function_on_array(f, input_data)
    return (output_data.shape[1:], output_data.dtype)
