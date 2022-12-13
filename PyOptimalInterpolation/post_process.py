import numpy as np
import xarray as xr
from astropy.convolution import convolve, Gaussian2DKernel
from typing import Callable


class PostProcessModule():
    def __init__(self, method: Callable):
        self._postprocess = method
    
    def __call__(self, param_fields: xr.Dataset, postprocess_kwargs):
        for key in param_fields.keys():
            param = param_fields.data_vars[key].values
            new_param = self._postprocess(param, **postprocess_kwargs[key])
            param_fields[key] = (('y', 'x'), new_param)


# --- List of post processing methods ---

def clip(param: np.ndarray, vmin=0, vmax=1e10):
    """
    Apply clippingto a given hyperparameter field
    """
    param[param >= vmax] = vmax
    param[param <= vmin] = vmin
    return param


def smooth(param: np.ndarray, std=1):
    """
    Apply Gaussian smoothing to a given hyperparameter field
    """
    param = convolve(param, Gaussian2DKernel(x_stddev=std, y_stddev=std))
    return param


def clip_and_smooth(param: np.ndarray, vmin=0, vmax=1e10, std=1):
    """
    Apply clipping and Gaussian smoothing to a given hyperparameter field
    """
    param = clip(param, vmin, vmax)
    param = smooth(param, std)
    return param