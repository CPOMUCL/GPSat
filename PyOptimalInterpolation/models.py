import inspect
import re
import os
import platform
import subprocess
import pandas as pd
import scipy
import gpflow
import numpy as np
import xarray as xr
import time
import pickle
import warnings

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.client import device_lib

from gpflow.utilities import set_trainable
from abc import ABC, abstractmethod
from astropy.convolution import convolve, Gaussian2DKernel
from typing import List, Dict

from PyOptimalInterpolation.decorators import timer


# ------- Base class ---------

class BaseGPRModel(ABC):
    def __init__(self,
                 data=None,
                 coords_col=None,
                 obs_col=None,
                 coords=None,
                 obs=None,
                 coords_scale=None,
                 obs_scale=None,
                 obs_mean=None,
                 # kernel=None,
                 # prior_mean=None,
                 verbose=True,
                 **kwargs):
        """
        """

        # --
        # data
        # --

        # assign data to model
        if data is not None:
            assert coords_col is not None, "data was provided, but coord_col was not"
            assert obs_col is not None, "data was provided, but obs_col was not"

            # require the columns for selecting data are not str
            if isinstance(coords_col, str):
                coords_col = [coords_col]
            if isinstance(obs_col, str):
                obs_col = [obs_col]

            # TODO: should data be copied?

            # select relevant data - as np.arrays
            # - taking values makes copy(?)
            self.obs = data.loc[:, obs_col].values
            self.coords = data.loc[:, coords_col].values

            # store the column names
            self.obs_col = obs_col
            self.coords_col = coords_col
        # otherwise expect to have coords and obs provided directly
        else:

            assert obs is not None, f"data is {data}, and so is obs: {obs}, provide either"
            assert coords is not None, f"data is {data}, and so is coords: {coords}, provide either"

            assert isinstance(obs, np.ndarray), "if obs is provided directly it must be an np.array"
            assert isinstance(coords, np.ndarray), "if obs is provided directly it must be an np.array"

            if len(obs.shape) == 1:
                print("obs is 1-d array, setting to 2-d")
                obs = obs[:, None]

            if len(coords.shape) == 1:
                print("coords is 1-d array, setting to 2-d")
                coords = coords[:, None]

            assert len(obs) == len(coords), "obs and coords lengths don't match "

            self.obs = obs
            self.coords = coords

            # if column 'names' not provide generate default values
            # - these could be np.arrays...
            if coords_col is None:
                coords_col = [_ for _ in range(self.coords.shape[1])]
            if obs_col is None:
                obs_col = [0]
            self.coords_col = coords_col
            self.obs_col = obs_col

        # nan check
        assert not np.isnan(self.coords).any(), "nans found in coords"
        assert not np.isnan(self.obs).any(), "nans found in obs"

        # observation mean - to be subtracted from observations

        # TODO: review where should de-meaning be done
        # remove mean of observations data?
        if obs_mean == "local":
            if verbose:
                print(f"setting obs_mean with mean of obs_col: {obs_col}")
            obs_mean = np.mean(self.obs, axis=0)
        else:
            obs_mean = np.array([0])[None, :]

        if isinstance(obs_mean, list):
            obs_mean = np.array(obs_mean)[None, :]
        elif isinstance(obs_mean, (int, float)):
            obs_mean = np.array([obs_mean])[None, :]

        if verbose > 1:
            print(f"obs_mean set to: {obs_mean}")
        self.obs_mean = obs_mean

        # scale coordinates and / or observations?
        if obs_scale is None:
            obs_scale = 1
        elif isinstance(obs_scale, list):
            obs_scale = np.array(obs_scale)[None, :]
        elif isinstance(obs_scale, (int, float)):
            obs_scale = np.array([obs_scale])[None, :]

        if verbose > 1:
            print(f"obs_scale set to: {obs_scale}")
        self.obs_scale = obs_scale

        if coords_scale is None:
            coords_scale = 1
        elif isinstance(coords_scale, list):
            coords_scale = np.array(coords_scale)[None, :]
        elif isinstance(coords_scale, (int, float)):
            coords_scale = np.array([coords_scale])[None, :]

        if verbose > 1:
            print(f"coords_scale set to: {coords_scale}")
        self.coords_scale = coords_scale

        # scale coords / obs
        # - will this affect values in place if taken from a data? (dataframe)
        self.coords /= self.coords_scale
        self.obs -= self.obs_mean
        self.obs /= self.obs_scale

        # ---
        # prior mean and kernel functions
        # ---

        # assigning kernel, and prior mean function should be specific to the underlyin engine

        # kernel - either string or function?

        # ---
        # device information
        # ---

        self.gpu_name, self.cpu_name = self._get_device_names()

    def _get_device_names(self):

        gpu_name = None
        cpu_name = self._get_processor_name()

        try:
            dev = device_lib.list_local_devices()
            for d in dev:
                # check if device is GPU
                # - NOTE: will break after first GPU
                if (d.device_type == "GPU") & (gpu_name is None):
                    print("found GPU")
                    try:
                        name_loc = re.search("name:(.*?),", d.physical_device_desc).span(0)
                        gpu_name = d.physical_device_desc[(name_loc[0] + 6):(name_loc[1] - 1)]
                    except Exception as e:
                        print("there was some issue getting GPU name")
                        print(e)
                    break
        except Exception as e:
            print(e)
        return gpu_name, cpu_name

    @staticmethod
    def _get_processor_name():
        # ref: https://stackoverflow.com/questions/4842448/getting-processor-information-in-python
        if platform.system() == "Windows":
            return platform.processor()
        elif platform.system() == "Darwin":
            os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
            command = "sysctl -n machdep.cpu.brand_string"
            return subprocess.check_output(command).strip()
        elif platform.system() == "Linux":
            command = "cat /proc/cpuinfo"
            all_info = subprocess.check_output(command, shell=True).decode().strip()
            for line in all_info.split("\n"):
                if "model name" in line:
                    return re.sub(".*model name.*:", "", line, 1).lstrip()
        return None

    @abstractmethod
    def predict(self, coords):
        """method to generate prediction at given coords"""
        pass

    @abstractmethod
    def optimise_hyperparameters(self):
        pass

    @abstractmethod
    def get_hyperparameters(self):
        pass

    @abstractmethod
    def set_hyperparameters(self):
        pass

    @abstractmethod
    def get_marginal_log_likelihood(self):
        pass


# ------- GPflow models ---------
class GPflowGPRModel(BaseGPRModel):

    @timer
    def __init__(self,
                 data=None,
                 coords_col=None,
                 obs_col=None,
                 coords=None,
                 obs=None,
                 coords_scale=None,
                 obs_scale=None,
                 obs_mean=None,
                 kernel="Matern32",
                 kernel_kwargs=None,
                 mean_function=None,
                 mean_func_kwargs=None,
                 noise_variance=None,
                 likelihood=None,
                 **kwargs):
        # TODO: handle kernel (hyper) parameters

        # --
        # set data
        # --

        super().__init__(data=data,
                         coords_col=coords_col,
                         obs_col=obs_col,
                         coords=coords,
                         obs=obs,
                         coords_scale=coords_scale,
                         obs_scale=obs_scale,
                         obs_mean=obs_mean)

        # --
        # set kernel
        # --

        # TODO: allow for upper and lower bounds to be set of kernel
        #

        assert kernel is not None, "kernel was not provide"

        # if kernel is str: get function
        if isinstance(kernel, str):
            # if additional kernel kwargs not provide use empty dict
            if kernel_kwargs is None:
                kernel_kwargs = {}

            # get the kernel function (still requires
            kernel = getattr(gpflow.kernels, kernel)

            # check signature parameters
            kernel_signature = inspect.signature(kernel).parameters

            # dee if it takes lengthscales
            # - want to initialise with appropriate length (one length scale per coord)
            if ("lengthscales" in kernel_signature) & ("lengthscale" not in kernel_kwargs):
                kernel_kwargs['lengthscales'] = np.ones(self.coords.shape[1])
                print(f"setting lengthscales to: {kernel_kwargs['lengthscales']}")

            # initialise kernel
            kernel = kernel(**kernel_kwargs)

        # TODO: would like to check kernel is correct type / instance

        # --
        # prior mean function
        # --

        if isinstance(mean_function, str):
            if mean_func_kwargs is None:
                mean_func_kwargs = {}
            mean_function = getattr(gpflow.mean_functions, mean_function)(**mean_func_kwargs)

        # ---
        # model
        # ---

        # TODO: allow for model type (e.g. "GPR" to be specified as input?)
        self.model = gpflow.models.GPR(data=(self.coords, self.obs),
                                       kernel=kernel,
                                       mean_function=mean_function,
                                       noise_variance=noise_variance,
                                       likelihood=likelihood)

    def update_obs_data(self,
                        data=None,
                        coords_col=None,
                        obs_col=None,
                        coords=None,
                        obs=None,
                        coords_scale=None,
                        obs_scale=None):

        super().__init__(data=data,
                         coords_col=coords_col,
                         obs_col=obs_col,
                         coords=coords,
                         obs=obs,
                         coords_scale=coords_scale,
                         obs_scale=obs_scale)

        self.model.data = (self.coords, self.obs)


    @timer
    def predict(self, coords, full_cov=False, apply_scale=True):
        """method to generate prediction at given coords"""
        # TODO: allow for only y, or f to be returned
        # convert coords as needed
        if isinstance(coords, pd.Series):
            if self.coords_col is not None:
                coords = coords[self.coords_col].values
            else:
                coords = coords.values
        if isinstance(coords, list):
            coords = np.array(coords)
        # assert isinstance(coords, np.ndarray)
        if len(coords.shape) == 1:
            coords = coords[None, :]

        assert isinstance(coords, np.ndarray), f"coords should be an ndarray (one can be converted from)"
        coords = coords.astype(self.coords.dtype)

        if apply_scale:
            coords = coords / self.coords_scale

        y_pred = self.model.predict_y(Xnew=coords, full_cov=False, full_output_cov=False)
        f_pred = self.model.predict_f(Xnew=coords, full_cov=full_cov)

        # TODO: obs_scale should be applied to predictions
        # z = (x-u)/sig; x = z * sig + u

        if not full_cov:
            out = {
                "f*": f_pred[0].numpy()[:, 0],
                "f*_var": f_pred[1].numpy()[:, 0],
                "y": y_pred[0].numpy()[:, 0],
                "y_var": y_pred[1].numpy()[:, 0],
                "f_bar": self.obs_mean[:, 0]
            }
        else:
            f_cov = f_pred[1].numpy()[0,...]
            f_var = np.diag(f_cov)
            y_var = y_pred[1].numpy()[:, 0]
            # y_cov = K(x,x) + sigma^2 I
            # f_cov = K(x,x), so need to add sigma^2 to diag of f_var
            y_cov = f_cov.copy()
            # get the extra variance needed to diagonal - could use self.model.likelihood.variance.numpy() instead(?)
            diag_var = y_var - f_var
            y_cov[np.arange(len(y_cov)), np.arange(len(y_cov))] += diag_var
            out = {
                "f*": f_pred[0].numpy()[:, 0],
                "f*_var": f_var,
                "y": y_pred[0].numpy()[:, 0],
                "y_var": y_pred[1].numpy()[:, 0],
                "f*_cov": f_cov,
                "y_cov": y_cov,
                "f_bar": self.obs_mean[:, 0]
            }

        return out

    @timer
    def optimise_hyperparameters(self, opt=None, **kwargs):

        # TODO: add option to return opt_logs

        if opt is None:
            opt = gpflow.optimizers.Scipy()

        m = self.model
        opt_logs = opt.minimize(m.training_loss,
                                m.trainable_variables,
                                options=dict(maxiter=10000),
                                **kwargs)
        if not opt_logs['success']:
            print("*" * 10)
            print("optimization failed!")
            # TODO: determine if should return None for failed optimisation
            # return None

        # get the hyper parameters, sca
        hyp_params = self.get_hyperparameters()
        # marginal log likelihood
        mll = self.get_marginal_log_likelihood()
        out = {
            "optimise_success": opt_logs['success'],
            "marginal_loglikelihood": mll,
            **hyp_params
        }

        return out

    def get_marginal_log_likelihood(self):
        """get the marginal log likelihood"""

        return self.model.log_marginal_likelihood().numpy()

    @timer
    def get_hyperparameters(self):

        # length scales
        # TODO: determine here if want to change the length scale names
        #  to correspond with dimension names
        # lscale = {f"ls_{self.coords_col[i]}": _
        #           for i, _ in enumerate(self.model.kernel.lengthscales.numpy())}
        lscale = self.model.kernel.lengthscales.numpy()

        # variances
        kvar = float(self.model.kernel.variance.numpy())
        lvar = float(self.model.likelihood.variance.numpy())

        # check for mean_function parameters
        # if self.model.mean_function.name != "zero":
        #
        #     if self.model.mean_function.name == "constant":
        #         mean_func_params["mean_func"] = self.model.mean_function.name
        #         mean_func_params["mean_func_c"] = float(self.model.mean_function.c.numpy())
        #     else:
        #         warnings.warn(f"mean_function.name: {self.model.mean_function.name} not understood")

        out = {
            # **lscale,
            "lengthscales": lscale,
            "kernel_variance": kvar,
            "likelihood_variance": lvar,
            # **mean_func_params
        }

        return out

    def set_hyperparameters(self, param_dict=None, lengthscales=None, kernel_variance=None, likelihood_variance=None):

        if param_dict is not None:
            assert isinstance(param_dict, dict), "param_dict provide but is type: {type(param_dict)}"
            lengthscales = param_dict.get("lengthscales", None)
            kernel_variance = param_dict.get("kernel_variance", None)
            likelihood_variance = param_dict.get("likelihood_variance", None)

        if lengthscales is not None:
            self.model.kernel.lengthscales.assign(lengthscales)

        if kernel_variance is not None:
            self.model.kernel.variance.assign(kernel_variance)

        if likelihood_variance is not None:
            self.model.likelihood.variance.assign(likelihood_variance)

        return self.get_hyperparameters()

    def apply_param_transform(self, obj, bijector, param_name, **bijector_kwargs):

        # check obj is correct

        # check parameter name is in obj
        assert hasattr(obj, param_name), \
            f"obj of type: {type(obj)}\ndoes not have param_name: {param_name} as attribute"

        # get the parameter
        p = getattr(obj, param_name)

        # check bijector
        if isinstance(bijector, str):
            bijector = getattr(tfp.bijectors, bijector)

        # TODO: check bijector is the correct type
        # TODO: print bijector ?

        # initialise bijector, given the specific
        bij = bijector(**bijector_kwargs)

        # create a new parameter with different transform
        new_p = gpflow.Parameter(p,
                                 trainable=p.trainable,
                                 prior=p.prior,
                                 name=p.name.split(":")[0],
                                 transform=bij)
        # set parameter
        setattr(obj, param_name, new_p)

    @timer
    def set_lengthscale_constraints(self, low, high, obj=None, move_within_tol=True, tol=1e-8, scale=False):

        if obj is None:
            obj = self.model.kernel

        # check inputs
        # - get original length scales
        org_ls = obj.lengthscales

        if isinstance(low, (list, tuple)):
            low = np.array(low)
        elif isinstance(low, (int, float)):
            low = np.array([low])

        if isinstance(high, (list, tuple)):
            high = np.array(high)
        elif isinstance(high, (int, float)):
            high = np.array([high])


        assert len(low.shape) == 1
        assert len(high.shape) == 1

        # - input lengths
        assert len(org_ls.numpy()) == len(low), "len of low constraint does not match lengthscale length"
        assert len(org_ls.numpy()) == len(high), "len of high constraint does not match lengthscale length"

        assert np.all(low <= high), "all values in high constraint must be greater than low"

        # scale the bound by the coordinate scale value
        if scale:
            # self.coords_scale expected to be 2-d
            low = low / self.coords_scale[0, :]
            high = high / self.coords_scale[0, :]

        # extract the current length scale values
        # - does numpy() make a copy of values?
        ls_vals = org_ls.numpy()

        # if the current values are outside of tolerances then move them in
        if move_within_tol:
            # require current length scales are more than tol for upper bound
            ls_vals[ls_vals > (high - tol)] = high[ls_vals > (high - tol)] - tol
            # similarly for the lower bound
            ls_vals[ls_vals < (low + tol)] = low[ls_vals < (low + tol)] + tol

        # if the length scale values have changed then assign the new values
        if (obj.lengthscales.numpy() != ls_vals).any():
            obj.lengthscales.assign(ls_vals)

        # apply constrains
        # - is it required to provide low/high as tf.constant
        self.apply_param_transform(obj=obj,
                                   bijector="Sigmoid",
                                   param_name="lengthscales",
                                   low=tf.constant(low),
                                   high=tf.constant(high))

    def _apply_sigmoid_constraints(self, lb=None, ub=None, eps=1e-8):


        # apply constraints, if both supplied
        # TODO: error or warn if both upper and lower not provided
        if (lb is not None) & (ub is not None):

            # length scale upper bound
            ls_lb = lb * self.scale_inputs
            ls_ub = ub * self.scale_inputs

            # sigmoid function: to be used for length scales
            sig = tfp.bijectors.Sigmoid(low=tf.constant(ls_lb),
                                        high=tf.constant(ls_ub))
            # TODO: determine if the creation / redefining of the Parameter below requires
            #  - as many parameters as given

            # check if length scales are at bounds - move them off if they are
            # ls_scales = k.lengthscales.numpy()
            # if (ls_scales == ls_lb).any():
            #     ls_scales[ls_scales == ls_lb] = ls_ub[ls_scales == ls_lb] + 1e-6
            # if (ls_scales == ls_ub).any():
            #     ls_scales[ls_scales == ls_ub] = ls_ub[ls_scales == ls_ub] - 1e-6
            #
            # # if the length scale values have changed then assign the new values
            # if (k.lengthscales.numpy() != ls_scales).any():
            #     k.lengthscales.assign(ls_scales)
            # p = k.lengthscales
            #
            # k.lengthscales = gpflow.Parameter(p,
            #                                   trainable=p.trainable,
            #                                   prior=p.prior,
            #                                   name=p.name.split(":")[0],
            #                                   transform=sig)


# ------- scikit-learn model ---------

import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel

class sklearnGPRModel(BaseGPRModel):
    @timer
    def __init__(self,
                 data=None,
                 coords_col=None,
                 obs_col=None,
                 coords=None,
                 obs=None,
                 coords_scale=None,
                 obs_scale=None,
                 obs_mean=None,
                 kernel="Matern",
                 kernel_kwargs=None,
                 mean_value=None,
                 kernel_variance=None,
                 likelihood_variance=None,
                 param_bounds=None,
                 **kwargs):
        # TODO: handle kernel (hyper) parameters
        # NOTE: sklearn only handles constant mean
        # NOTE: sklearn only deals with Gaussian likelihood. Likelihood variance is not trainable.

        # --
        # set data
        # --
        super().__init__(data=data,
                         coords_col=coords_col,
                         obs_col=obs_col,
                         coords=coords,
                         obs=obs,
                         coords_scale=coords_scale,
                         obs_scale=obs_scale,
                         obs_mean=obs_mean)

        # --
        # set kernel
        # --

        # TODO: allow for upper and lower bounds to be set of kernel
        #

        assert kernel is not None, "kernel was not provided"

        # if kernel is str: get function
        if isinstance(kernel, str):
            # if additional kernel kwargs not provide use empty dict
            if kernel_kwargs is None:
                kernel_kwargs = {}
            
            # get the kernel function (still requires
            kernel = getattr(sklearn.gaussian_process.kernels, kernel)

            # check signature parameters
            kernel_signature = inspect.signature(kernel).parameters

            # dee if it takes lengthscales
            # - want to initialise with appropriate length (one length scale per coord)
            # TODO: adapt for scikit
            if ("length_scale" in kernel_signature) & ("length_scale" not in kernel_kwargs):
                kernel_kwargs['length_scale'] = np.ones(self.coords.shape[1])
                print(f"setting lengthscales to: {kernel_kwargs['length_scale']}")

            # initialise kernel
            kernel = kernel(**kernel_kwargs)

        # --
        # add constant mean
        # --
        if mean_value is not None:
            kernel += ConstantKernel(mean_value)

        # --
        # include variances
        # --
        if kernel_variance is not None:
            kernel *= ConstantKernel(np.sqrt(kernel_variance))

        # --
        # set hyperparameter bounds
        # --
        if param_bounds is not None:
            for hyperparameter in kernel.hyperparameters:
                hyperparameter.bounds = param_bounds[hyperparameter.name]

        # ---
        # model
        # ---
        self.model = GaussianProcessRegressor(kernel=kernel,
                                              alpha=likelihood_variance,
                                              n_restarts_optimizer=2)

    @timer
    def predict(self, coords, full_cov=False, apply_scale=True):
        """method to generate prediction at given coords"""
        # TODO: allow for only f to be returned
        # convert coords as needed
        if isinstance(coords, pd.Series):
            if self.coords_col is not None:
                coords = coords[self.coords_col].values
            else:
                coords = coords.values
        if isinstance(coords, list):
            coords = np.array(coords)
        # assert isinstance(coords, np.ndarray)
        if len(coords.shape) == 1:
            coords = coords[None, :]

        assert isinstance(coords, np.ndarray), f"coords should be an ndarray (one can be converted from)"
        coords = coords.astype(self.coords.dtype)

        if apply_scale:
            coords = coords / self.coords_scale
        
        if full_cov:
            return_std = False
            return_cov = True
        else:
            return_std = True
            return_cov = False

        f_pred = self.model.predict(X=coords,
                                    return_std=return_std,
                                    return_cov=return_cov)

        # TODO: obs_scale should be applied to predictions
        # z = (x-u)/sig; x = z * sig + u

        if not full_cov:
            out = {
                "f*": f_pred[0][0],
                "f*_var": f_pred[1][0],
                "f_bar": self.obs_mean[0,0]
            }
        else:
            f_cov = f_pred[1]
            f_var = np.diag(f_cov)
            out = {
                "f*": f_pred[0][0],
                "f*_var": f_var,
                "f*_cov": f_cov,
                "f_bar": self.obs_mean[0,0]
            }

        return out

    @timer
    def optimise_hyperparameters(self, opt=None, **kwargs):

        # TODO: add option to return opt_logs

        if opt is None:
            opt = 'fmin_l_bfgs_b'

        X = self.coords
        y = self.obs

        try:
            self.model = self.model.fit(X, y)
            success = True
            mll = self.get_marginal_log_likelihood()
        except:
            print("*" * 10)
            print("optimization failed!")
            success = False
            mll = np.nan

        # get the hyper parameters, sca
        hyp_params = self.get_hyperparameters()

        out = {
            "optimise_success": success,
            "marginal_loglikelihood": mll,
            **hyp_params
        }

        return out

    def get_marginal_log_likelihood(self):
        """get the marginal log likelihood"""
        return self.model.log_marginal_likelihood()

    @timer
    def get_hyperparameters(self):
        # Below works for Matern. Not checked with other kernels.
        try:
            kernel = self.model.kernel_ # Only available after training
        except:
            kernel = self.model.kernel

        if self.model.kernel.__class__ == "sklearn.gaussian_process.kernels.Sum":
            # Deal with mean
            k = kernel.k1
            k1 = k.k1
            k2 = k.k2
        else:
            k1 = kernel.k1
            k2 = kernel.k2
        param_dict = {}
        param_dict['lengthscales'] = k1.length_scale
        param_dict['kernel_variance'] = k2.constant_value
        return param_dict

    def set_hyperparameters(self, param_dict):
        # Below works for Matern. Not checked with other kernels.
        try:
            kernel = self.model.kernel_
        except:
            kernel = self.model.kernel

        if kernel.__class__ == "sklearn.gaussian_process.kernels.Sum":
            # Deal with mean
            k = kernel.k1
            k1 = k.k1
            k2 = k.k2
        else:
            k1 = kernel.k1
            k2 = kernel.k2
        k1.length_scale = param_dict['lengthscales']
        k2.constant_value = param_dict['variance']

    @timer
    def set_lengthscale_constraints(self, low, high, move_within_tol=True, tol=1e-8, scale=False):
        ls = self.get_hyperparameters()['lengthscales']

        if isinstance(low, (list, tuple)):
            low = np.array(low)
        elif isinstance(low, (int, float)):
            low = np.array([low])

        if isinstance(high, (list, tuple)):
            high = np.array(high)
        elif isinstance(high, (int, float)):
            high = np.array([high])

        assert len(ls) == len(low), "len of low constraint does not match lengthscale length"
        assert len(ls) == len(high), "len of high constraint does not match lengthscale length"
        assert np.all(low <= high), "all values in high constraint must be greater than low"

        # scale the bound by the coordinate scale value
        if scale:
            # self.coords_scale expected to be 2-d
            low = low / self.coords_scale[0, :]
            high = high / self.coords_scale[0, :]

        # if the current values are outside of tolerances then move them in
        if move_within_tol:
            # require current length scales are more than tol for upper bound
            ls[ls > (high - tol)] = high[ls > (high - tol)] - tol
            # similarly for the lower bound
            ls[ls < (low + tol)] = low[ls < (low + tol)] + tol

        length_scale_bounds = []
        for l, h in zip(low, high):
            length_scale_bounds.append((l,h))

        # Below works for Matern. Not checked with other kernels.
        try:
            kernel = self.model.kernel_ # Only available after training
        except:
            kernel = self.model.kernel

        if kernel.__class__ == "sklearn.gaussian_process.kernels.Sum":
            # Deal with mean
            k = kernel.k1.k1
        else:
            k = kernel.k1
        
        k.length_scale_bounds = length_scale_bounds


# ------- ... ---------

class SpatiotemporalOptimalInterpolation(ABC):
    """
    Base class containing the bare bone functionality for optimal interpolation using local GPs
    """
    def __init__(self, training_data, gp_model, kernel, init_hyperparameters, mean_function=None, grid_res=50, training_radius=300*1000, time_window=9):
        X, z = training_data
        x, y, t = X.T
        self.x_train = x
        self.y_train = y
        self.t_train = t
        self.z_train = z # Must be of shape (N, out_dim)
        self.kdt_tree = scipy.spatial.cKDTree(np.array([x, y]).T)
        self.gp_model = gp_model
        self.kernel = kernel
        self.mean_function = mean_function
        self.resolution = grid_res
        self.radius = training_radius
        self.T = time_window
        self._hyperparam_keys = None
        self._init_hyperparameters = init_hyperparameters

    def _clip_and_smooth(self, param, vmin=0, vmax=1e10, std=1):
        """
        Apply clipping and Gaussian smoothing to a given hyperparameter field
        """
        param = param.values # convert DataArray to numpy array
        # Clipping
        param[param >= vmax] = vmax
        param[param <= vmin] = vmin
        # Smoothing
        param = convolve(param, Gaussian2DKernel(x_stddev=std, y_stddev=std))
        return param

    def _post_process(self, param_fields: xr.Dataset, postprocess_kwargs):
        """
        Post process all trainable hyperparameters
        """
        assert set(postprocess_kwargs.keys()) == set(self._hyperparam_keys)
        for key in self._hyperparam_keys:
            param = self._clip_and_smooth(param_fields.data_vars[key], **postprocess_kwargs[key])
            param_fields[key] = (('y', 'x'), param)

    @abstractmethod
    def _get_hyperparameters(self, model):
        """
        Method to get the hyperparameters of the model

        Output:
        Hyperparameters of the model given by a dictionary whose keys are the hyperparameter labels
        and the values are the corresponding values
        """

    @abstractmethod
    def _get_kernel(self):
        """
        Method to instantiate kernel
        """

    @abstractmethod
    def _get_mean(self):
        """
        Method to get the mean function
        """

    @abstractmethod
    def _build_model(self, X, y, kernel, mean, hyperparameters):
        """
        Method to compile GP model
        """

    @abstractmethod
    def _optimise(self, model):
        """
        Method to optimise hyperparameters of the model
        """

    @abstractmethod
    def _get_prediction(self, model, gridpoint) -> List:
        """
        Method to get the predicted mean and variance of the model

        Output: List of form (mean, variance)
        """

    def train(self, dates, region: xr.Dataset, trainable_hyperparameters: List, postprocess_kwargs=None):
        """
        Main training loop.
        ------
        Args
        ------
        :dates: A list containing the dates of consideration
        :region: An xarray Dataset object containing coordinates x and y for the region of consideration
        :trainable_hyperparameters: A list containing the names of the trainable hyperparameters.
                                    The names must be compatible with the keys in self._init_hyperparameters
        :postprocess_kwargs: A nested dictionary of form
                            {hyperparameter name : dictionary of key word arguments to feed into self._clip_and_smooth}
        """
        self._hyperparam_keys = trainable_hyperparameters
        for date in dates:
            print(f"Training on date: {date}")
            xdim = len(region['x'])
            ydim = len(region['y'])
            x_coords = region['x'].values
            y_coords = region['y'].values
            data_vars = {key: (('y', 'x'), np.ones((ydim, xdim))) for key in trainable_hyperparameters}
            hyperparam_fields = xr.Dataset(data_vars=data_vars, coords={'x': x_coords, 'y': y_coords})
            count = 0
            cumulative_time = 0
            for i, x in enumerate(x_coords):
                for j, y in enumerate(y_coords):
                    time0 = time.time()

                    # 1. Select training data within a ball of radius r of the current grid point
                    ID = self.kdt_tree.query_ball_point(x=np.array([x,y]).T, r=self.radius)
                    inputs = np.array([self.x_train[ID], self.y_train[ID], self.t_train[ID]]).T
                    outputs = self.z_train[ID]
                    
                    # 2. Setup GP model
                    kernel = self._get_kernel()
                    mean = self._get_mean()
                    model = self._build_model(inputs, outputs, kernel, mean, self._init_hyperparameters)

                    # 3. Set trainable hyperparameters (TODO: Fix. Not general)
                    hyperparams = self._get_hyperparameters(model)
                    for key, val in hyperparams.items():
                        if key not in trainable_hyperparameters:
                            gpflow.set_trainable(val, False)

                    # 4. Optimise hyperparameters
                    self._optimise(model)
                    
                    # 5. Cache optimised hyperparameters
                    hyperparams_new = self._get_hyperparameters(model)
                    for key in trainable_hyperparameters:
                        hyperparam_fields.data_vars[key][j,i] = hyperparams_new[key]

                    time1 = time.time()
                    cumulative_time += time1-time0
                    count += 1
                    print(f"Time elapsed for gridpoint {count}/{xdim*ydim}: {time1-time0:.3f}s")

            # Post-process hyperparameters
            print(f"Post-processing hyperparameters...")
            if postprocess_kwargs is not None:
                self._post_process(hyperparam_fields, postprocess_kwargs)

            # Log results
            print(f"Saving results...")
            path = f"log/hyperparameters_{date}_{self.resolution}km.nc" # TODO: add logdir to class attribute
            hyperparam_fields.to_netcdf(path)

            print(f"Complete.")
            print(f"Total time for training: {cumulative_time:.3f}s")\

    def predict(self, dates, region: xr.Dataset):
        """
        Make predictions
        """
        T_mid = self.T // 2 # choose central day in the time window to make predictions
        x_coords = region['x'].values
        y_coords = region['y'].values
        gridded_mean = np.zeros((len(dates), len(y_coords), len(x_coords)))
        gridded_var = np.zeros((len(dates), len(y_coords), len(x_coords)))
        for t, date in enumerate(dates):
            path = f"log/hyperparameters_{date}_{self.resolution}km.nc"
            hyperparam_dataset = xr.load_dataset(path)
            for i, x in enumerate(x_coords):
                for j, y in enumerate(y_coords):
                    # TODO: Steps 1 and 2 below are almost identical to that in training step. Add as separate method.
                    # 1. Select training data within a ball of radius r of the current grid point
                    ID = self.kdt_tree.query_ball_point(x=np.array([x,y]).T, r=self.radius)
                    inputs = np.array([self.x_train[ID], self.y_train[ID], self.t_train[ID]]).T
                    outputs = self.z_train[ID]

                    # 2. Setup GP model
                    kernel = self._get_kernel()
                    mean = self._get_mean()
                    keys = list(hyperparam_dataset.keys())
                    vars = [hyperparam_dataset[key].values[j,i] for key in keys]
                    hyperparams = dict(zip(keys, vars))
                    model = self._build_model(inputs, outputs, kernel, mean, hyperparams)
                    
                    # 3. Predict at gridpoint
                    gridpoint = np.atleast_2d(np.array([x,y,T_mid])) #test input
                    mean, var = self._get_prediction(model, gridpoint)
                    gridded_mean[t,j,i] = mean
                    gridded_var[t,j,i] = var

        return gridded_mean, gridded_var

# ------- Pure python model ---------

...

# ------- GPflow models ---------

class GPR_OptimalInterpolation(SpatiotemporalOptimalInterpolation):
    def __init__(self, training_data, kernel, init_hyperparameters, max_iter=10_000, grid_res=50, training_radius=300*1000, time_window=9):
        self.opt = gpflow.optimizers.Scipy()
        self.max_iter = max_iter
        super().__init__(training_data,
                         gp_model=gpflow.models.GPR,
                         kernel=kernel,
                         init_hyperparameters=init_hyperparameters,
                         mean_function=gpflow.mean_functions.Constant,
                         grid_res=grid_res,
                         training_radius=training_radius,
                         time_window=time_window)

    def _get_mean(self):
        """ Get constant mean
        TODO: extend to more general means
        """
        mean = self.mean_function(c=np.array([np.mean(self.z_train)]))
        gpflow.set_trainable(mean.c, False)
        return mean

    def _get_kernel(self):
        hyperparameters = self._init_hyperparameters
        ls = (hyperparameters['lengthscale_x'], hyperparameters['lengthscale_y'], hyperparameters['lengthscale_t'])
        kernel = self.kernel(lengthscales=ls)
        return kernel

    def _get_hyperparameters(self, model: gpflow.models.GPR) -> Dict:
        hyperparameters = {}
        hyperparam_keys = ['lengthscale_x', 'lengthscale_y', 'lengthscale_t', 'kernel_variance', 'observation_variance']
        hyperparam_vals = [*model.kernel.lengthscales, model.kernel.variance, model.likelihood.variance]
        for key, val in zip(hyperparam_keys, hyperparam_vals):
            hyperparameters[key] = val
        return hyperparameters

    def _build_model(self, X, y, kernel, mean, hyperparameters):
        model = self.gp_model(data=(X, y), kernel=kernel, mean_function=mean)
        ls = (hyperparameters['lengthscale_x'], hyperparameters['lengthscale_y'], hyperparameters['lengthscale_t'])
        model.kernel.lengthscales.assign(ls)
        model.kernel.variance.assign(hyperparameters['kernel_variance'])
        model.likelihood.variance.assign(hyperparameters['observation_variance'])
        return model

    def _optimise(self, model: gpflow.models.GPR):
        self.opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=self.max_iter))

    def _get_prediction(self, model: gpflow.models.GPR, gridpoint: np.ndarray) -> List:
        mean, var = model.predict_f(gridpoint)
        return mean.numpy()[0,0], var.numpy()[0,0]


class SGPR_OptimalInterpolation(SpatiotemporalOptimalInterpolation):
    ...


class SVGP_OptimalInterpolation(SpatiotemporalOptimalInterpolation):
    ...

    