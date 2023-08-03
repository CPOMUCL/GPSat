import inspect
import pandas as pd
import numpy as np
from typing import List, Dict
from GPSat.decorators import timer
from GPSat.models import BaseGPRModel

import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.base import clone


# ------- scikit-learn model ---------

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
                 verbose=True,
                 *,
                 kernel="Matern",
                 kernel_kwargs=None,
                 mean_value=None,
                 kernel_variance=1.,
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
                         obs_mean=obs_mean,
                         verbose=verbose)

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

            # see if it takes lengthscales
            # - want to initialise with appropriate length (one length scale per coord)
            # TODO: adapt for scikit
            if ("length_scale" in kernel_signature) & ("length_scale" not in kernel_kwargs):
                kernel_kwargs['length_scale'] = np.ones(self.coords.shape[1])
                if verbose:
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
        if likelihood_variance is None:
            likelihood_variance = 1.

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

        try:
            _ = self.model.kernel_ # This is to check if model has been trained. NOTE: `kernel_` only exists after model has been trained.
                                   # If not trained, gpr effectively has not been 'fitted' to the data, in which case we need to 'fake fit'.

            f_pred = self.model.predict(X=coords,
                                        return_std=return_std,
                                        return_cov=return_cov)
        except AttributeError as e:
            # if fit has not been called certain attributes will be missing
            # by calling with optimizer=None no optimization will be done
            # TODO: this should be reviewed and validated
            self._fake_fit()
            f_pred = self.model.predict(X=coords,
                                        return_std=return_std,
                                        return_cov=return_cov)

        # TODO: obs_scale should be applied to predictions
        # z = (x-u)/sig; x = z * sig + u

        if not full_cov:
            out = {
                "f*": np.atleast_1d(f_pred[0]),
                "f*_var": np.atleast_1d(f_pred[1]**2)
            }
        else:
            f_cov = f_pred[1]
            f_var = np.diag(f_cov)
            out = {
                "f*": np.atleast_1d(f_pred[0]),
                "f*_var": f_var,
                "f*_cov": f_cov
            }

        return out

    @property
    def param_names(self) -> list:
        return ["lengthscales", "kernel_variance", "likelihood_variance"]

    def _extract_k1k2(self):
        """
        Extracts k1: Matern kernel
                 k2: Constant kernel, which models the amplitude
        """
        # Below works for Matern. Not checked with other kernels.
        try:
            kernel = self.model.kernel_ # Only available after training
        except:
            kernel = self.model.kernel

        if self.model.kernel.__class__ == sklearn.gaussian_process.kernels.Sum:
            # Deal with mean
            k = kernel.k1
            k1 = k.k1
            k2 = k.k2
        elif self.model.kernel.__class__ == sklearn.gaussian_process.kernels.Product:
            k1 = kernel.k1
            k2 = kernel.k2
        else:
            k1 = kernel
            k2 = None

        return (k1, k2)

    def get_lengthscales(self):
        k1, k2 = self._extract_k1k2()
        # TODO: output should be numpy array always
        return k1.length_scale

    def get_kernel_variance(self):
        k1, k2 = self._extract_k1k2()
        if k2 is None:
            return 1.
        else:
            return k2.constant_value**2

    def get_likelihood_variance(self):
        return self.model.alpha

    def set_lengthscales(self, lengthscales):
        k1, k2 = self._extract_k1k2()
        k1.length_scale = lengthscales

    def set_kernel_variance(self, kernel_variance):
        k1, k2 = self._extract_k1k2()
        if k2 is None:
            pass
        else:
            k2.constant_value = np.sqrt(kernel_variance)

    def set_likelihood_variance(self, likelihood_variance):
        self.model.alpha = likelihood_variance

    @timer
    def optimise_parameters(self, opt=None, **kwargs):

        # TODO: add option to return opt_logs

        if opt is None:
            self.model.optimizer = 'fmin_l_bfgs_b'
        else:
            self.model.optimizer = opt

        X = self.coords
        y = self.obs

        try:
            self.model = self.model.fit(X, y)
            success = True
        except:
            print("*" * 10)
            print("optimization failed!")
            success = False

        return success

    @timer
    def get_objective_function_value(self):
        """get the marginal log likelihood"""
        try:
            # this only works if self.model.fit as been already called
            return self.model.log_marginal_likelihood()
        # if model has not been trained then self.model.log_marginal_likelihood_value_ will not exist
        # and thus cause an attribution error
        except AttributeError as e:
            # kernel_ (and other attributes) only gets assigned when self.model.fit(...) is called
            # call fit with optimizer=None via:
            self._fake_fit()
            return -self.model.log_marginal_likelihood()

    def _fake_fit(self):
        """call model.fit with optimizer None"""
        optimizer = self.model.optimizer
        self.model.optimizer = None
        self.model.fit(X=self.coords, y=self.obs)
        self.model.optimizer = optimizer

    def _preprocess_constraint(self, param_name, low, high, move_within_tol=True, tol=1e-8, scale=False):
        assert param_name in self.param_names, f"param_name must be one of {self.param_names}"

        param = self.get_parameters()[param_name]
        param = np.atleast_1d(param)

        if isinstance(low, (list, tuple)):
            low = np.array(low)
        elif isinstance(low, (int, float)):
            low = np.array([low])

        if isinstance(high, (list, tuple)):
            high = np.array(high)
        elif isinstance(high, (int, float)):
            high = np.array([high])

        assert len(param) == len(low), f"len of low constraint does not match size of parameter {param_name}"
        assert len(param) == len(high), f"len of high constraint does not match size of parameter {param_name}"
        assert np.all(low <= high), "all values in high constraint must be greater than low"

        # scale the bound by the coordinate scale value
        if scale:
            # self.coords_scale expected to be 2-d
            low = low / self.coords_scale[0, :]
            high = high / self.coords_scale[0, :]

        # if the current values are outside of tolerances then move them in
        if move_within_tol:
            # require current length scales are more than tol for upper bound
            param[param > (high - tol)] = high[param > (high - tol)] - tol
            # similarly for the lower bound
            param[param < (low + tol)] = low[param < (low + tol)] + tol

        return low, high

    @timer
    def set_lengthscales_constraints(self, low, high, move_within_tol=True, tol=1e-8, scale=False):
        low, high = self._preprocess_constraint("lengthscales", low, high, move_within_tol, tol, scale)

        length_scale_bounds = []
        for l, h in zip(low, high):
            length_scale_bounds.append((l,h))

        # Below works for Matern. Not checked with other kernels.
        try:
            kernel = self.model.kernel_ # Only available after training
        except:
            kernel = self.model.kernel

        if kernel.__class__ == sklearn.gaussian_process.kernels.Sum:
            # Deal with mean
            k = kernel.k1.k1
        elif kernel.__class__ == sklearn.gaussian_process.kernels.Product:
            k = kernel.k1
        else:
            k = kernel
        
        k.length_scale_bounds = length_scale_bounds

    @timer
    def set_kernel_variance_constraints(self, low, high, move_within_tol=True, tol=1e-8, scale=False):
        low, high = self._preprocess_constraint("kernel_variance", low, high, move_within_tol, tol, scale)

        # Below works for Matern. Not checked with other kernels.
        try:
            kernel = self.model.kernel_ # Only available after training
        except:
            kernel = self.model.kernel

        if kernel.__class__ == sklearn.gaussian_process.kernels.Sum:
            # Deal with mean
            k = kernel.k1
        else:
            k = kernel
        
        k.constant_value_bounds = (low[0], high[0])

        