import inspect
import pandas as pd
import gpflow
import numpy as np
from copy import copy

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.client import device_lib

from gpflow.utilities import set_trainable
from typing import List, Dict

from PyOptimalInterpolation.decorators import timer
from PyOptimalInterpolation.models import BaseGPRModel


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


    @property
    def param_names(self) -> list:
        return ["lengthscales", "kernel_variance", "likelihood_variance"]

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
            coords = coords[None, :] # Is this correct?

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
                # "y": y_pred[0].numpy()[:, 0],
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
                # "y": y_pred[0].numpy()[:, 0],
                "y_var": y_pred[1].numpy()[:, 0],
                "f*_cov": f_cov,
                "y_cov": y_cov,
                "f_bar": self.obs_mean[:, 0]
            }

        return out

    @timer
    def optimise_parameters(self, opt=None, **kwargs):

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
        hyp_params = self.get_parameters()
        # marginal log likelihood
        mll = self.get_objective_function_value()
        out = {
            "optimise_success": opt_logs['success'],
            "marginal_loglikelihood": mll,
            **hyp_params
        }

        return out

    def get_objective_function_value(self):
        """get the marginal log likelihood"""

        return self.model.log_marginal_likelihood().numpy()

    def get_lengthscales(self):
        return self.model.kernel.lengthscales.numpy()

    def get_kernel_variance(self):
        return float(self.model.kernel.variance.numpy())

    def get_likelihood_variance(self):
        return float(self.model.likelihood.variance.numpy())

    def set_lengthscales(self, lengthscales):
        self.model.kernel.lengthscales.assign(lengthscales)

    def set_kernel_variance(self, kernel_variance):
        self.model.kernel.variance.assign(kernel_variance)

    def set_likelihood_variance(self, likelihood_variance):
        self.model.likelihood.variance.assign(likelihood_variance)

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

        # Reshape p if necessary
        if len(p.shape) == 0:
            p = gpflow.Parameter(np.atleast_1d(p.numpy()),
                                 trainable=p.trainable,
                                 prior=p.prior,
                                 name=p.name.split(":")[0],
                                 transform=bij)

        # create a new parameter with different transform
        new_p = gpflow.Parameter(p,
                                 trainable=p.trainable,
                                 prior=p.prior,
                                 name=p.name.split(":")[0],
                                 transform=bij)
        # set parameter
        setattr(obj, param_name, new_p)

    @timer
    def set_lengthscale_constraints(self, low, high, obj=None, move_within_tol=True, tol=1e-8, scale=False, scale_magnitude=None):

        if obj is None:
            obj = self.model.kernel

        # check inputs
        # - get original length scales
        org_ls = obj.lengthscales

        if isinstance(low, (list, tuple)):
            low = np.array(low)
        elif isinstance(low, (int, np.int64, float)):
            low = np.array([low])

        if isinstance(high, (list, tuple)):
            high = np.array(high)
        elif isinstance(high, (int, np.int64, float)):
            high = np.array([high])

        assert len(low.shape) == 1
        assert len(high.shape) == 1

        # extract the current length scale values
        # - does numpy() make a copy of values?
        ls_vals = np.atleast_1d(org_ls.numpy())

        # - input lengths
        assert len(ls_vals) == len(low), "len of low constraint does not match lengthscale length"
        assert len(ls_vals) == len(high), "len of high constraint does not match lengthscale length"

        assert np.all(low <= high), "all values in high constraint must be greater than low"

        # scale the bound by the coordinate scale value
        if scale:
            if scale_magnitude is None:
                # self.coords_scale expected to be 2-d
                low = low / self.coords_scale[0, :]
                high = high / self.coords_scale[0, :]
            else:
                low = low / scale_magnitude
                high = high / scale_magnitude

        # if the current values are outside of tolerances then move them in
        if move_within_tol:
            # require current length scales are more than tol for upper bound
            ls_vals[ls_vals > (high - tol)] = high[ls_vals > (high - tol)] - tol
            # similarly for the lower bound
            ls_vals[ls_vals < (low + tol)] = low[ls_vals < (low + tol)] + tol

        # if the length scale values have changed then assign the new values
        if (np.atleast_1d(obj.lengthscales.numpy()) != ls_vals).any():
            obj.lengthscales.assign(ls_vals)

        # apply constrains
        # - is it required to provide low/high as tf.constant
        self.apply_param_transform(obj=obj,
                                   bijector="Sigmoid",
                                   param_name="lengthscales",
                                   low=tf.constant(low),
                                   high=tf.constant(high))

    def _apply_sigmoid_constraints(self, lb=None, ub=None, eps=1e-8):
        # TODO: _apply_sigmoid_constraints needs work...

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



class GPflowSGPRModel(GPflowGPRModel):
    """
    Model using SGPR (Sparse GPR)
    """
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
                 num_inducing_points=None,
                 train_inducing_points=False,
                 kernel_kwargs=None,
                 mean_function=None,
                 mean_func_kwargs=None,
                 noise_variance=None,
                 likelihood=None):
        # TODO: handle kernel (hyper) parameters
        # TODO: include options for inducing points (random or grid)

        # --
        # set data
        # --

        BaseGPRModel.__init__(self,
                              data=data,
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

        # --
        # Set inducing points
        # --
        if (num_inducing_points is None) or (len(self.coords) < num_inducing_points):
            # If number of inducing points is not specified or if it is greater than the number of data points,
            # we set it to coincide with the data points
            print("setting inducing points to data points...")
            self.inducing_points = self.coords
        else:
            X = copy(self.coords)
            np.random.shuffle(X)
            self.inducing_points = X[:num_inducing_points]

        # ---
        # model
        # ---

        # TODO: allow for model type (e.g. "GPR" to be specified as input?)
        self.model = gpflow.models.SGPR(data=(self.coords, self.obs),
                                        kernel=kernel,
                                        mean_function=mean_function,
                                        noise_variance=noise_variance,
                                        likelihood=likelihood,
                                        inducing_variable=self.inducing_points)

        if not train_inducing_points:
            set_trainable(self.model.inducing_variable.Z, False)

    def get_objective_function_value(self):
        """get the marginal log likelihood"""

        return self.model.elbo().numpy()

        