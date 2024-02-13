import inspect
import warnings
import re

import pandas as pd
import gpflow
import numpy as np
import itertools
from copy import copy

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.client import device_lib

from gpflow.base import Parameter
from gpflow.config import default_float
from gpflow.utilities import set_trainable, triangular
from gpflow.utilities.traversal import leaf_components
from gpflow.models.util import inducingpoint_wrapper
from gpflow.inducing_variables import InducingVariables

from typing import List, Dict, Union, Optional

from GPSat.decorators import timer
from GPSat.models import BaseGPRModel
from GPSat.utils import cprint

# ------- GPflow models ---------

class GPflowBaseModel(BaseGPRModel):
    """
    Model based on the GPflow implementation of exact Gaussian process regression (GPR).

    See :class:`~GPSat.models.base_model.BaseGPRModel` for a complete list of attributes and methods.
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
                 verbose=True,
                 *,
                 model="GPR",
                 kernel="Matern32",
                 kernel_kwargs=None,
                 mean_function=None,
                 mean_func_kwargs=None,
                 likelihood: gpflow.likelihoods.Gaussian = None,
                 **kwargs):
        """
        Parameters
        ----------
        data
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        coords_col
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        obs_col
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        coords
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        obs
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        coords_scale
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        obs_scale
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        obs_mean
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        verbose
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        kernel: str | gpflow.kernels, default "Matern32"
            The kernel used for GPR. We can use the following `GPflow kernels <https://gpflow.github.io/GPflow/develop/api/gpflow/kernels/index.html>`_,
            which can be passed as a string:
            "Cosine", "Exponential", "Matern12", "Matern32", "Matern52", "RationalQuadratic" or "RBF" (equivalently "SquaredExponential").
        kernel_kwargs: dict, optional
            Keyword arguments to be passed to the GPflow kernel specified in ``kernel``.
        mean_function: str | gpflow.mean_functions, optional
            `GPflow mean function <https://gpflow.github.io/GPflow/develop/notebooks/getting_started/mean_functions.html>`_ to model the prior mean.
        mean_func_kwargs: dict, optional
            Keyword arguments to be passed to the GPflow mean function specified in ``mean_function``.
        noise_variance: float, optional
            Variance of Gaussian likelihood. Unnecessary if ``likelihood`` is specified explicitly.
        likelihood: gpflow.likelihoods.Gaussian, optional
            GPflow model for Gaussian likelihood used to model data uncertainty.
            Can use custom GPflow Gaussian likelihood class here.
            Unnecessary if using a vanilla Gaussian likelihood and ``noise_variance`` is specified.

        """
        # TODO: handle kernel (hyper) parameters
        # TODO: remove duplicate __init_ code -

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

        # TODO: refactor the kernel section
        # TODO: allow for upper and lower bounds to be set of kernel

        assert kernel is not None, "kernel was not provided"

        # if kernel is str: get function
        if isinstance(kernel, str):
            # if additional kernel kwargs not provide use empty dict
            if kernel_kwargs is None:
                kernel_kwargs = {}

            # get the kernel function (still requires
            kernel = getattr(gpflow.kernels, kernel)

            # check signature parameters
            # kernel_signature = inspect.signature(kernel).parameters

            # TODO: review setting of length scales, is it needed - what happens if it's not done?
            # see if it takes lengthscales
            # - want to initialise with appropriate length (one length scale per coord)
            # if ("lengthscales" in kernel_signature) & ("lengthscales" not in kernel_kwargs):
            #     kernel_kwargs['lengthscales'] = np.ones(self.coords.shape[1])
            #     print(f"setting lengthscales to: {kernel_kwargs['lengthscales']}")

            # initialise kernel
            # print(f"kernel_kwargs: {kernel_kwargs}")
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
        # TODO: want to allow any valid model, and for additional parameters to be provided via kwargs

        self.model = getattr(gpflow.models, model)(data=(self.coords, self.obs),
                                                   kernel=kernel,
                                                   likelihood=likelihood,
                                                   mean_function=mean_function,
                                                   **kwargs)


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
    def predict(self):
        pass

    @timer
    def optimise_parameters(self, max_iter=10_000, fixed_params=None, **opt_kwargs):
        """
        Method to optimise the kernel hyperparameters using a scipy optimizer (``method = L-BFGS-B`` by default).

        Parameters
        ----------
        max_iter: int, default 10000
            The maximum number of interations permitted for optimisation.
            The optimiser runs either until convergence or until the number of iterations reach ``max_iter``.
        fixed_params: list of str, default []
            Parameters to fix during optimisation. Should be one of "lengthscales", "kernel_variance" and "likelihood_variance".
        opt_kwargs: dict, optional
            Keyword arguments passed to `gpflow.optimizers.Scipy.minimize() <https://gpflow.github.io/GPflow/develop/_modules/gpflow/optimizers/scipy.html#Scipy.minimize>`_.

        Returns
        -------
        bool
            Indication of whether optimisation was successful or not, i.e. converges within the maximum number of iterations set.

        """
        if fixed_params is None:
            fixed_params = []

        self._fix_hyperparameters(fixed_params)

        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(self.model.training_loss,
                                self.model.trainable_variables,
                                options=dict(maxiter=max_iter),
                                **opt_kwargs)

        if not opt_logs['success']:
            cprint("*" * 10, c="WARNING")
            cprint("optimization failed!", c="WARNING")
            # TODO: determine if should return None for failed optimisation
            # return None

        return opt_logs['success']
    def _fix_hyperparameters(self, params_list, flag=False):
        for param in params_list:
            gpflow.set_trainable(self._params[param], flag)

    # -----
    # Getters/setters for model hyperparameters
    # -----

    def _params(self):
        tmp = leaf_components(self.model)
        # replace . with _ - for legacy hyper-parameter matching
        # and to allow writing to tables in hdf5/sql
        for k in list(tmp.keys()):
            new_k = re.sub("\.", "_", k)
            tmp[new_k] = tmp.pop(k)
        return tmp

    def _match_param_name(self, name, name_list):
        # simple function to allow for partial matching of name to a value in name_list
        # - specifically partial matching allows for name to match end of each element in name_list
        return [_ for _ in name_list if re.search(f"{name}$", _)]

    def _multi_match_param_name(self, name_list, *args):
        # matching input args to values in name_list, allowing for partial matching
        # - will error if more than one or zero matches found
        # - returns a dict of args to parameter name

        # check args are validate param_names
        bad_match, good_match = {}, {}
        for a in args:
            arg_match = self._match_param_name(a, name_list)

            if len(arg_match) == 1:
                good_match[a] = arg_match[0]
            elif len(arg_match) > 1:
                bad_match[a] = arg_match
            elif len(arg_match) == 0:
                bad_match[a] = []

        assert len(
            bad_match) == 0, "the following arguments had incorrect number of parameter name matches\n: {}".format(
            bad_match)
        return good_match

    # TODO: review use of param_names
    @property
    def param_names(self) -> List[str]:
        return list(self._params().keys())

    @timer
    def get_parameters(self, *args, return_dict=True) -> Union[dict, list]:
        # get a numpy representation of the parameters
        # - allow for partial matching of parameter
        # - getting data in numpy array is currently done for legacy reasons

        params = self._params()

        # if not args provided default to get all
        if len(args) == 0:
            args = params.keys()

        good_match = self._multi_match_param_name(list(params.keys()), *args)

        # extract the numpy arrays
        out = {}
        for k, v in good_match.items():
            # here will use/return the full name
            out[v] = params[v].numpy()

        return out if return_dict else [out[k] for k in out.keys()]

    @timer
    def set_parameters(self, **kwargs):

        params = self._params()

        good_match = self._multi_match_param_name(list(params.keys()),
                                                  *list(kwargs.keys()), )

        for k, v in kwargs.items():
            # get the parameter full name
            k_full = good_match[k]

            # assign value
            try:
                # special check for inducing points
                # TODO: determine if this is needed ever
                if isinstance(params[k_full], InducingVariables):
                    params[k_full].assign(inducingpoint_wrapper(v))
                else:
                    params[k_full].assign(v)

            # if there is a shape issue, handle
            # TODO: determine what the exceptions should be caught here
            except Exception as e:
                print(repr(e))
                param_shape = params[k_full].shape
                params[k_full].assign(v.reshape(param_shape))

    # -----
    # Applying constraints on the model hyperparameters
    # -----
    def _set_param_constraints(self,
                               param_name,
                               obj,  # GPflow object. Kernel or likelihood.
                               low,
                               high,
                               move_within_tol=True,
                               tol=1e-8,
                               scale=False,
                               scale_magnitude=None):
        """
        Parameters
        ----------
        """
        # TODO: review to see if can apply constraint just from parameter object


        assert hasattr(obj, param_name), \
            f"obj of type: {type(obj)}\ndoes not have param_name: {param_name} as attribute"
        # - get original parameter
        original_param = getattr(obj, param_name)

        if isinstance(low, (list, tuple)):
            low = np.array(low, dtype=np.float64)
        elif isinstance(low, (int, np.int64, float)):
            low = np.array([low], dtype=np.float64)

        if isinstance(high, (list, tuple)):
            high = np.array(high, dtype=np.float64)
        elif isinstance(high, (int, np.int64, float)):
            high = np.array([high], dtype=np.float64)

        assert len(low.shape) == 1
        assert len(high.shape) == 1

        # extract the current length scale values
        param_vals = np.atleast_1d(original_param.numpy())

        # - input lengths
        assert len(param_vals) == len(low), "len of low constraint does not match param length"
        assert len(param_vals) == len(high), "len of high constraint does not match param length"

        assert np.all(low <= high), "all values in high constraint must be greater than low"

        # scale the bound by the coordinate scale value
        if scale:
            if scale_magnitude is None:
                # NOTE: scaling by coords_scale only makes sense for length scales
                # for variances should scale by obs_scale (**2?)
                # self.coords_scale expected to be 2-d
                low = low / self.coords_scale[0, :]
                high = high / self.coords_scale[0, :]
            else:
                low = low / scale_magnitude
                high = high / scale_magnitude

        # if the current values are outside of tolerances then move them in
        if move_within_tol:
            # require current length scales are more than tol for upper bound
            param_vals[param_vals > (high - tol)] = high[param_vals > (high - tol)] - tol
            # similarly for the lower bound
            param_vals[param_vals < (low + tol)] = low[param_vals < (low + tol)] + tol

        # if the length scale values have changed then assign the new values
        if (np.atleast_1d(original_param.numpy()) != param_vals).any():
            try:
                getattr(obj, param_name).assign(param_vals)
            except ValueError as e:  # Occurs when original_param is a float and not an array
                getattr(obj, param_name).assign(param_vals[0])

        # apply constrains
        # - is it required to provide low/high as tf.constant
        self._apply_param_transform(obj=obj,
                                    bijector="Sigmoid",
                                    param_name=param_name,
                                    low=tf.constant(low),
                                    high=tf.constant(high))


    def _apply_param_transform(self, obj, bijector, param_name, **bijector_kwargs):

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


class GPflowGPRModel(GPflowBaseModel):
    """
    Model based on the GPflow implementation of exact Gaussian process regression (GPR).

    See :class:`~GPSat.models.base_model.BaseGPRModel` for a complete list of attributes and methods.
    """

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
                 # model="GPR",
                 kernel="Matern32",
                 kernel_kwargs=None,
                 mean_function=None,
                 mean_func_kwargs=None,
                 noise_variance=None,
                 likelihood: gpflow.likelihoods.Gaussian = None,
                 **kwargs):
        """
        Parameters
        ----------
        data
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        coords_col
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        obs_col
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        coords
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        obs
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        coords_scale
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        obs_scale
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        obs_mean
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        verbose
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        kernel: str | gpflow.kernels, default "Matern32"
            The kernel used for GPR. We can use the following `GPflow kernels <https://gpflow.github.io/GPflow/develop/api/gpflow/kernels/index.html>`_,
            which can be passed as a string:
            "Cosine", "Exponential", "Matern12", "Matern32", "Matern52", "RationalQuadratic" or "RBF" (equivalently "SquaredExponential").
        kernel_kwargs: dict, optional
            Keyword arguments to be passed to the GPflow kernel specified in ``kernel``.
        mean_function: str | gpflow.mean_functions, optional
            `GPflow mean function <https://gpflow.github.io/GPflow/develop/notebooks/getting_started/mean_functions.html>`_ to model the prior mean.
        mean_func_kwargs: dict, optional
            Keyword arguments to be passed to the GPflow mean function specified in ``mean_function``.
        noise_variance: float, optional
            Variance of Gaussian likelihood. Unnecessary if ``likelihood`` is specified explicitly.
        likelihood: gpflow.likelihoods.Gaussian, optional
            GPflow model for Gaussian likelihood used to model data uncertainty.
            Can use custom GPflow Gaussian likelihood class here.
            Unnecessary if using a vanilla Gaussian likelihood and ``noise_variance`` is specified.

        """
        # TODO: handle kernel (hyper) parameters
        # TODO: remove duplicate __init_ code -

        # --
        # set data, kernel and model
        # --

        super().__init__(data=data,
                         coords_col=coords_col,
                         obs_col=obs_col,
                         coords=coords,
                         obs=obs,
                         coords_scale=coords_scale,
                         obs_scale=obs_scale,
                         obs_mean=obs_mean,
                         verbose=verbose,
                         model="GPR",
                         kernel=kernel,
                         kernel_kwargs=kernel_kwargs,
                         mean_function=mean_function,
                         mean_func_kwargs=mean_func_kwargs,
                         likelihood=likelihood,
                         noise_variance=noise_variance,
                         **kwargs)


    @timer
    def predict(self, coords, full_cov=False, apply_scale=True) -> Dict[str, np.ndarray]:
        """
        Method to generate prediction at given coords.

        Parameters
        ----------
        coords: pandas series | pandas dataframe | list | numpy array
            Coordinate locations where we want to make predictions.
        full_cov: bool, default False
            Flag to determine whether to return a full covariance matrix at the prediction coords or just the marginal variances.
        apply_scale: bool, default True
            If ``True``, ``coords`` should be the raw, untransformed values. If ``False``, ``coords`` must be rescaled by ``self.coords_scale``.
            (see :class:`~GPSat.models.base_model.BaseGPRModel` attributes).

        Returns
        -------
        dict of numpy arrays
            - If ``full_cov = False``, returns a dictionary containing the posterior mean "f*", posterior variance "f*_var"
              and predictive variance "y_var" (i.e. the posterior variance + likelihood variance).
            - If ``full_cov = True``, returns a dictionary containing the posterior mean "f*", posterior marginal variance "f*_var",
              predictive marginal variance "y_var", full posterior covariance "f*_cov" and full predictive covariance "y_cov".

        """
        # TODO: allow for only y, or f to be returned
        # convert coords as needed
        if isinstance(coords, (pd.Series, pd.DataFrame)):
            if self.coords_col is not None:
                coords = coords[self.coords_col].values
            else:
                coords = coords.values
        if isinstance(coords, list):
            coords = np.array(coords)
        # assert isinstance(coords, np.ndarray)
        if len(coords.shape) == 1:
            coords = coords[None, :]  # Is this correct?

        assert isinstance(coords, np.ndarray), f"coords should be an ndarray (one can be converted from)"
        coords = coords.astype(self.coords.dtype)

        if apply_scale:
            coords = coords / self.coords_scale

        y_pred = self.model.predict_y(Xnew=coords, full_cov=False, full_output_cov=False)
        f_pred = self.model.predict_f(Xnew=coords, full_cov=full_cov)

        # TODO: obs_scale should be applied to predictions
        # z = (x-u)/sig; x = z * sig + u

        if not full_cov:
            # TODO: fix f_bar, to align with f* in terms of length, currently returning just one
            out = {
                "f*": f_pred[0].numpy()[:, 0],
                "f*_var": f_pred[1].numpy()[:, 0],
                # "y": y_pred[0].numpy()[:, 0],
                "y_var": y_pred[1].numpy()[:, 0],
                # "f_bar": self.obs_mean[:, 0]
            }

        else:
            f_cov = f_pred[1].numpy()[0, ...]
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
                # "f_bar": self.obs_mean[:, 0]
            }

        # better handle f_bar
        f_bar = self.obs_mean[:, 0]
        if len(f_bar) != len(out["f*"]):
            assert len(f_bar) == 1, f"'f_bar' did not match the length of 'f*' and f_bar len is not, got: {len(f_bar)}"
            out["f_bar"] = np.repeat(f_bar, len(out["f*"]))
        else:
            out["f_bar"] = f_bar

        return out



    # -----
    # Getters/setters for model hyperparameters
    # -----
    def get_objective_function_value(self):
        """Get the negative marginal log-likelihood loss."""
        # take negative as the objective function minimised is the Negative Log Likelihood
        return -self.model.log_marginal_likelihood().numpy()

    # -----
    # Applying constraints on the model hyperparameters
    # -----
    def _set_param_constraints(self,
                               param_name,
                               obj,  # GPflow object. Kernel or likelihood.
                               low,
                               high,
                               move_within_tol=True,
                               tol=1e-8,
                               scale=False,
                               scale_magnitude=None):

        """
        Parameters
        ----------
        """

        assert hasattr(obj, param_name), \
            f"obj of type: {type(obj)}\ndoes not have param_name: {param_name} as attribute"
        # - get original parameter
        original_param = getattr(obj, param_name)

        if isinstance(low, (list, tuple)):
            low = np.array(low, dtype=np.float64)
        elif isinstance(low, (int, np.int64, float)):
            low = np.array([low], dtype=np.float64)

        if isinstance(high, (list, tuple)):
            high = np.array(high, dtype=np.float64)
        elif isinstance(high, (int, np.int64, float)):
            high = np.array([high], dtype=np.float64)

        assert len(low.shape) == 1
        assert len(high.shape) == 1

        # extract the current length scale values
        param_vals = np.atleast_1d(original_param.numpy())

        # - input lengths
        assert len(param_vals) == len(low), "len of low constraint does not match param length"
        assert len(param_vals) == len(high), "len of high constraint does not match param length"

        assert np.all(low <= high), "all values in high constraint must be greater than low"

        # scale the bound by the coordinate scale value
        if scale:
            if scale_magnitude is None:
                # NOTE: scaling by coords_scale only makes sense for length scales
                # for variances should scale by obs_scale (**2?)
                # self.coords_scale expected to be 2-d
                low = low / self.coords_scale[0, :]
                high = high / self.coords_scale[0, :]
            else:
                low = low / scale_magnitude
                high = high / scale_magnitude

        # if the current values are outside of tolerances then move them in
        if move_within_tol:
            # require current length scales are more than tol for upper bound
            param_vals[param_vals > (high - tol)] = high[param_vals > (high - tol)] - tol
            # similarly for the lower bound
            param_vals[param_vals < (low + tol)] = low[param_vals < (low + tol)] + tol

        # if the length scale values have changed then assign the new values
        if (np.atleast_1d(original_param.numpy()) != param_vals).any():
            try:
                getattr(obj, param_name).assign(param_vals)
            except ValueError as e:  # Occurs when original_param is a float and not an array
                getattr(obj, param_name).assign(param_vals[0])

        # apply constrains
        # - is it required to provide low/high as tf.constant
        self._apply_param_transform(obj=obj,
                                    bijector="Sigmoid",
                                    param_name=param_name,
                                    low=tf.constant(low),
                                    high=tf.constant(high))

    @timer
    def set_lengthscales_constraints(self, low, high, move_within_tol=True, tol=1e-8, scale=False,
                                     scale_magnitude=None):
        """
        Sets constraints on the lengthscale hyperparameters.

        Parameters
        ----------
        low: list | int | float
            Minimal value for lengthscales. If specified as a ``list`` type, it should have length D (coordinate dimension) where
            the entries correspond to minimal values of the lengthscale in each dimension in the order given by
            ``self.coords_col`` (see :class:`~GPSat.models.base_model.BaseGPRModel` attributes).
            If ``int`` or ``float``, the same minimal values are assigned to each dimension.
        high: list | int | float
            Same as above, except specifying the maximal values.
        move_within_tol: bool, default True
            If ``True``, ensures that current hyperparam values are within the interval [low+tol, high-tol] for ``tol`` given below.
        tol: float, default 1e-8
            The tol value for when ``move_within_tol = True``.
        scale: bool, default False
            If ``True``, the ``low`` and ``high`` values are set with respect to the *untransformed* coord values.
            If ``False``, they are set with respect to the *transformed* values.
        scale_magnitude: int or float, optional
            The value with which one rescales the coord values if ``scale = True``. If ``None``, it will transform by
            ``self.coords_scale`` (see :class:`~GPSat.models.base_model.BaseGPRModel` attributes).

        """
        self._set_param_constraints(param_name='lengthscales',
                                    obj=self.model.kernel,
                                    low=low, high=high,
                                    move_within_tol=move_within_tol,
                                    tol=tol,
                                    scale=scale,
                                    scale_magnitude=scale_magnitude)

    @timer
    def set_kernel_variance_constraints(self, low, high, move_within_tol=True, tol=1e-8, scale=False,
                                        scale_magnitude=None):
        """
        Sets constraints on the kernel variance.

        Parameters
        ----------
        low: int | float
            Minimal value for kernel variance.
        high: list | int | float
            Maximal value for kernel variance.
        move_within_tol: bool, default True
            If ``True``, ensures that current hyperparam values are within the interval [low+tol, high-tol] for ``tol`` given below.
        tol: float, default 1e-8
            The tol value for when ``move_within_tol = True``.
        scale: bool, default False
            If ``True``, the ``low`` and ``high`` values are set with respect to the *untransformed* coord values.
            If ``False``, they are set with respect to the *transformed* values.
        scale_magnitude: int or float, optional
            The value with which one rescales the coord values if ``scale = True``. If ``None``, it will transform by
            ``self.coords_scale`` (see :class:`~GPSat.models.base_model.BaseGPRModel` attributes).

        """
        self._set_param_constraints(param_name='variance',
                                    obj=self.model.kernel,
                                    low=low, high=high,
                                    move_within_tol=move_within_tol,
                                    tol=tol,
                                    scale=scale,
                                    scale_magnitude=scale_magnitude)

    @timer
    def set_likelihood_variance_constraints(self, low, high, move_within_tol=True, tol=1e-8, scale=False,
                                            scale_magnitude=None):
        """
        Sets constraints on the likelihood variance.

        Parameters
        ----------
        low: int | float
            Minimal value for likelihood variance.
        high: list | int | float
            Maximal value for likelihood variance.
        move_within_tol: bool, default True
            If ``True``, ensures that current hyperparam values are within the interval [low+tol, high-tol] for ``tol`` given below.
        tol: float, default 1e-8
            The tol value for when ``move_within_tol=True``.
        scale: bool, default False
            If ``True``, the ``low`` and ``high`` values are set with respect to the *untransformed* coord values.
            If ``False``, they are set with respect to the *transformed* values.
        scale_magnitude: int or float, optional
            The value with which one rescales the coord values if ``scale=True``. If ``None``, it will transform by
            ``self.coords_scale`` (see :class:`~GPSat.models.base_model.BaseGPRModel` attributes).

        """
        self._set_param_constraints(param_name='variance',
                                    obj=self.model.likelihood,
                                    low=low, high=high,
                                    move_within_tol=move_within_tol,
                                    tol=tol,
                                    scale=scale,
                                    scale_magnitude=scale_magnitude)

    def _apply_param_transform(self, obj, bijector, param_name, **bijector_kwargs):

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


class GPflowSGPRModel(GPflowBaseModel):
    """
    Model using sparse GPR method to handle data size beyond capacity for exact GPR. This introduces a set of M pseudo data points
    referred to as the inducing points, which summarises information contained in the original dataset (see [T'09] for more details).

    Choosing a smaller number of inducing points, one is able to handle large data size up to order ~O(1e5). However, the prediction
    quality may also deteriorate with fewer inducing points so it is necessary to tune the number of inducing points to strike a good
    balance between efficiency and accuracy.

    See :class:`~GPSat.models.base_model.BaseGPRModel` for a complete list of attributes and methods.

    Notes
    -----
    - This is sub-classed from :class:`~GPSat.models.gpflow_models.GPflowGPRModel` and uses the same
      :func:`~GPSat.models.gpflow_models.GPflowGPRModel.predict()` method.
    - Has O(NM^2) computational complexity and O(NM) memory scaling.
    - Several techniques for inducing point selection exists (e.g. see `this GPflow tutorial <https://gpflow.github.io/GPflow/develop/notebooks/getting_started/large_data.html>`_),
      however we have only implemented the *random selection* method, where inducing points are initialised as M random sub-samples of
      the training data.
    
    References
    ----------
    \[T'09\] Titsias, Michalis. "Variational learning of inducing variables in sparse Gaussian processes." Artificial intelligence and statistics. PMLR, 2009.
    
    """
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
                 kernel="Matern32",
                 num_inducing_points=500,
                 kernel_kwargs=None,
                 mean_function=None,
                 mean_func_kwargs=None,
                 noise_variance=None,  # Variance of Gaussian likelihood
                 likelihood: gpflow.likelihoods.Gaussian=None,
                 **kwargs
                 ):
        """
        Parameters
        ----------
        data
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        coords_col
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        obs_col
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        coords
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        obs
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        coords_scale
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        obs_scale
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        obs_mean
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        verbose
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        kernel
            See :func:`GPflowGPRModel.__init__() <GPSat.models.gpflow_models.GPflowGPRModel.__init__>`
        kernel_kwargs
            See :func:`GPflowGPRModel.__init__() <GPSat.models.gpflow_models.GPflowGPRModel.__init__>`
        mean_function
            See :func:`GPflowGPRModel.__init__() <GPSat.models.gpflow_models.GPflowGPRModel.__init__>`
        mean_func_kwargs
            See :func:`GPflowGPRModel.__init__() <GPSat.models.gpflow_models.GPflowGPRModel.__init__>`
        noise_variance
            See :func:`GPflowGPRModel.__init__() <GPSat.models.gpflow_models.GPflowGPRModel.__init__>`
        likelihood
            See :func:`GPflowGPRModel.__init__() <GPSat.models.gpflow_models.GPflowGPRModel.__init__>`
        num_inducing_points: int, default 500
            The number of inducing points.

        """
        # TODO: handle kernel (hyper) parameters
        # TODO: include options for inducing points (random or grid)

        # --
        # Set inducing points
        # --

        # require num_inducing_points is specified.
        # if set to number of observations than method is not sparse - it's just GPR
        assert num_inducing_points is not None, "num_inducing_points is None, must be specified for SGPR"
        if len(coords) < num_inducing_points:
            # if num_inducing_points is greater than the number of data points,
            # we set it to coincide with the data points
            print("number of inducing points is more than number of data points, "
                  "setting inducing points to data points...")
            inducing_points = coords
        else:
            X = copy(coords)
            np.random.shuffle(X)
            inducing_points = X[:num_inducing_points]

        # --
        # set data, kernel and model
        # --

        super().__init__(data=data,
                         coords_col=coords_col,
                         obs_col=obs_col,
                         coords=coords,
                         obs=obs,
                         coords_scale=coords_scale,
                         obs_scale=obs_scale,
                         obs_mean=obs_mean,
                         verbose=verbose,
                         model="SGPR",
                         kernel=kernel,
                         kernel_kwargs=kernel_kwargs,
                         mean_function=mean_function,
                         mean_func_kwargs=mean_func_kwargs,
                         likelihood=likelihood,
                         noise_variance=noise_variance,
                         inducing_variable=inducing_points,
                         **kwargs)

    # @property
    # def param_names(self) -> list:
    #     """
    #     Returns a list of model hyperparameter names ("lengthscales", "kernel_variance" and "likelihood_variance"),
    #     in addition to "inducing points".
    #     """
    #     return super().param_names + ["inducing_points"]
    #
    # def get_inducing_points(self) -> np.ndarray:
    #     """Get the inducing point locations."""
    #     # get the model values, not those stored in self, although they should be kept the same
    #     # return self.model.inducing_variable.Z
    #     return self.model.inducing_variable.Z.numpy()
    #
    # def set_inducing_points(self, inducing_points):
    #     """
    #     Setter method for inducing point locations.
    #
    #     Parameters
    #     ----------
    #     inducing_points: np.ndarray
    #         Inducing point locations specified as a numpy array of size [M, D].
    #     """
    #     # set the model values, and to self (for reference only?)
    #     self.model.inducing_variable = inducingpoint_wrapper(inducing_points)
    #     self.inducing_points = inducing_points

    def get_objective_function_value(self):
        """Get the ELBO value for current state."""
        return self.model.elbo().numpy()

    @timer
    def optimise_parameters(self,
                            train_inducing_points=False,
                            max_iter=10_000,
                            fixed_params=[],
                            **opt_kwargs):
        """
        Method to optimise the model parameters (kernel hyperparmeters + inducing point locations) using a scipy optimizer
        (``method = L-BFGS-B`` by default).
        
        Parameters
        ----------
        train_inducing_points: bool, default False
            Flag to specify whether to optimise the inducing point locations or not. Setting this to ``True`` may improve results,
            however may also lead to slower convergence.
        max_iter: int, default 10000
            The maximum number of interations permitted for optimisation.
            The optimiser runs either until convergence or until the number of iterations reach ``max_iter``.
        fixed_params: list of str, default []
            Parameters to fix during optimisation. Should be one of "lengthscales", "kernel_variance" and "likelihood_variance".
        opt_kwargs: dict, optional
            Keyword arguments passed to `gpflow.optimizers.Scipy.minimize() <https://gpflow.github.io/GPflow/develop/_modules/gpflow/optimizers/scipy.html#Scipy.minimize>`_.

        Returns
        -------
        bool
            Indication of whether optimisation was successful or not, i.e. converges within the maximum number of iterations set.

        """

        self._fix_hyperparameters(fixed_params)

        # TODO: change this to use param_names: inducing_variable_Z
        if not train_inducing_points:
            set_trainable(self.model.inducing_variable.Z, False)
        
        opt_success = super().optimise_parameters(max_iter, **opt_kwargs)

        return opt_success

 
class GPflowSVGPModel(GPflowGPRModel):
    """
    Model using SVGP (Sparse Variational GP [H'13]) to deal with even larger data size (even when compared to SGPR), in addition to handling 
    non-Gaussian likelihoods.
    
    Key differences with SGPR are (1) stochastic optimisation of parameters via mini-batching of training data,
    and (2) gradient-based optimisation of the variational distribution, parameterised by a mean and cholesky factor of the covariance,
    as opposed to exact computation. The former allows handling of larger data + inducing point sizes and the latter allows handling
    of non-Gaussian likelihoods (see [H'13] for more details).
    
    See :class:`~GPSat.models.base_model.BaseGPRModel` for a complete list of attributes and methods.

    Notes
    -----
    - This is sub-classed from :class:`~GPSat.models.gpflow_models.GPflowGPRModel` and uses the same
      :func:`~GPSat.models.gpflow_models.GPflowGPRModel.predict()` method.
    - Introduces an extra hyperparameter ``minibatch_size`` to be tuned.
    - Has O(BM^2 + M^3) computational complexity and O(BM + M^2) memory scaling, where B is the minibatch size.
    - Saving the variational parameters to the results file may be memory intensive due to the M^2 memory scaling of the cholesky factor.
      Consider leaving them out of the results file when running experiments (TODO: cross reference ModelConfig).

    References
    ----------
    \[H'13\] Hensman, James, Nicolo Fusi, and Neil D. Lawrence. "Gaussian processes for big data." arXiv preprint arXiv:1309.6835 (2013).

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
                 verbose=True,
                 *,
                 kernel="Matern32",
                 num_inducing_points=None,
                 minibatch_size=None,
                 kernel_kwargs=None,
                 mean_function=None,
                 mean_func_kwargs=None,
                 noise_variance=None,
                 likelihood=None,
                 likelihood_kwargs=None,
                 **kwargs):
        """
        Parameters
        ----------
        data
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        coords_col
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        obs_col
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        coords
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        obs
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        coords_scale
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        obs_scale
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        obs_mean
            See :func:`BaseGPRModel.__init__() <GPSat.models.base_model.BaseGPRModel.__init__>`
        kernel
            See :func:`GPflowGPRModel.__init__() <GPSat.models.gpflow_models.GPflowGPRModel.__init__>`
        kernel_kwargs
            See :func:`GPflowGPRModel.__init__() <GPSat.models.gpflow_models.GPflowGPRModel.__init__>`
        mean_function
            See :func:`GPflowGPRModel.__init__() <GPSat.models.gpflow_models.GPflowGPRModel.__init__>`
        mean_func_kwargs
            See :func:`GPflowGPRModel.__init__() <GPSat.models.gpflow_models.GPflowGPRModel.__init__>`
        noise_variance
            See :func:`GPflowGPRModel.__init__() <GPSat.models.gpflow_models.GPflowGPRModel.__init__>`
        likelihood: str or gpflow.likelihoods, optional
            A `GPflow likelihoods object <https://gpflow.github.io/GPflow/develop/api/gpflow/likelihoods/index.html#>`_
            for modelling the likelihood. This is not necessarily a Gaussian.
            For available GPflow likelihoods, pass a string (e.g. ``likelihood = "StudentT"``).
            However if not specified, it will default to a Gaussian likelihood with variance given by ``noise_variance``.
        likelihood_kwargs: dict, optional
            Keyword arguments passed to ``likelihood``.
        num_inducing_points: int, optional
            The number of inducing points. If not specified, it will set the inducing points to be the data points,
            in which case the algorithm becomes equivalent to VGP.
        minibatch_size: int, optional
            The size of minibatch used for stochastic estimation of the loss function. Using smaller 
            batch sizes will result in increased per-iteration efficiency, however optimisation becomes more noisy.
            If not specified, it will not apply minibatching.

        """

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
        # data
        # ---
        X = self.coords
        Y = self.obs
        self.train_dataset = tf.data.Dataset.from_tensor_slices((X, Y)).repeat().shuffle(X.shape[0])

        if minibatch_size is None:
            self.minibatch_size = self.coords.shape[0] # Set to full-batch gradient descent if minibatch size is not specified
        else:
            self.minibatch_size = minibatch_size

        # ---
        # model
        # ---
        if likelihood is None:
            likelihood = gpflow.likelihoods.Gaussian(noise_variance)
        elif isinstance(likelihood, str):
            # TODO: check on various gpflow likelihoods
            likelihood = getattr(gpflow.likelihoods, likelihood)
            likelihood_kwargs = {} if likelihood_kwargs is None else likelihood_kwargs
            likelihood = likelihood(**likelihood_kwargs)
        else:
            # TODO: to implement custom likelihood case
            raise NotImplementedError

        # TODO: allow for this method to take in additional arguments(?)
        self.model = gpflow.models.SVGP(kernel=kernel,
                                        mean_function=mean_function,
                                        likelihood=likelihood,
                                        inducing_variable=self.inducing_points,
                                        num_data=self.coords.shape[0])

    def get_objective_function_value(self):
        """Get the ELBO averaged over minibatches."""
        elbo = self.model.elbo
        train_iter = iter(self.train_dataset.batch(self.minibatch_size))
        num_batches = self.coords.shape[0] // self.minibatch_size
        evals = [elbo(minibatch).numpy() for minibatch in itertools.islice(train_iter, np.min([100, num_batches]))]
        return np.mean(evals)

    def _fix_variational_parameters(self, fixed_params):
        if "inducing_points" in fixed_params:
            set_trainable(self.model.inducing_variable.Z, False)
        if "inducing_mean" in fixed_params:
            set_trainable(self.model.q_mu, False)
        if "inducing_chol" in fixed_params:
            set_trainable(self.model.q_sqrt, False)

    @timer
    def optimise_parameters(self,
                            train_inducing_points=False,
                            natural_gradients=False,
                            fixed_params=[],
                            gamma=0.1,
                            learning_rate=1e-2,
                            max_iter=10_000,
                            persistence=100,
                            check_every=10,
                            early_stop=True,
                            verbose=False):
        """
        Method to optimise the model parameters (kernel hyperparmeters + variational parameters).
        We use the Adam optimiser for stochastic optimisation of the model parameters.

        Parameters
        ----------
        train_inducing_points: bool, default False
            Flag to determine whether or not to optimise inducing point locations.
        natural_gradients: bool, default False
            Option to use natural gradients to optimise the variational parameters (inducing mean and cholesky).
            Previous investigations indicate benefits of using them over using Adam to optimise all parameters.
            (see more details `here <https://gpflow.github.io/GPflow/develop/notebooks/advanced/natural_gradients.html#Natural-gradients>`_)
        gamma: float, default 0.1
            Step length for natural gradient. When not using minibatches, best to set ``gamma = 1.0``.
            However, empirically shown to be better using smaller ``gamma`` e.g. 0.1 when minibatching.
        fixed_params: list of str, default []
            Parameters to fix during optimisation. Should be one of "lengthscales", "kernel_variance", "likelihood_variance",
            "inducing points", "inducing_mean" and "inducing_chol".
        learning_rate: float, default 1e-2
            Learning rate for Adam optimizer.
        max_iter: int, default 10000
            The maximum number of interations permitted for optimisation.
            The optimiser runs either until convergence (see discussion on the convergence criterion in **Notes** below)
            or until the number of iterations reach ``max_iter``.
        early_stop: bool, default True
            Flag to set early stopping criterion (see **Notes** below). If ``False``, it will run until number of iterations
            reach ``max_iter``, which can be quite slow.
        persistence: int, default 100
            See **Notes** below.
        check_every: int, default 10
            See **Notes** below.
        verbose: bool, default False
            Set verbosity of model optimisation. If ``True``, displays the loss every ``check_every`` steps.
        
        Returns
        -------
        bool
            Indication of whether optimisation was successful or not.

        Notes
        -----
        Since we use stochastic optimisation, traditional convergence criterion to stop early does not apply here.
        We instead devise a stopping criterion as follows:

        - Check the ELBO every ``check_every`` iterations.
        - If the ELBO does not improve after ``persistence`` iterations, stop optimisation.

        This stopping criterion will be enabled if ``early_stop`` is set to ``True``.

        """

        if (not train_inducing_points) and ("inducing_points" not in fixed_params):
            fixed_params.append("inducing_points")
        
        self._fix_hyperparameters(fixed_params)
        self._fix_variational_parameters(fixed_params)

        if natural_gradients:
            # make q_mu and q_sqrt non training to adam
            gpflow.utilities.set_trainable(self.model.q_mu, False)
            gpflow.utilities.set_trainable(self.model.q_sqrt, False)

            # select the variational parameters for natural gradients
            variational_vars = [(self.model.q_mu, self.model.q_sqrt)]
            natgrad_opt = gpflow.optimizers.NaturalGradient(gamma=gamma)

        # parameters for adam to train
        adam_vars = self.model.trainable_variables
        adam_opt = tf.optimizers.Adam(learning_rate)

        train_iter = iter(self.train_dataset.batch(self.minibatch_size))
        loss_fn = self.model.training_loss_closure(train_iter, compile=True)

        @tf.function
        def optimisation_step():
            """
            Apply natural gradients to update the inducing mean + covariance (q_mu, q_sqrt)
            and the adam optimizer to update the model hyperparameters / inducing point locations.
            This allows for better + faster convergence.
            TODO: Double check. Just using adam seems more stable and quicker
            """
            if natural_gradients:
                natgrad_opt.minimize(loss_fn, variational_vars)
            adam_opt.minimize(loss_fn, adam_vars)

        # initialise the maximum elbo
        max_elbo = -np.inf
        stopped_early = False
        max_count = 0
        for step in range(max_iter):
            optimisation_step()
            if step % check_every == 0:
                # loss_fn() will give the training_loss (negative elbo) for current batch
                elbo = -loss_fn().numpy()
                if np.isnan(elbo):
                    print("Optimisation failed...")
                    stopped_early = True
                    opt_success = False
                    break
                if verbose:
                    print(f"step: {step},  elbo: {elbo:.2f}")
                # check if new elbo estimate is larger than previous
                if (elbo > max_elbo) and early_stop:
                    max_elbo = elbo
                    max_count = 0
                else:
                    max_count += check_every
                    # stop optimisation if elbo hasn't increased for [persistence] steps
                    if (max_count >= persistence) and early_stop:
                        print("objective did not improve stopping")
                        stopped_early = True
                        opt_success = True
                        break

        opt_success = opt_success if stopped_early else np.nan

        return opt_success

    @property
    def param_names(self) -> list:
        """
        Returns a list of model hyperparameter names ("lengthscales", "kernel_variance" and "likelihood_variance"),
        in addition to the variational hyperparameters ("inducing points", "inducing_mean" and "inducing_chol").

        The "inducing_mean" and "inducing_chol" are respectively, the mean and cholesky factor of the covariance of
        the Gaussian variational distribution used to approximate the true posterior distribution.
        
        """
        return super().param_names + ["inducing_points", "inducing_mean", "inducing_chol"]

    def get_inducing_points(self) -> np.ndarray:
        """Get the inducing point locations."""
        # get the model values, not those stored in self, although they should be kept the same
        # return self.model.inducing_variable.Z
        return self.model.inducing_variable.Z.numpy()

    def set_inducing_points(self, inducing_points):
        """
        Setter method for inducing point locations.
        
        Parameters
        ----------
        inducing_points: np.ndarray
            Inducing point locations specified as a numpy array of size [M, D],
            where D is the input dimension size.

        """
        # set the model values, and to self (for reference only?)
        self.model.inducing_variable = inducingpoint_wrapper(inducing_points)
        self.inducing_points = inducing_points

    def get_inducing_mean(self) -> np.ndarray:
        """Get the mean of the variational distribution."""
        return self.model.q_mu.numpy()

    def set_inducing_mean(self, q_mu):
        """
        Setter method for the inducing mean.
        
        Parameters
        ----------
        q_mu: np.ndarray
            Inducing mean values specified as a numpy array of size [M, 1].

        """
        self.model.q_mu = Parameter(q_mu, dtype=default_float())

    def get_inducing_chol(self) -> np.ndarray:
        """Get the cholesky factor of the covariace of the variational distribution."""
        return self.model.q_sqrt.numpy()

    def set_inducing_chol(self, q_sqrt):
        """
        Setter method for the inducing cholesky factor.
        
        Parameters
        ----------
        q_sqrt: np.ndarray
            Inducing cholesky values specified as a numpy array of size [1, M, M].

        """
        self.model.q_sqrt = Parameter(q_sqrt, transform=triangular())


if __name__ == "__main__":

    # TODO: find notebooks for SVGP, SGPR, and GPR stand alone examples?

    import matplotlib.pyplot as plt
    from GPSat.plot_utils import plot_gpflow_minimal_example

    # --
    # toy data
    # ---

    X = np.array(
        [
            [0.865], [0.666], [0.804], [0.771], [0.147], [0.866], [0.007], [0.026],
            [0.171], [0.889], [0.243], [0.028],
        ]
    )
    Y = np.array(
        [
            [1.57], [3.48], [3.12], [3.91], [3.07], [1.35], [3.80], [3.82], [3.49],
            [1.30], [4.00], [3.82],
        ]
    )

    # initialise the model
    m = GPflowSGPRModel(coords=X, obs=Y)

    org_params = m.get_parameters()
    optimised = m.optimise_parameters()
    opt_params = m.get_parameters()

    tmp = {k: float(v) if len(v.shape) == 0 else v for k, v in org_params.items()}
    m.set_parameters(**tmp)

    m = GPflowGPRModel(coords=X, obs=Y)


    org_params = m.get_parameters()

    optimised = m.optimise_parameters()

    opt_params = m.get_parameters()

    tmp = {k: float(v) if len(v.shape) == 0 else v for k,v in org_params.items()}

    # check the assigning or parameters works as expected
    # - namely the floats are converted back to 0-dimensional arrays
    m.set_parameters(**tmp)
    #
    res = plot_gpflow_minimal_example(GPflowGPRModel,
                                      model_init=None,
                                      opt_params=None,
                                      pred_params=None)





