import inspect
import warnings

import pandas as pd
import gpflow
import numpy as np
import itertools
from copy import copy

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.client import device_lib

from gpflow.utilities import set_trainable
from gpflow.models.util import inducingpoint_wrapper

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
                 *,
                 kernel="Matern32",
                 kernel_kwargs=None,
                 mean_function=None,
                 mean_func_kwargs=None,
                 noise_variance=None, # Variance of Gaussian likelihood. Unnecessary if likelihood is specified
                 likelihood: gpflow.likelihoods.Gaussian=None,
                 **kwargs):
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
            kernel = getattr(gpflow.kernels, kernel)

            # check signature parameters
            kernel_signature = inspect.signature(kernel).parameters

            # see if it takes lengthscales
            # - want to initialise with appropriate length (one length scale per coord)
            if ("lengthscales" in kernel_signature) & ("lengthscales" not in kernel_kwargs):
                kernel_kwargs['lengthscales'] = np.ones(self.coords.shape[1])
                print(f"setting lengthscales to: {kernel_kwargs['lengthscales']}")

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
        if isinstance(coords, (pd.Series, pd.DataFrame)):
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
            # TODO: fix f_bar, to align with f* in terms of length, currently returning just one
            out = {
                "f*": f_pred[0].numpy()[:, 0],
                "f*_var": f_pred[1].numpy()[:, 0],
                # "y": y_pred[0].numpy()[:, 0],
                "y_var": y_pred[1].numpy()[:, 0],
                # "f_bar": self.obs_mean[:, 0]
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

    @timer
    def optimise_parameters(self, max_iter=10_000, **opt_kwargs):

        opt = gpflow.optimizers.Scipy()
        m = self.model
        opt_logs = opt.minimize(m.training_loss,
                                m.trainable_variables,
                                options=dict(maxiter=max_iter),
                                **opt_kwargs)

        if not opt_logs['success']:
            print("*" * 10)
            print("optimization failed!")
            # TODO: determine if should return None for failed optimisation
            # return None

        return opt_logs['success']

    # -----
    # Getters/setters for model hyperparameters
    # -----
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
        # expect float, allow for 1d ndarray of length 1
        if isinstance(kernel_variance, np.ndarray):

            assert (len(kernel_variance) == 1) & (len(kernel_variance.shape) == 1), \
                f"set_kernel_variance expected to receive float, or np.array with len(1), shape:(1,), got" \
                f"len: {len(kernel_variance)}, shape: {kernel_variance.shape}"

            kernel_variance = kernel_variance[0]

        self.model.kernel.variance.assign(kernel_variance)

    def set_likelihood_variance(self, likelihood_variance):
        # expect float, allow for 1d ndarray of length 1
        if isinstance(likelihood_variance, np.ndarray):

            assert (len(likelihood_variance) == 1) & (len(likelihood_variance.shape) == 1), \
                f"set_likelihood_variance expected to receive float, or np.array with len(1), shape:(1,), got" \
                f"len: {len(likelihood_variance)}, shape: {likelihood_variance.shape}"

            likelihood_variance = likelihood_variance[0]

        # HACK: to handle setting variance below variance_lower_bound attribute
        if hasattr(self.model.likelihood, "variance_lower_bound"):
            if likelihood_variance < self.model.likelihood.variance_lower_bound:
                warnings.warn("\n***\ntrying to set likelihood_variance to value less than "
                              "model.likelihood.variance_lower_bound\nwill set to variance_lower_bound\n***\n")
                likelihood_variance = self.model.likelihood.variance_lower_bound

        self.model.likelihood.variance.assign(likelihood_variance)

    # -----
    # Applying constraints on the model hyperparameters
    # -----
    def _set_param_constraints(self,
                               param_name,
                               obj, # GPflow object. Kernel or likelihood.
                               low,
                               high,
                               move_within_tol=True,
                               tol=1e-8,
                               scale=False,
                               scale_magnitude=None):
                               
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
            except ValueError as e: # Occurs when original_param is a float and not an array
                getattr(obj, param_name).assign(param_vals[0])

        # apply constrains
        # - is it required to provide low/high as tf.constant
        self._apply_param_transform(obj=obj,
                                    bijector="Sigmoid",
                                    param_name=param_name,
                                    low=tf.constant(low),
                                    high=tf.constant(high))

    @timer
    def set_lengthscales_constraints(self, low, high, move_within_tol=True, tol=1e-8, scale=False, scale_magnitude=None):
        self._set_param_constraints(param_name='lengthscales',
                                    obj=self.model.kernel,
                                    low=low, high=high,
                                    move_within_tol=move_within_tol,
                                    tol=tol,
                                    scale=scale,
                                    scale_magnitude=scale_magnitude)

    @timer
    def set_kernel_variance_constraints(self, low, high, move_within_tol=True, tol=1e-8, scale=False, scale_magnitude=None):
        self._set_param_constraints(param_name='variance',
                                    obj=self.model.kernel,
                                    low=low, high=high,
                                    move_within_tol=move_within_tol,
                                    tol=tol,
                                    scale=scale,
                                    scale_magnitude=scale_magnitude)

    @timer
    def set_likelihood_variance_constraints(self, low, high, move_within_tol=True, tol=1e-8, scale=False, scale_magnitude=None):
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


class GPflowSGPRModel(GPflowGPRModel):
    """
    Model using SGPR (Sparse GPR. Titsias 2009)
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
                 *,
                 kernel="Matern32",
                 num_inducing_points=None,
                 train_inducing_points=False,
                 kernel_kwargs=None,
                 mean_function=None,
                 mean_func_kwargs=None,
                 noise_variance=None,  # Variance of Gaussian likelihood
                 likelihood: gpflow.likelihoods.Gaussian=None,
                 **kwargs
                 ):
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
            if ("lengthscales" in kernel_signature) & ("lengthscales" not in kernel_kwargs):
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
        # model
        # ---

        # TODO: allow for model type (e.g. "GPR" to be specified as input?)
        self.model = gpflow.models.SGPR(data=(self.coords, self.obs),
                                        kernel=kernel,
                                        mean_function=mean_function,
                                        noise_variance=noise_variance,
                                        inducing_variable=self.inducing_points,
                                        likelihood=likelihood)

    @property
    def param_names(self) -> list:
        return super().param_names + ["inducing_points"]

    def get_inducing_points(self):
        # get the model values, not those stored in self, although they should be kept the same
        # return self.model.inducing_variable.Z
        return self.model.inducing_variable.Z.numpy()

    def set_inducing_points(self, inducing_points):

        # set the model values, and to self (for reference only?)
        self.model.inducing_variable = inducingpoint_wrapper(inducing_points)
        self.inducing_points = inducing_points


    def get_objective_function_value(self):
        """get the marginal log likelihood"""
        return self.model.elbo().numpy()

    @timer
    def optimise_parameters(self,
                            train_inducing_points=False,
                            max_iter=10_000,
                            **opt_kwargs):

        if not train_inducing_points:
            set_trainable(self.model.inducing_variable.Z, False)
        
        opt_success = super().optimise_parameters(max_iter, **opt_kwargs)

        return opt_success

 
class GPflowSVGPModel(GPflowGPRModel):
    """
    Model using SVGP (Hensman et al. )
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
                 *,
                 kernel="Matern32",
                 num_inducing_points=None,
                 minibatch_size=None,
                 kernel_kwargs=None,
                 mean_function=None,
                 mean_func_kwargs=None,
                 noise_variance=None,
                 likelihood=None,
                 **kwargs):
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

        self.model = gpflow.models.SVGP(kernel=kernel,
                                        mean_function=mean_function,
                                        likelihood=likelihood,
                                        inducing_variable=self.inducing_points,
                                        num_data=self.coords.shape[0])

    def get_objective_function_value(self):
        """get the elbo averaged over minibatches"""
        elbo = self.model.elbo
        train_iter = iter(self.train_dataset.batch(self.minibatch_size))
        num_batches = self.coords.shape[0] // self.minibatch_size
        evals = [elbo(minibatch).numpy() for minibatch in itertools.islice(train_iter, np.min([100, num_batches]))]
        return np.mean(evals)

    @timer
    def optimise_parameters(self,
                            train_inducing_points=False,
                            natural_gradients=False,
                            gamma=0.1,
                            learning_rate=1e-2,
                            max_iter=10_000,
                            persistence=100,
                            check_every=10,
                            early_stop=True,
                            verbose=False):
        """
        :param gamma: step length for natural gradient
        :param learning_rate: learning rate for Adam optimizer
        """

        if not train_inducing_points:
            set_trainable(self.model.inducing_variable.Z, False)

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

