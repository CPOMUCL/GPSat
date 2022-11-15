import gpflow
import numpy as np
import xarray as xr
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Union, Type


# ------- Base class ---------

class LocalGPExpert(ABC):
    def __init__(self, parameters: Dict):
        self._init_parameters = parameters
        self.parameters = parameters
        self.gp_model = None
    
    def compile(self, data: Tuple[np.ndarray, np.ndarray], parameters: Union[Dict, None]):
        """
        Compile GP model
        """
        X, y = data
        if parameters is not None:
            self.parameters = parameters
        else:
            self.parameters = self._init_parameters
        self.gp_model = self._build_model(X, y)

    @abstractmethod
    def _build_model(self, X, y):
        """
        Method to instantiate GP model

        Output:
        A GP model object
        """

    @abstractmethod
    def get_parameters(self):
        """
        Method to get the parameters of the model

        Output:
        Parameters of the model given by a dictionary whose keys are the parameter labels
        and the values are the corresponding values
        """

    @abstractmethod
    def optimise(self):
        """
        Method to optimise hyperparameters of the model
        """

    @abstractmethod
    def predict(self, gridpoints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method to get the predicted mean and variance of the model

        Output: Tuple of form (mean, variance)
        """


# ------- GPflow experts ---------

class GPflowGPRExpert(LocalGPExpert):
    def __init__(self,
                 kernel_cls: Type[gpflow.kernels.Stationary],
                 parameters: Dict,
                 max_iter: int=10_000,
                 fixed_hyperparams: List=None):

        hyperparams = ['lengthscale_x', 'lengthscale_y', 'lengthscale_t', 'kernel_variance', 'observation_variance']
        assert list(parameters.keys()) == hyperparams
        self.opt = gpflow.optimizers.Scipy()
        self._model_cls = gpflow.models.GPR
        self._kernel_cls = kernel_cls
        self._mean_fn = gpflow.mean_functions.Constant
        self._fixed_hyperparams = fixed_hyperparams
        self._max_iter = max_iter

        super().__init__(parameters)

    def _get_kernel(self, hyperparameters):
        ls = (hyperparameters['lengthscale_x'], hyperparameters['lengthscale_y'], hyperparameters['lengthscale_t'])
        var = hyperparameters['kernel_variance']
        kernel = self._kernel_cls(lengthscales=ls, variance=var)
        return kernel

    def _get_mean(self, y):
        """ Only implemented for constant mean
        TODO: extend to more general means
        """
        # mean = self._mean_fn(c=np.array([np.mean(y)]))
        mean = self._mean_fn(c=np.array([0]))
        gpflow.set_trainable(mean.c, False)
        return mean

    def _set_untrainable(self, model):
        hyperparams = self.get_parameters(model)
        if self._fixed_hyperparams is None:
            pass
        else:
            for key, val in hyperparams.items():
                if key in self._fixed_hyperparams:
                    gpflow.set_trainable(val, False)
    
    def _build_model(self, X, y):
        hyperparameters = self.parameters
        kernel = self._get_kernel(hyperparameters)
        mean = self._get_mean(y)
        model = self._model_cls(data=(X, y), kernel=kernel, mean_function=mean)
        model.likelihood.variance.assign(hyperparameters['observation_variance'])
        self._set_untrainable(model)
        return model

    def get_parameters(self, model=None):
        if model is None:
            model = self.gp_model
        hyperparam_keys = ['lengthscale_x', 'lengthscale_y', 'lengthscale_t', 'kernel_variance', 'observation_variance']
        hyperparam_vals = [*model.kernel.lengthscales, model.kernel.variance, model.likelihood.variance]
        hyperparams = dict(zip(hyperparam_keys, hyperparam_vals))
        return hyperparams

    def optimise(self):
        model = self.gp_model
        self.opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=self._max_iter))

    def predict(self, gridpoints):
        mean, var = self.gp_model.predict_f(gridpoints)
        return mean.numpy()[0,0], var.numpy()[0,0]


class GPflowSGPRExpert(LocalGPExpert):
    ...


class GPflowSVGPExpert(LocalGPExpert):
    ...

    