import inspect
import gpflow
import numpy as np

from PyOptimalInterpolation.decorators import timer
from PyOptimalInterpolation.models import BaseGPRModel
from PyOptimalInterpolation.models.gpflow_models import GPflowGPRModel

# Clone from https://github.com/HJakeCunningham/ASVGP
from ASVGP.asvgp.gpr import GPR_kron
from ASVGP.asvgp.basis import B1Spline, B2Spline, B3Spline

from copy import copy
from typing import Union


class GPflowASVGPModel(GPflowGPRModel):
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
                 kernels="Matern32",
                 spline_order: int=1, # To be determined by Matern order?
                 num_inducing_features: Union[int, list]=None,
                 kernel_kwargs=None,
                 mean_function=None,
                 mean_func_kwargs=None,
                 margin: Union[float, list]=None):
        # TODO: handle kernel (hyper) parameters
        # TODO: Currently does not handle variable ms + does not incorporate mean function
        # NOTE: kernel_kwargs here is a list of kernel kwargs (dict) per dimension.
        #       Also admits a single dict, meaning the kernel kwargs will be the same across dimensions.

        """
        Args:
            num_inducing_features: Number of Fourier features. If int, the same number of Fourier features
                                   is specified per dimension. If list, the i-th entry corresponds to the 
                                   number of Fourier feature in dimension i.
            margin: The amount by which we increase the domain size (relative to scaled coordinates), necessary for VFF.
                    If float, the domain size is increased by the same amount per dimension.
        """

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

        assert kernels is not None, "kernel was not provided"
        assert num_inducing_features is not None, "Number of inducing points per dimension not specified"

        # if kernel is str: get function
        if isinstance(kernels, str):
            # get the kernel function (still requires
            kernel = getattr(gpflow.kernels, kernels)
        else:
            # TODO: Implement other case
            ...

        # if additional kernel kwargs not provide use empty dict
        if kernel_kwargs is None:
            kernel_kwargs = []

            # check signature parameters
            kernel_signature = inspect.signature(kernel).parameters

            # dee if it takes lengthscales
            # - want to initialise with appropriate length (one length scale per coord)
            if ("lengthscales" in kernel_signature) & ("lengthscale" not in kernel_kwargs):
                for _ in range(self.coords.shape[1]): # Define 1D kernels
                    kernel_kwargs.append(dict(lengthscales=1.0))
        
        elif isinstance(kernel_kwargs, dict):
            kernel_kwargs_ = copy(kernel_kwargs)
            kernel_kwargs = [kernel_kwargs_ for _ in range(self.coords.shape[1])]
        
        assert len(kernel_kwargs) == self.coords.shape[1]

        # initialise kernels
        kernels = [kernel(**kernel_kwargs[i]) for i in range(self.coords.shape[1])]

        # --
        # prior mean function
        # --

        if isinstance(mean_function, str):
            if mean_func_kwargs is None:
                mean_func_kwargs = {}
            mean_function = getattr(gpflow.mean_functions, mean_function)(**mean_func_kwargs)

        # --
        # set spline basis
        # --
        if margin is None:
            margin = [1e-8 for _ in range(self.coords.shape[1])]
        elif isinstance(margin, (int, float)):
            margin = [margin for _ in range(self.coords.shape[1])]

        assert len(margin) == self.coords.shape[1], "length of margin list must match number of coordinate dimensions"

        a_list = []; b_list = []
        for i, coords in enumerate(self.coords.T):
            a_list.append(coords.min()-margin[i])
            b_list.append(coords.max()+margin[i])

        if isinstance(num_inducing_features, int):
            m_list = [num_inducing_features]
        elif isinstance(num_inducing_features, list):
            m_list = [num for num in num_inducing_features]

        bases = [self._get_basis(a, b, m, k) for (a, b, m, k) in zip(a_list, b_list, m_list, kernels)]

        # ---
        # model
        # ---
        self.model = GPR_kron(data=(self.coords, self.obs),
                              kernels=kernels,
                              bases=bases)

    def _get_basis(self, a, b, m, kernel):
        """Returns spline basis appropriate for the Matern kernel order"""
        if isinstance(kernel, gpflow.kernels.Matern12):
            return B1Spline(a, b, m)
        elif isinstance(kernel, gpflow.kernels.Matern32):
            return B2Spline(a, b, m)
        elif isinstance(kernel, gpflow.kernels.Matern52):
            return B3Spline(a, b, m)
        else:
            return NotImplementedError

    def get_objective_function_value(self):
        """get the marginal log likelihood"""
        return self.model.elbo().numpy()

    def get_lengthscales(self):
        return np.array([kernel.lengthscales.numpy() for kernel in self.model.kernels])

    def get_kernel_variance(self):
        return np.prod([float(kernel.variance.numpy()) for kernel in self.model.kernels])

    def set_lengthscales(self, lengthscales):
        for kernel, ls in zip(self.model.kernels, lengthscales):
            kernel.lengthscales.assign(ls)

    def set_kernel_variance(self, kernel_variance):
        """
        Assignment of variance to each 1D kernel is non-unique.
        We assign equal weights to each kernel for simplicity.
        """
        dim = self.coords.shape[1]
        var = kernel_variance**(1/dim)
        for kernel in self.model.kernels:
            kernel.variance.assign(var)

    def set_lengthscale_constraints(self, low, high, move_within_tol=True, tol=1e-8, scale=False):
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

        for i, kern in enumerate(self.model.kernels):
            super().set_lengthscale_constraints(low[i], high[i], kern, move_within_tol,
                                                tol, scale, scale_magnitude=self.coords_scale[0,i])

# TODO: Change AS-VGP code so that it does predict_f and predict_f_sparse properly

