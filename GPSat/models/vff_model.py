import inspect
import gpflow
import numpy as np

from GPSat.decorators import timer
from GPSat.models import BaseGPRModel
from GPSat.models.gpflow_models import GPflowGPRModel
from GPSat.vff import GPR_kron

from copy import copy
from typing import Union, List


class GPflowVFFModel(GPflowGPRModel):
    @timer
    def __init__(self,
                 data=None,
                 coords_col=None,
                 obs_col=None,
                 coords=None,
                 coords_scale=None,
                 obs=None,
                 obs_scale=None,
                 obs_mean=None,
                 *,
                 kernels="Matern32",
                 num_inducing_features: Union[int, list]=None,
                 kernel_kwargs=None,
                 mean_function=None,
                 mean_func_kwargs=None,
                 domain_size: Union[float, List[float]]=None,
                 expert_loc=None,
                 **kwargs):
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

        # ---
        # model
        # ---
        if isinstance(domain_size, (int, float)):
            domain_size = [domain_size for _ in range(self.coords.shape[1])]

        assert len(domain_size) == self.coords.shape[1], "length of margin list must match number of coordinate dimensions"

        a_list = []; b_list = []
        if domain_size is None:
            for i, coords in enumerate(self.coords.T):
                a_list.append(coords.min()-1e-8)
                b_list.append(coords.max()+1e-8)
        else:
            # Set expert location to center of data if not specified
            if expert_loc is None:
                expert_loc = np.mean(self.coords, axis=0) * self.coords_scale.squeeze()
            # Set boundaries [a,b] of the domain of inducing features in each dimension
            for i, coords in enumerate(self.coords.T):
                a = (expert_loc[i] - domain_size[i])/self.coords_scale[0,i]
                b = (expert_loc[i] + domain_size[i])/self.coords_scale[0,i]
                # Ensure that the data lies in the domain of inducing features
                a = a if a < coords.min() else coords.min()-1e-8
                b = b if b > coords.max() else coords.max()+1e-8
                a_list.append(a)
                b_list.append(b)

        if isinstance(num_inducing_features, int):
            m_list = [np.arange(num_inducing_features)]
        elif isinstance(num_inducing_features, list):
            m_list = [np.arange(num) for num in num_inducing_features]

        self.model = GPR_kron(data=(self.coords, self.obs),
                              ms=m_list,
                              a=a_list,
                              b=b_list,
                              kernel_list=kernels)

    def get_objective_function_value(self):
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

    @timer
    def set_lengthscales_constraints(self, low, high, move_within_tol=True, tol=1e-8, scale=False, scale_magnitude=None):
        for i, kern in enumerate(self.model.kernels):
            self._set_param_constraints(param_name='lengthscales',
                                        obj=kern,
                                        low=low[i], high=high[i],
                                        move_within_tol=move_within_tol,
                                        tol=tol,
                                        scale=scale,
                                        scale_magnitude=self.coords_scale[0, i])

    @timer
    def set_kernel_variance_constraints(self, low, high, move_within_tol=True, tol=1e-8, scale=False, scale_magnitude=None):
        alpha = 1/self.coords.shape[1]
        for i, kern in enumerate(self.model.kernels):
            self._set_param_constraints(param_name='variance',
                                        obj=kern,
                                        low=low**alpha,
                                        high=high**alpha,
                                        move_within_tol=move_within_tol,
                                        tol=tol,
                                        scale=scale,
                                        scale_magnitude=scale_magnitude)



        