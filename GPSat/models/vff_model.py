import inspect
import gpflow
import numpy as np

from GPSat.decorators import timer
from GPSat.models import BaseGPRModel
from GPSat.models.gpflow_models import GPflowGPRModel
from GPSat.vff import GPR_kron

from copy import copy
from typing import Union, List, Optional


class GPflowVFFModel(GPflowGPRModel):
    """
    GPSat model using VFF (variational Fourier features) to handle large data size in low dimensions.
    
    This is a prime example of the *interdomain approach* where pseudo data points (called *inducing features*)
    are placed on a transformed domain instead of the physical domain. For VFF, these inducing features
    are placed in the frequency domain, which can achieve better scaling in the number of data points 
    compared to SGPR, owing to the orthogonality of the sinusoidal basis functions (see [H'17] for more details).
    
    However, VFF requires using a separable kernel in each dimension, resulting in poor scaling in the
    input dimensions. Thus, benefits are usually seen for lower dimensional problems such as 1D, 2D and
    possibly 3D in some cases.

    See :class:`~GPSat.models.base_model.BaseGPRModel` for a complete list of attributes and methods.

    Notes
    -----
    - This is sub-classed from :class:`~GPSat.models.gpflow_models.GPflowGPRModel` and uses the same
      :func:`~GPSat.models.gpflow_models.GPflowGPRModel.predict()` method.
    - Likewise, it uses the same :func:`~GPSat.models.gpflow_models.GPflowGPRModel.get_likelihood_variance()`,
      :func:`~GPSat.models.gpflow_models.GPflowGPRModel.set_likelihood_variance()` and 
      :func:`~GPSat.models.gpflow_models.GPflowGPRModel.set_likelihood_variance_constraints()` methods.
    - We place inducing features in each input dimension and the effective number M of inducing features is the 
      *product* of the per-dimension number of inducing features.
    - Has O(NM^2) pre-computation cost, O(M^3) per-iteration complexity and O(NM) memory scaling.
    - Crucially, VFF is restricted to work in a finite domain. This introduces an extra variable ``domain_size``
      to be tuned, which can affect performance. As a rule of thumb, the ``domain_size`` should be large enough to
      subsume the training and inference regions, but making it too large can lead to predictions that are overly smooth.

    References
    ----------
    \[H'17\] Hensman, James, Nicolas Durrande, and Arno Solin. "Variational Fourier Features for Gaussian Processes." J. Mach. Learn. Res. (2017).
    
    """
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
                 kernel="Matern32",
                 num_inducing_features: Optional[Union[int, list]]=None,
                 kernel_kwargs=None,
                 mean_function=None,
                 mean_func_kwargs=None,
                 domain_size: Optional[Union[float, List[float]]]=None,
                 expert_loc=None,
                 **kwargs):
        # TODO: Checked mean function properly
        # TODO: Currently likelihood variance cannot be set at initialisation. Add a `noise_variance` argument.

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
        kernel: str
            See :func:`GPflowGPRModel.__init__() <GPSat.models.gpflow_models.GPflowGPRModel.__init__>`
            
            We have only implemented the case where the same kernel is used per dimension.
            This is to be extended in the future.
        kernel_kwargs: dict | list of dict, optional
            If given as a single ``dict``, it passes the same keyword arguments to the kernel in each dimension.
            If given as a ``list``, the ``i``'th entry corresponds to the keyword arguments passed to the kernel in dimension ``i``.
        num_inducing_features: int | list of int
            The number of Fourier features in each dimension. If given as a ``list``, the length must be equal to
            the input dimensions i.e. the length of ``self.coords_col`` (see :class:`~GPSat.models.base_model.BaseGPRModel`)
            and the entries correspond to the number of inducing features in each dimension.
            If given as ``int``, the same number of inducing features are set per input dimension.
        domain_size: float | list of float, optional
            The (unscaled) size of the fininte domain where VFF is defined.
            If given as a ``list``, this defines a cuboidal domain centered at ``expert_loc`` with size
            ``2 * domain_size[i]`` in each dimension ``i``. If given as a ``float``, this defines a cubic domain
            with size ``2 * domain_size`` in each dimension.
        expert_loc: np.array, optional
            The center of the cuboidal domain where Fourier basis is defined.

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

        assert kernel is not None, "kernel was not provided"
        assert num_inducing_features is not None, "Number of inducing points per dimension not specified"

        # if kernel is str: get function
        if isinstance(kernel, str):
            # get the kernel function (still requires
            kernel = getattr(gpflow.kernels, kernel)
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
        """Get the ELBO value for current state."""
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
        Setter method for kernel variance.
        
        Parameters
        ----------
        kernel_variance: float
            We assign equal variance to each 1D kernel such that they multiply to ``kernel_variance``.

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



        