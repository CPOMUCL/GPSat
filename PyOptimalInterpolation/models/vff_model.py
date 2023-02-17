import inspect
import gpflow
import numpy as np

from PyOptimalInterpolation.decorators import timer
from PyOptimalInterpolation.models import BaseGPRModel
from PyOptimalInterpolation.models.gpflow_models import GPflowGPRModel


"""
Install VFF with
`git clone https://github.com/HJakeCunningham/VFF.git`
"""
import VFF

class GPflowVFFModel(GPflowGPRModel):
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
                 num_inducing_points=None,
                 kernel_kwargs=None,
                 mean_function=None,
                 mean_func_kwargs=None,
                 margin=None):
        # TODO: handle kernel (hyper) parameters
        # TODO: Currently does not handle variable ms + does not incorporate mean function

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
        assert num_inducing_points is not None, "Number of inducing points per dimension not specified"

        # if kernel is str: get function
        if isinstance(kernels, str):
            kernel = kernels

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
            kernels = [kernel(**kernel_kwargs) for _ in range(self.coords.shape[1])]

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
        if margin is None:
            margin = [1e-8 for _ in range(self.coords.shape[1])]
        elif isinstance(margin, (int, float)):
            margin = [margin for _ in range(self.coords.shape[1])]

        assert len(margin) == self.coords.shape[1], "length of margin list must match number of coordinate dimensions"

        a = []; b = []
        for i, coords in enumerate(self.coords.T):
            a.append(coords.min()-margin[i])
            b.append(coords.max()+margin[i])

        # if isinstance(num_inducing_points, int):
        #     ms = np.arange(num_inducing_points)
        # elif isinstance(num_inducing_points, list):
        #     ms = [np.arange(num) for num in num_inducing_points]

        ms = np.arange(num_inducing_points)

        self.model = VFF.gpr.GPR_kron(data=(self.coords, self.obs),
                                      ms=ms,
                                      a=a,
                                      b=b,
                                      kernel_list=kernels)

    def get_objective_function_value(self):
        """get the marginal log likelihood"""
        return self.model.elbo().numpy()


        