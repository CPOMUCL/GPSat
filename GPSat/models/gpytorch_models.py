import inspect
import numpy as np
import pandas as pd
import torch
import gpytorch
from gpytorch.kernels import ScaleKernel, GridInterpolationKernel
from GPSat.decorators import timer
from GPSat.models import BaseGPRModel


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


# ---------- GPyTorch modules ----------

class ExactGPR(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, kernel, likelihood, mean=None):
        super(ExactGPR, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean() if mean is None else mean
        self.mean_module.to(device)
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ---------- GPyTorch local expert models ----------

class GPyTorchGPRModel(BaseGPRModel):
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
                 kernel="MaternKernel",
                 kernel_kwargs=None,
                 mean_function: gpytorch.means.Mean=None,
                 mean_func_kwargs: dict=None,
                 noise_variance: float=None, # Variance of Gaussian likelihood. Unnecessary if likelihood is specified
                 likelihood: gpytorch.likelihoods.GaussianLikelihood=None,
                 **kwargs):
        
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

        self.coords = torch.tensor(self.coords, requires_grad=False, dtype=torch.float32)
        self.obs = torch.tensor(self.obs, requires_grad=False, dtype=torch.float32).squeeze()
        self.obs_mean = torch.tensor(self.obs_mean, requires_grad=False, dtype=torch.float32).squeeze()
        self.obs_scale = torch.tensor(self.obs_scale, requires_grad=False, dtype=torch.float32).squeeze()

        # --
        # set kernel
        # --
        assert kernel is not None, "kernel was not provided"

        # if kernel is str: get function
        if isinstance(kernel, str):
            # if additional kernel kwargs not provide use empty dict
            if kernel_kwargs is None:
                kernel_kwargs = {}
            
            # get the kernel function (still requires
            kernel = getattr(gpytorch.kernels, kernel)

            # check signature parameters
            kernel_signature = inspect.signature(kernel).parameters

            # see if it takes lengthscales
            # - want to initialise with appropriate length (one length scale per coord)
            # TODO: adapt for scikit
            if ("lengthscales" in kernel_signature) & ("lengthscales" not in kernel_kwargs):
                kernel_kwargs['lengthscales'] = np.ones(self.coords.shape[1])
                print(f"setting lengthscales to: {kernel_kwargs['lengthscales']}")

            # --
            # initialise kernel
            # --
            kernel = self._initialise_kernel(kernel)

        # --
        # prior mean function
        # --
        if isinstance(mean_function, str):
            if mean_func_kwargs is None:
                mean_func_kwargs = {}
            mean_function = getattr(gpytorch.means, mean_function)(**mean_func_kwargs)
            mean_function.to(device)

        # ---
        # initialise model
        # ---
        if likelihood is None:
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
            self.likelihood.noise = 1. if noise_variance is None else noise_variance
        else:
            self.likelihood = likelihood.to(device)

        self.model = ExactGPR(train_x=self.coords,
                              train_y=self.obs,
                              kernel=kernel,
                              likelihood=self.likelihood,
                              mean=mean_function).to(device)

        self.set_parameters(**kernel_kwargs)

    def _initialise_kernel(self, kernel, *args, **kwargs):
        # This method will be overridden for KISS-GP implementation
        kernel = ScaleKernel(kernel(ard_num_dims=self.coords.shape[1]))
        return kernel.to(device)
    
    @timer
    def predict(self, coords, full_cov=False, apply_scale=True):
        """method to generate prediction at given coords"""
        # Get into evaluation (predictive posterior) mode
        self.model.eval()
        self.likelihood.eval()

        # TODO: allow for only y, or f to be returned
        # convert coords as needed
        if isinstance(coords, (pd.Series, pd.DataFrame)):
            if self.coords_col is not None:
                coords = coords[self.coords_col].values
            else:
                coords = coords.values

        if isinstance(coords, (list, np.ndarray)):
            coords = torch.tensor(coords).to(device)

        if len(coords.shape) == 1:
            coords = coords[None, :] # Is this correct?

        assert isinstance(coords, torch.Tensor), f"coords should be a torch tensor"
        coords = coords.type(self.coords.dtype)

        if apply_scale:
            coords = coords / self.coords_scale

        # Compute f* and y*
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f_pred = self.model(coords)
            y_pred = self.likelihood(f_pred)

        if not full_cov:
            out = {
                "f*": f_pred.mean.cpu().detach().numpy(),
                "f*_var": f_pred.variance.cpu().detach().numpy(),
                "y_var": y_pred.variance.cpu().detach().numpy(),
                "f_bar": self.obs_mean.item()
            }
        else:
            out = {
                "f*": f_pred.mean.cpu().detach().numpy(),
                "f*_var": f_pred.variance.cpu().detach().numpy(),
                "y_var": y_pred.variance.cpu().detach().numpy(),
                "f*_cov": f_pred.covariance_matrix.cpu().detach().numpy(),
                "y_cov": y_pred.covariance_matrix.cpu().detach().numpy(),
                "f_bar": self.obs_mean.item()
            }

        return out

    @timer
    def optimise_parameters(self, optimiser='adam', iterations=30):
        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        if optimiser == 'adam':
            # Use the adam optimizer
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

            for i in range(iterations):
                optimizer.zero_grad()
                output = self.model(self.coords.to(device))
                loss = -mll(output, torch.squeeze(self.obs.to(device)))
                loss.backward()
                optimizer.step()

        elif optimiser == 'lbfgs':
            # Use LBFGS optimizer
            optimizer = torch.optim.LBFGS(self.model.parameters())

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

            for i in range(iterations):
                def closure():
                    optimizer.zero_grad()
                    output = self.model(self.coords.to(device))
                    loss = -mll(output, torch.squeeze(self.obs.to(device)))
                    loss.backward()
                    return loss
                optimizer.step(closure)

    def get_objective_function_value(self):
        self.model.eval()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        with torch.no_grad():
            output = self.model(self.coords.to(device))
            loss = mll(output, self.obs.to(device))
        return loss.item()

    # -----
    # Getters/setters for model hyperparameters
    # -----
    @property
    def param_names(self) -> list:
        return ["smoothness", "lengthscales", "kernel_variance", "likelihood_variance"]

    def get_smoothness(self):
        """
        Smoothness of Matern kernel (e.g. 0.5, 1.5, 2.5) specified explicitly in GPyTorch
        """
        return self.model.covar_module.base_kernel.nu

    def get_lengthscales(self):
        return self.model.covar_module.base_kernel.lengthscale

    def get_kernel_variance(self):
        return self.model.covar_module.outputscale

    def get_likelihood_variance(self):
        return self.likelihood.noise
    
    def set_smoothness(self, smoothness):
        self.model.covar_module.base_kernel.nu = smoothness

    def set_lengthscales(self, lengthscales):
        self.model.covar_module.base_kernel.lengthscale = torch.atleast_2d(torch.tensor(lengthscales)).to(device)

    def set_kernel_variance(self, kernel_variance):
        self.model.covar_module.outputscale = kernel_variance

    def set_likelihood_variance(self, likelihood_variance):
        self.likelihood.noise = likelihood_variance

    # -----
    # Applying constraints on the model hyperparameters
    # -----
    def _preprocess_constraint(self, param_name, low, high, move_within_tol=True, tol=1e-8, scale=False):
        assert param_name in self.param_names, f"param_name must be one of {self.param_names}"

        param = self.get_parameters()[param_name]

        if isinstance(low, (list, tuple)):
            low = torch.tensor(low).to(device)
        elif isinstance(low, (int, float)):
            low = torch.tensor([low]).to(device)

        if isinstance(high, (list, tuple)):
            high = torch.tensor(high).to(device)
        elif isinstance(high, (int, float)):
            high = torch.tensor([high]).to(device)

        assert len(param[0]) == len(low), "len of low constraint does not match paramlength"
        assert len(param[0]) == len(high), "len of high constraint does not match param length"
        assert torch.all(low <= high), "all values in high constraint must be greater than low"

        # scale the bound by the coordinate scale value
        if scale:
            # self.coords_scale expected to be 2-d
            low = low / self.coords_scale[0, :]
            high = high / self.coords_scale[0, :]

        # if the current values are outside of tolerances then move them in
        if move_within_tol:
            # require current length scales are more than tol for upper bound
            for i in range(self.coords.shape[1]):
                if param[0,i] > (high[i] - tol):
                    param[0,i] = high[i] - tol
                # similarly for the lower bound
                if param[0,i] < (low[i] + tol):
                    param[0,i] = low[i] + tol

        return low, high

    @timer
    def set_lengthscales_constraints(self, low, high, move_within_tol=True, tol=1e-8, scale=False):
        (low, high) = self._preprocess_constraint('lengthscales', low, high, move_within_tol, tol, scale)
        self.model.covar_module.base_kernel.register_constraint("raw_lengthscale",
                                                                gpytorch.constraints.Interval(low, high)
                                                                )

    @timer
    def set_kernel_variance_constraints(self, low, high, move_within_tol=True, tol=1e-8, scale=False):
        (low, high) = self._preprocess_constraint('kernel_variance', low, high, move_within_tol, tol, scale)
        self.model.covar_module.register_constraint("raw_outputscale",
                                                    gpytorch.constraints.Interval(low, high)
                                                    )

    @timer
    def set_likelihood_variance_constraints(self, low, high, move_within_tol=True, tol=1e-8, scale=False):
        (low, high) = self._preprocess_constraint('likelihood_variance', low, high, move_within_tol, tol, scale)
        self.model.likelihood.register_constraint("raw_noise",
                                                  gpytorch.constraints.Interval(low, high)
                                                  )


class GPyTorchKISSGPModel(GPyTorchGPRModel):
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
                 kernel="MaternKernel",
                 kernel_kwargs=None,
                 mean_function: gpytorch.means.Mean=None,
                 mean_func_kwargs: dict=None,
                 noise_variance: float=None, # Variance of Gaussian likelihood. Unnecessary if likelihood is specified
                 likelihood: gpytorch.likelihoods.GaussianLikelihood=None,
                 **kwargs):

        super().__init__(data=data,
                         coords_col=coords_col,
                         obs_col=obs_col,
                         coords=coords,
                         obs=obs,
                         coords_scale=coords_scale,
                         obs_scale=obs_scale,
                         obs_mean=obs_mean,
                         kernel=kernel,
                         kernel_kwargs=kernel_kwargs,
                         mean_function=mean_function,
                         mean_func_kwargs=mean_func_kwargs,
                         noise_variance=noise_variance,
                         likelihood=likelihood)

    def _initialise_kernel(self, kernel, *args, **kwargs):
        grid_size = gpytorch.utils.grid.choose_grid_size(self.coords)
        num_dims = self.coords.shape[1]
        kernel = ScaleKernel(GridInterpolationKernel(
                            kernel(ard_num_dims=num_dims),
                            grid_size=grid_size, num_dims=num_dims
                            )
                        )
        return kernel.to(device)

    def get_smoothness(self):
        """
        Smoothness of Matern kernel (e.g. 0.5, 1.5, 2.5) specified explicitly in GPyTorch
        """
        return self.model.covar_module.base_kernel.base_kernel.nu

    def get_lengthscales(self):
        return self.model.covar_module.base_kernel.base_kernel.lengthscale
    
    def set_smoothness(self, smoothness):
        self.model.covar_module.base_kernel.base_kernel.nu = smoothness

    def set_lengthscales(self, lengthscales):
        self.model.covar_module.base_kernel.base_kernel.lengthscale = torch.atleast_2d(torch.tensor(lengthscales)).to(device)

    @timer
    def set_lengthscale_constraints(self, low, high, move_within_tol=True, tol=1e-8, scale=False):
        (low, high) = self._preprocess_constraint('lengthscales', low, high, move_within_tol, tol, scale)
        self.model.covar_module.base_kernel.base_kernel.register_constraint("raw_lengthscale",
                                                                gpytorch.constraints.Interval(low, high)
                                                                )


if __name__ == "__main__":
    # Testing local experts
    import numpy as np
    import pandas as pd
    from sklearn.gaussian_process.kernels import Matern
    from sklearn.gaussian_process import GaussianProcessRegressor

    # # Generate random data from matern-3/2 model
    # np.random.seed(23435)

    # kernel = Matern(length_scale=0.8, nu=3/2)
    # gp = GaussianProcessRegressor(kernel)

    # x = np.linspace(0, 10, 100)[:,None]
    # f = gp.sample_y(x, random_state=0)

    # N = 50
    # eps = 1e-2
    # indices = np.arange(100)
    # np.random.shuffle(indices)
    # x_train = x[indices[:N]]
    # y_train = f[indices[:N]] + eps*np.random.randn(N,1)

    # df = pd.DataFrame(data={'x': x_train[:,0], 'y': y_train[:,0]})

    # # Fit matern-3/2 gp on training data
    # gp.alpha = eps**2
    # gp.fit(x_train, y_train)
    # ls = gp.kernel_.length_scale
    # ml = gp.log_marginal_likelihood()

    # # Get prediction at random point
    # test_index = np.random.randint(0,99)
    # x_test = x[[test_index]]
    # pred_mean, pred_std = gp.predict(x_test, return_std=True)

    # model = GPyTorchGPRModel(data=df,
    #                         obs_col='y',
    #                         coords_col='x',
    #                         obs_mean=None,
    #                         kernel='MaternKernel',
    #                         kernel_kwargs={'smoothness': 1.5,
    #                                        'lengthscales': 1.})

    # model.set_parameters(likelihood_variance=eps**2)

    # model.set_lengthscale_constraints(low=1e-10, high=1e5)

    # result = model.optimise_parameters()
    # out = model.predict(coords=x_test)

    # print(out)

    # Set up 2D grid
    import gpflow 
    xs = np.arange(0, 5, 0.1)
    ys = np.arange(0, 5, 0.1)

    Xs, Ys = np.meshgrid(xs, ys)

    xdim, ydim = Xs.shape

    X = np.stack([Xs.ravel(), Ys.ravel()], axis=1)

    # Generate synthetic data as before
    kern = gpflow.kernels.Matern52()

    Kxx = kern(X, X)

    # Sample independent latent GPs
    np.random.seed(1)
    noise = np.random.randn(xdim, ydim, 2)

    Chol = np.linalg.cholesky(Kxx)
    f = Chol @ noise[...,0].ravel()

    f = f.reshape(xdim, ydim)

    # Generate training data
    num_obs = 1000
    obs_noise = 1e-2

    # Get observations at num_obs random locations
    rng = np.random.default_rng(0)
    x_idxs = np.arange(xdim)
    y_idxs = np.arange(ydim)
    X_idxs, Y_idxs = np.meshgrid(x_idxs[1:-1], y_idxs[1:-1], indexing='ij')
    all_idxs = np.stack([X_idxs.flatten(), Y_idxs.flatten()], axis=1)
    idxs = rng.choice(all_idxs, num_obs, replace=False)

    x_train = [X.reshape(xdim, ydim, 2)[tuple(idx)] for idx in idxs]
    y_train = [f[tuple(idx)].squeeze() + np.sqrt(obs_noise)*np.random.randn(1) for idx in idxs]
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    df = pd.DataFrame(data={'x': x_train[:,0], 'y': x_train[:,1], 'obs': y_train[:,0]})

    # model = GPyTorchGPRModel(data=df,
    #                         obs_col='obs',
    #                         coords_col=['x', 'y'],
    #                         obs_mean=None,
    #                         kernel='MaternKernel',
    #                         kernel_kwargs={'smoothness': 1.5,
    #                                        'lengthscales': [1., 1.]})

    model = GPyTorchKISSGPModel(data=df,
                                obs_col='obs',
                                coords_col=['x', 'y'],
                                obs_mean=None,
                                kernel='MaternKernel',
                                kernel_kwargs={'smoothness': 1.5,
                                                'lengthscales': [1., 1.]})

    model.set_parameters(likelihood_variance=1e-2**2)

    model.set_lengthscale_constraints(low=[1e-10, 1e-10], high=[10, 10])

    print(model.get_objective_function_value())

    result = model.optimise_parameters()

    print(model.get_parameters())


