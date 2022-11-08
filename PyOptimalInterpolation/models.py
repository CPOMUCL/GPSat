import scipy
import gpflow
import numpy as np
import xarray as xr
import time
import pickle
from abc import ABC, abstractmethod
from astropy.convolution import convolve, Gaussian2DKernel
from typing import List, Dict


# ------- Base class ---------

class SpatiotemporalOptimalInterpolation(ABC):
    """
    Base class containing the bare bone functionality for optimal interpolation using local GPs
    """
    def __init__(self, training_data, gp_model, kernel, init_hyperparameters, mean_function=None, grid_res=50, training_radius=300*1000, time_window=9):
        X, z = training_data
        x, y, t = X.T
        self.x_train = x
        self.y_train = y
        self.t_train = t
        self.z_train = z # Must be of shape (N, out_dim)
        self.kdt_tree = scipy.spatial.cKDTree(np.array([x, y]).T)
        self.gp_model = gp_model
        self.kernel = kernel
        self.mean_function = mean_function
        self.resolution = grid_res
        self.radius = training_radius
        self.T = time_window
        self._hyperparam_keys = None
        self._init_hyperparameters = init_hyperparameters

    def _clip_and_smooth(self, param, vmin=0, vmax=1e10, std=1):
        """
        Apply clipping and Gaussian smoothing to a given hyperparameter field
        """
        param = param.values # convert DataArray to numpy array
        # Clipping
        param[param >= vmax] = vmax
        param[param <= vmin] = vmin
        # Smoothing
        param = convolve(param, Gaussian2DKernel(x_stddev=std, y_stddev=std))
        return param

    def _post_process(self, param_fields: xr.Dataset, postprocess_kwargs):
        """
        Post process all trainable hyperparameters
        """
        assert set(postprocess_kwargs.keys()) == set(self._hyperparam_keys)
        for key in self._hyperparam_keys:
            param = self._clip_and_smooth(param_fields.data_vars[key], **postprocess_kwargs[key])
            param_fields[key] = (('y', 'x'), param)

    @abstractmethod
    def _get_hyperparameters(self, model):
        """
        Method to get the hyperparameters of the model

        Output:
        Hyperparameters of the model given by a dictionary whose keys are the hyperparameter labels
        and the values are the corresponding values
        """

    @abstractmethod
    def _get_kernel(self):
        """
        Method to instantiate kernel
        """

    @abstractmethod
    def _get_mean(self):
        """
        Method to get the mean function
        """

    @abstractmethod
    def _build_model(self, X, y, kernel, mean, hyperparameters):
        """
        Method to compile GP model
        """

    @abstractmethod
    def _optimise(self, model):
        """
        Method to optimise hyperparameters of the model
        """

    @abstractmethod
    def _get_prediction(self, model, gridpoint) -> List:
        """
        Method to get the predicted mean and variance of the model

        Output: List of form (mean, variance)
        """

    def train(self, dates, region: xr.Dataset, trainable_hyperparameters: List, postprocess_kwargs=None):
        """
        Main training loop.
        ------
        Args
        ------
        :dates: A list containing the dates of consideration
        :region: An xarray Dataset object containing coordinates x and y for the region of consideration
        :trainable_hyperparameters: A list containing the names of the trainable hyperparameters.
                                    The names must be compatible with the keys in self._init_hyperparameters
        :postprocess_kwargs: A nested dictionary of form
                            {hyperparameter name : dictionary of key word arguments to feed into self._clip_and_smooth}
        """
        self._hyperparam_keys = trainable_hyperparameters
        for date in dates:
            print(f"Training on date: {date}")
            xdim = len(region['x'])
            ydim = len(region['y'])
            x_coords = region['x'].values
            y_coords = region['y'].values
            data_vars = {key: (('y', 'x'), np.ones((ydim, xdim))) for key in trainable_hyperparameters}
            hyperparam_fields = xr.Dataset(data_vars=data_vars, coords={'x': x_coords, 'y': y_coords})
            count = 0
            cumulative_time = 0
            for i, x in enumerate(x_coords):
                for j, y in enumerate(y_coords):
                    time0 = time.time()

                    # 1. Select training data within a ball of radius r of the current grid point
                    ID = self.kdt_tree.query_ball_point(x=np.array([x,y]).T, r=self.radius)
                    inputs = np.array([self.x_train[ID], self.y_train[ID], self.t_train[ID]]).T
                    outputs = self.z_train[ID]
                    
                    # 2. Setup GP model
                    kernel = self._get_kernel()
                    mean = self._get_mean()
                    model = self._build_model(inputs, outputs, kernel, mean, self._init_hyperparameters)

                    # 3. Set trainable hyperparameters (TODO: Fix. Not general)
                    hyperparams = self._get_hyperparameters(model)
                    for key, val in hyperparams.items():
                        if key not in trainable_hyperparameters:
                            gpflow.set_trainable(val, False)

                    # 4. Optimise hyperparameters
                    self._optimise(model)
                    
                    # 5. Cache optimised hyperparameters
                    hyperparams_new = self._get_hyperparameters(model)
                    for key in trainable_hyperparameters:
                        hyperparam_fields.data_vars[key][j,i] = hyperparams_new[key]

                    time1 = time.time()
                    cumulative_time += time1-time0
                    count += 1
                    print(f"Time elapsed for gridpoint {count}/{xdim*ydim}: {time1-time0:.3f}s")

            # Post-process hyperparameters
            print(f"Post-processing hyperparameters...")
            if postprocess_kwargs is not None:
                self._post_process(hyperparam_fields, postprocess_kwargs)

            # Log results
            print(f"Saving results...")
            path = f"log/hyperparameters_{date}_{self.resolution}km.nc" # TODO: add logdir to class attribute
            hyperparam_fields.to_netcdf(path)

            print(f"Complete.")
            print(f"Total time for training: {cumulative_time:.3f}s")\

    def predict(self, dates, region: xr.Dataset):
        """
        Make predictions
        """
        T_mid = self.T // 2 # choose central day in the time window to make predictions
        x_coords = region['x'].values
        y_coords = region['y'].values
        gridded_mean = np.zeros((len(dates), len(y_coords), len(x_coords)))
        gridded_var = np.zeros((len(dates), len(y_coords), len(x_coords)))
        for t, date in enumerate(dates):
            path = f"log/hyperparameters_{date}_{self.resolution}km.nc"
            hyperparam_dataset = xr.load_dataset(path)
            for i, x in enumerate(x_coords):
                for j, y in enumerate(y_coords):
                    # TODO: Steps 1 and 2 below are almost identical to that in training step. Add as separate method.
                    # 1. Select training data within a ball of radius r of the current grid point
                    ID = self.kdt_tree.query_ball_point(x=np.array([x,y]).T, r=self.radius)
                    inputs = np.array([self.x_train[ID], self.y_train[ID], self.t_train[ID]]).T
                    outputs = self.z_train[ID]

                    # 2. Setup GP model
                    kernel = self._get_kernel()
                    mean = self._get_mean()
                    keys = list(hyperparam_dataset.keys())
                    vars = [hyperparam_dataset[key].values[j,i] for key in keys]
                    hyperparams = dict(zip(keys, vars))
                    model = self._build_model(inputs, outputs, kernel, mean, hyperparams)
                    
                    # 3. Predict at gridpoint
                    gridpoint = np.atleast_2d(np.array([x,y,T_mid])) #test input
                    mean, var = self._get_prediction(model, gridpoint)
                    gridded_mean[t,j,i] = mean
                    gridded_var[t,j,i] = var

        return gridded_mean, gridded_var

# ------- Pure python model ---------

...

# ------- GPflow models ---------

class GPR_OptimalInterpolation(SpatiotemporalOptimalInterpolation):
    def __init__(self, training_data, kernel, init_hyperparameters, max_iter=10_000, grid_res=50, training_radius=300*1000, time_window=9):
        self.opt = gpflow.optimizers.Scipy()
        self.max_iter = max_iter
        super().__init__(training_data,
                         gp_model=gpflow.models.GPR,
                         kernel=kernel,
                         init_hyperparameters=init_hyperparameters,
                         mean_function=gpflow.mean_functions.Constant,
                         grid_res=grid_res,
                         training_radius=training_radius,
                         time_window=time_window)

    def _get_mean(self):
        """ Get constant mean
        TODO: extend to more general means
        """
        mean = self.mean_function(c=np.array([np.mean(self.z_train)]))
        gpflow.set_trainable(mean.c, False)
        return mean

    def _get_kernel(self):
        hyperparameters = self._init_hyperparameters
        ls = (hyperparameters['lengthscale_x'], hyperparameters['lengthscale_y'], hyperparameters['lengthscale_t'])
        kernel = self.kernel(lengthscales=ls)
        return kernel

    def _get_hyperparameters(self, model: gpflow.models.GPR) -> Dict:
        hyperparameters = {}
        hyperparam_keys = ['lengthscale_x', 'lengthscale_y', 'lengthscale_t', 'kernel_variance', 'observation_variance']
        hyperparam_vals = [*model.kernel.lengthscales, model.kernel.variance, model.likelihood.variance]
        for key, val in zip(hyperparam_keys, hyperparam_vals):
            hyperparameters[key] = val
        return hyperparameters

    def _build_model(self, X, y, kernel, mean, hyperparameters):
        model = self.gp_model(data=(X, y), kernel=kernel, mean_function=mean)
        ls = (hyperparameters['lengthscale_x'], hyperparameters['lengthscale_y'], hyperparameters['lengthscale_t'])
        model.kernel.lengthscales.assign(ls)
        model.kernel.variance.assign(hyperparameters['kernel_variance'])
        model.likelihood.variance.assign(hyperparameters['observation_variance'])
        return model

    def _optimise(self, model: gpflow.models.GPR):
        self.opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=self.max_iter))

    def _get_prediction(self, model: gpflow.models.GPR, gridpoint: np.ndarray) -> List:
        mean, var = model.predict_f(gridpoint)
        return mean.numpy()[0,0], var.numpy()[0,0]


class SGPR_OptimalInterpolation(SpatiotemporalOptimalInterpolation):
    ...


class SVGP_OptimalInterpolation(SpatiotemporalOptimalInterpolation):
    ...

    