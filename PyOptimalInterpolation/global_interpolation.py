import scipy
import numpy as np
import xarray as xr
import time
from typing import List, Dict, Tuple
from PyOptimalInterpolation import LocalGPExpert
from PyOptimalInterpolation.post_process import PostProcessModule


class GlobalInterpolation():
    def __init__(self,
                 training_data: Tuple[np.ndarray, np.ndarray],
                 local_expert: LocalGPExpert,
                 dates: List,
                 region: xr.Dataset,
                 grid_res: int=50,
                 training_radius: float=300.*1000,
                 time_window: int=9,
                 post_process: PostProcessModule=None,
                 logdir='./log'):
        """
        Args
        --------
        
        """  
        X, z = training_data
        x, y, t = X.T
        self.x_train = x
        self.y_train = y
        self.t_train = t
        self.z_train = z # shape (N, out_dim)
        self.kdt_tree = scipy.spatial.cKDTree(np.array([x, y]).T)
        self.local_expert = local_expert
        self.dates = dates
        self.region = region
        self.x_coords = self.region['x'].values
        self.y_coords = self.region['y'].values
        self.resolution = grid_res
        self.R = training_radius
        self.T = time_window
        self.post_process = post_process
        self.logdir = logdir

    def _compile_model_at_gridpoint(self, gridpoint, hyperparams=None):
        """
        Compile the local expert model at a given gridpoint.
        Data lying within radius R is selected for training.
        TODO: also choose data within time window T
        """
        ID = self.kdt_tree.query_ball_point(x=gridpoint, r=self.R)
        inputs = np.array([self.x_train[ID], self.y_train[ID], self.t_train[ID]]).T
        outputs = self.z_train[ID]
        self.local_expert.compile((inputs, outputs), hyperparams)

    def train(self, postprocess_kwargs: Dict=None):
        """
        Main training loop. Loops through the gridpoints of the given region at the selected dates
        and learns the optimal parameters of the local expert model at each point.
        The hyperparameter field is also post-processed according to the selected method before they are saved.

        -----
        Args:
        -----
        TODO: Include options for skipping gridpoints and different prior means

        """
        trainable_hyperparameters = self.local_expert.parameters

        for date in self.dates:
            print(f"Training on date: {date}")
            xdim = len(self.region['x'])
            ydim = len(self.region['y'])

            # Create an xarray dataset to store hyperparameters
            data_vars = {key: (('y', 'x'), np.ones((ydim, xdim))) for key in trainable_hyperparameters}
            hyperparam_fields = xr.Dataset(data_vars=data_vars, coords={'x': self.x_coords, 'y': self.y_coords})

            count = 0
            cumulative_time = 0
            for i, x in enumerate(self.x_coords):
                for j, y in enumerate(self.y_coords):
                    time0 = time.time()

                    # Compile local expert model and optimise
                    gridpoint = np.array([x,y]).T
                    self._compile_model_at_gridpoint(gridpoint)
                    self.local_expert.optimise()
                    
                    # Cache optimal hyperparameters
                    hyperparams_new = self.local_expert.get_parameters()
                    for key in trainable_hyperparameters:
                        hyperparam_fields.data_vars[key][j,i] = hyperparams_new[key]

                    time1 = time.time()
                    cumulative_time += time1-time0
                    count += 1
                    print(f"Time elapsed for gridpoint {count}/{xdim*ydim}: {time1-time0:.3f}s")

            # Post-process hyperparameters
            print(f"Post-processing hyperparameters...")
            if postprocess_kwargs is not None:
                self.post_process(hyperparam_fields, postprocess_kwargs)

            # Log results
            print(f"Saving results...")
            fname = f"/hyperparameters_{date}_{self.resolution}km.nc" # TODO: add logdir to class attribute
            path = self.logdir + fname
            hyperparam_fields.to_netcdf(path)

            print(f"Complete.")
            print(f"Total time for training: {cumulative_time:.3f}s")

    def predict(self):
        """
        Make predictions at given region and dates
        """
        T_mid = self.T // 2 # choose central day in the time window to make predictions (TODO: This should be selected in the outer loop)
        gridded_mean = np.zeros((len(self.dates), len(self.y_coords), len(self.x_coords)))
        gridded_var = np.zeros((len(self.dates), len(self.y_coords), len(self.x_coords)))
        for t, date in enumerate(self.dates):
            # Load hyperparameters from netcdf file
            fname = f"/hyperparameters_{date}_{self.resolution}km.nc"
            path = self.logdir + fname
            hyperparam_dataset = xr.load_dataset(path)
            for i, x in enumerate(self.x_coords):
                for j, y in enumerate(self.y_coords):
                    # Get hyperparameters at gridpoint
                    keys = list(hyperparam_dataset.keys())
                    vars = [hyperparam_dataset[key].values[j,i] for key in keys]
                    hyperparams = dict(zip(keys, vars))

                    # Compile local model
                    gridpoint = np.array([x,y]).T
                    self._compile_model_at_gridpoint(gridpoint, hyperparams)
                    
                    # Predict at gridpoint
                    gridpoint = np.atleast_2d(np.array([x,y,T_mid])) #test input
                    mean, var = self.local_expert.predict(gridpoint)
                    gridded_mean[t,j,i] = mean
                    gridded_var[t,j,i] = var

        return gridded_mean, gridded_var


