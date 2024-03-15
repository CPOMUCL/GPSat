import inspect
import re
import os
import platform
import subprocess
import pandas as pd
import numpy as np
from tensorflow.python.client import device_lib
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Literal, Optional
from GPSat.decorators import timer
from GPSat.utils import cprint


# ------- Base class ---------

class BaseGPRModel(ABC):
    """
    Base class for all ``GPSat`` models. Every local expert model must inherit from this class.
    This is to enforce consistent naming across different models for basic attributes and methods
    such as training and predicting.

    Attributes
    ----------
    obs: np.ndarray | None
        A numpy array consisting of all observed values from satellite measurements.
        If de-meaning and rescaling (see below) is applicable, it stores the transformed values (*not* the original values).
        This has shape (N, P), where N is the number of data points and P is the dimension of observations.
    obs_col: list of str | list of int
        The variable name(s) of the observations. Relevant if the observations are extracted from
        a dataframe, in which case ``obs_col`` is the column name correspoding to the observations.
        If not specified, it will default to ``[0]``.
    coords: numpy array | None
        A numpy array consisting of all coordinate values where satellite measurements were made.
        If rescaling (see below) is applicable, it stores the rescaled values (*not* the original values).
        This has shape (N, D), where N is the number of data points and D is the dimension of input coordinates.
    coords_col: list of str | list of int
        The variable name(s) of the coordinates. Relevant if the coordinate readings are extracted from
        a dataframe, in which case ``coords_col`` should contain the column names correspoding to the coordinates.
        If not specified, it will default to a list of indices, e.g. ``[0, 1, 2]`` for three-dimensional inputs.
    obs_mean: numpy array
        The mean value of observations. This value gets subtracted from the observation data for de-meaning.
    obs_scale: numpy array
        The value(s) with which we rescale the observation data. Default is 1.
    coords_scale: numpy array
        The value(s) with which we rescale the coordinates data. The shape must match the shape of the ``coords`` array. Default is 1.
    gpu_name:
        Name of GPU if availabe, used for training/prediction.
    cpu_name:
        Processor name of the machine where experiments were run on.

    Methods
    -------
    predict(coords)
        Makes predictions on new coordinates specified by the array ``coords``.
        *This is an abstract method that must be overridden by all inheriting classes.*
    optimise_parameters()
        Fits model on training data by optimising the parameters/hyperparameters.
        *This is an abstract method that must be overridden by all inheriting classes.*
    get_objective_function_value()
        Returns the value of the objective function used to train the model.
        *This is an abstract method that must be overridden by all inheriting classes.*
    param_names()
        A property method to retrieve the names of the parameters/hyperparameters of the model.
        *This is an abstract method that must be overridden by all inheriting classes.*
    get_parameters(*args, return_dict=True)
        Retrieves the values of parameters.
    set_parameters(**kwargs)
        Sets values of parameters.
    set_parameter_constraints(constraints_dict, **kwargs)
        Sets constraints on parameters.

    Notes
    -----
    - To keep notations consistent, we will denote the number of datapoints by N, the input dimension by D and output dimension by P.
    - All inheriting classes must override the methods ``predict``, ``optimise_parameters``, ``get_objective_function_value`` and ``param_names``
      (see below).
    - In addition, all inheriting classes must contain the getter/setter methods ``get_*`` and ``set_*`` for all ``*`` in ``param_names``.
      e.g. if ``param_names = ['A', 'B']`` then the methods ``get_A``, ``set_A``, ``get_B``, ``set_B`` should be defined.
      Additionally, the method ``set_*_constraints`` can also be defined, which will be used to constrain the values of the parameters during optimisation.

    """
    def __init__(self,
                 data: Optional[pd.DataFrame] = None,
                 coords_col: Union[str, List[str], None] = None,
                 obs_col: Union[str, List[str], None] = None,
                 coords: Optional[np.ndarray] = None,
                 obs: Optional[np.ndarray] = None,
                 coords_scale: Union[int, float, List[Union[int, float]], None] = None,
                 obs_scale: Union[int, float, List[Union[int, float]], None] = None,
                 obs_mean: Union[Literal['local'], int, float, List[Union[int, float]], None] = None,
                 # kernel=None,
                 # prior_mean=None,
                 verbose: bool = True,
                 **kwargs):
        """
        Parameters
        ----------
        data: pandas dataframe, optional.
            A pandas dataframe containing the training data. If not specified, ``coords`` and ``obs`` must be
            specified explicitly.
        coords_col: str | list of str | None, default None.
            The column names in ``data`` corresponding to the input coordinates where measurements were made.
            e.g. ``coords_col = ['x', 'y', 't']``.
        obs_col: str | list of str | None, default None.
            The column names in ``data`` corresponding to the measurement values.
        coords: numpy array, optional.
            A numpy array of shape (N, D) specifying the input coordinates explicitly.
            If D = 1, an array of shape (N,) is also allowed.
            Only used if ``data`` is None.
        obs: numpy array, optional.
            A numpy array of shape [N, P] specifying the measurement values explicitly.
            If P = 1, an array of shape (N,) is also allowed.
            Only used if ``data`` is None.
        coords_scale: int | float | list of int or float | None, default None.
            The value(s) by which we rescale the input coordinate values. If the coordinate is D-dimensional,
            we can specify a list of length D whose entries correspond to the scaling for each dimension.
        obs_scale: int | float | list of int or float | None, default None.
            The value(s) by which we rescale the output measurement values. If the measurements are P-dimensional,
            we can specify a list of length P whose entries correspond to the scaling for each output dimension.
        obs_mean: 'local' | int | float | list of int or float | None, default None.
            Value to subtract from observations. The purpose is to calibrate observations in order to use kernels
            with mean zero if one wishes. Setting ``obs_mean = 'local'`` allows us to use the mean value of the array ``self.obs``.
        verbose: bool, default True
            Set verbosity of model initialisation.

        """

        # --
        # data
        # --

        # assign data to model
        if data is not None:
            assert coords_col is not None, "data was provided, but coord_col was not"
            assert obs_col is not None, "data was provided, but obs_col was not"

            # require the columns for selecting data are not str
            if isinstance(coords_col, str):
                coords_col = [coords_col]
            if isinstance(obs_col, str):
                obs_col = [obs_col]

            # TODO: should data be copied?

            # select relevant data - as np.arrays
            # - taking values makes copy(?)
            self.obs = data.loc[:, obs_col].values
            self.coords = data.loc[:, coords_col].values

            # store the column names
            self.obs_col = obs_col
            self.coords_col = coords_col
        # otherwise expect to have coords and obs provided directly
        else:

            assert obs is not None, f"data is {data}, and so is obs: {obs}, provide either"
            assert coords is not None, f"data is {data}, and so is coords: {coords}, provide either"

            assert isinstance(obs, np.ndarray), "if obs is provided directly it must be an np.array"
            assert isinstance(coords, np.ndarray), "if obs is provided directly it must be an np.array"

            if len(obs.shape) == 1:
                if verbose:
                    print("obs is 1-d array, setting to 2-d")
                obs = obs[:, None]

            if len(coords.shape) == 1:
                if verbose:
                    print("coords is 1-d array, setting to 2-d")
                coords = coords[:, None]

            assert len(obs) == len(coords), "obs and coords lengths don't match "

            self.obs = obs
            self.coords = coords

            # if column 'names' not provide generate default values
            # - these could be np.arrays...
            if coords_col is None:
                coords_col = [_ for _ in range(self.coords.shape[1])]
            if obs_col is None:
                obs_col = [0]
            self.coords_col = coords_col
            self.obs_col = obs_col

        # nan check
        assert not np.isnan(self.coords).any(), "nans found in coords"
        assert not np.isnan(self.obs).any(), "nans found in obs"

        # observation mean - to be subtracted from observations

        # TODO: review where should de-meaning be done
        # remove mean of observations data?
        if obs_mean == "local":
            if verbose:
                print(f"setting obs_mean with mean of obs_col: {obs_col}")
            obs_mean = np.mean(self.obs, axis=0)
        else:
            obs_mean = np.array([0])[None, :]

        if isinstance(obs_mean, list):
            obs_mean = np.array(obs_mean)[None, :]
        elif isinstance(obs_mean, (int, float)):
            obs_mean = np.array([obs_mean])[None, :]

        if verbose > 1:
            print(f"obs_mean set to: {obs_mean}")
        self.obs_mean = obs_mean

        # scale coordinates and / or observations?
        if obs_scale is None:
            obs_scale = np.atleast_2d(1)
        elif isinstance(obs_scale, list):
            obs_scale = np.array(obs_scale)[None, :]
        elif isinstance(obs_scale, (int, float)):
            obs_scale = np.array([obs_scale])[None, :]

        if verbose > 1:
            print(f"obs_scale set to: {obs_scale}")
        self.obs_scale = obs_scale

        if coords_scale is None:
            coords_scale = np.atleast_2d(1)
        elif isinstance(coords_scale, list):
            coords_scale = np.array(coords_scale)[None, :]
        elif isinstance(coords_scale, (int, float)):
            coords_scale = np.array([coords_scale])[None, :]

        if verbose > 1:
            print(f"coords_scale set to: {coords_scale}")
        self.coords_scale = coords_scale

        # scale coords / obs
        # - will this affect values in place if taken from a data? (dataframe)
        # Convert int to float to perform division
        if self.coords.dtype == 'int':
            self.coords = self.coords.astype(float)
            
        if self.obs.dtype == 'int':
            self.obs = self.obs.astype(float)

        self.coords /= self.coords_scale
        self.obs -= self.obs_mean
        self.obs /= self.obs_scale

        # ---
        # prior mean and kernel functions
        # ---

        # assigning kernel, and prior mean function should be specific to the underlyin engine

        # kernel - either string or function?

        # ---
        # device information
        # ---

        self.gpu_name, self.cpu_name = self._get_device_names()

        if verbose > 1:
            cprint("detected the following devices:", "OKBLUE")
            cprint(f"cpu_name: {self.cpu_name}", "BOLD")
            cprint(f"gpu_name: {self.gpu_name}", "BOLD")

        # ----
        # check param_names each have a get/set method
        # ----

        pnames = self.param_names
        if verbose > 1:
            print(f"checking param_names: {pnames} each have a get_*, set_( method")

        for pn in pnames:
            assert not bool(re.search(" ", pn)), f"param_name: '{pn}' has a space (' ') in it, which is prohibited"
            _ = getattr(self, f"set_{pn}")
            _ = getattr(self, f"get_{pn}")

    def _get_device_names(self):

        gpu_name = None
        cpu_name = self._get_processor_name()

        try:
            dev = device_lib.list_local_devices()
            for d in dev:
                # check if device is GPU
                # - NOTE: will break after first GPU
                if (d.device_type == "GPU") & (gpu_name is None):
                    cprint("found GPU", "BOLD")
                    try:
                        name_loc = re.search("name:(.*?),", d.physical_device_desc).span(0)
                        gpu_name = d.physical_device_desc[(name_loc[0] + 6):(name_loc[1] - 1)]
                    except Exception as e:
                        print("there was some issue getting GPU name")
                        print(e)
                    break
        except Exception as e:
            print(e)
        return gpu_name, cpu_name

    @staticmethod
    def _get_processor_name():
        # ref: https://stackoverflow.com/questions/4842448/getting-processor-information-in-python
        if platform.system() == "Windows":
            return platform.processor()
        elif platform.system() == "Darwin":
            os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
            command = "sysctl -n machdep.cpu.brand_string"
            try:
                out = subprocess.check_output(command).strip()
            except FileNotFoundError as e:
                out = "Unable to get cpu info for system: Darwin"
            return out
        elif platform.system() == "Linux":
            command = "cat /proc/cpuinfo"
            all_info = subprocess.check_output(command, shell=True).decode().strip()
            for line in all_info.split("\n"):
                if "model name" in line:
                    return re.sub(".*model name.*:", "", line, 1).lstrip()
        else:
            print(f"getting processor name for system: {platform.system()} not implemented")
            return f"Unable to get cpu info for system: {platform.system()}"


    @abstractmethod
    def predict(self, coords: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Method to generate prediction at given coords.
        *Any inheriting class should override this method.*

        Parameters
        ----------
        coords: numpy array
            Coordinate values where we wish to make predictions.

        Returns
        -------
        dict
            Predictions at the given coordinate locations. Should be a dictionary containing the
            mean and variance of the predictions, as well as other variables one wishes to save.
        
        """
        pass

    @abstractmethod
    def optimise_parameters(self):
        """
        Method to fit data on model by optimising (hyper/variational)-parameters.
        *Any inheriting class should override this method.*
        """
        pass

    @property
    @abstractmethod
    def param_names(self) -> List[str]:
        """
        Property method that returns the names of parameters in a list.
        *Any inheriting class should override this method.*

        Each parameter name should have a ``get_*`` and ``set_*`` method.
        e.g. if ``param_names = ['A', 'B']`` then methods ``get_A``, ``set_A``, ``get_B``, ``set_B``
        should be defined.

        Additionally, one can specify a ``set_*_constraints`` method that imposes constraints
        on the parameters during training, if applicable.
        """
        pass

    @timer
    def get_parameters(self, *args, return_dict=True) -> Union[dict, list]:
        """
        Get parameter values. Loops through the ``get_*`` methods for all ``*`` in ``param_names`` or ``args``.
        If ``return_dict`` is ``True``, it returns a dictionary of param name-value pairs and if ``False``,
        returns a list of all parameter values in the order listed in ``param_names``.

        Parameters
        ----------
        args: list of str
            A list of parameter names whose values we wish to retrieve. If it is an empty list, it will
            return the values of *all* parameters in ``self.param_names``.
        return_dict: bool, default True
            Option to return the result as a dictionary or as a list.

        Returns
        -------
        dict or list
            A dictionary (if ``return_dict=True``) or list (if ``return_dict=False``) containing the
            parameter values.

        """

        # if not args provided default to get all
        if len(args) == 0:
            args = self.param_names
        # check args are validate param_names
        for a in args:
            assert a in self.param_names, f"cannot get parameters for: {a}, it's not in param_names: {self.param_names}"
        # either return values in dict or list
        if return_dict:
            return {a: getattr(self, f"get_{a}")() for a in args}
        else:
            return [getattr(self, f"get_{a}")() for a in args]

    @timer
    def set_parameters(self, **kwargs):
        """
        Set parameter values.

        Parameters
        ----------
        kwargs: dict
            A dictionary of parameter name--value pairs that we wish to set.
            Parameter names must be a subset of ``self.param_names`` otherwise returns an error.

        """
        # TODO: allow for a nan check?
        for k, v in kwargs.items():
            assert k in self.param_names, f"cannot get parameters for: {k}, it's not in param_names: {self.param_names}"
            # TODO: allow for additional arguments to be supplied?
            #  - or should set_paramname() only take in one argument i.e. the parameter values
            getattr(self, f"set_{k}")(v)

    def set_parameter_constraints(self, constraints_dict, **kwargs):
        """
        Set constraints on parameters, e.g. maximum or minimum values.

        Parameters
        ----------
        constraints_dict: dict of dict
            A dictionary of parameter name--constraints pair, where the constraints are specified as
            dictionaries of arguments to be passed to the ``set_*_constraints`` method.
        kwargs: dict
            A global dictionary of keyword arguments to be passed to all ``set_*_constraints`` method.

        """
        for k, v in constraints_dict.items():
            assert k in self.param_names, f"cannot get parameters for: {k}, it's not in param_names: {self.param_names}"
            getattr(self, f"set_{k}_constraints")(**v, **kwargs)

    @abstractmethod
    def get_objective_function_value(self) -> np.ndarray:
        """
        Get value of objection function used to train the model.
        e.g. the log marginal likelihood when using exact GPR.
        *Any inheriting class should override this method.*
        """
        pass


