import inspect
import re
import os
import platform
import subprocess
import pandas as pd
import numpy as np
from tensorflow.python.client import device_lib
from abc import ABC, abstractmethod
from typing import List, Dict
from PyOptimalInterpolation.decorators import timer


# ------- Base class ---------

class BaseGPRModel(ABC):
    def __init__(self,
                 data=None,
                 coords_col=None,
                 obs_col=None,
                 coords=None,
                 obs=None,
                 coords_scale=None,
                 obs_scale=None,
                 obs_mean=None,
                 # kernel=None,
                 # prior_mean=None,
                 verbose=True,
                 **kwargs):
        """
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
                print("obs is 1-d array, setting to 2-d")
                obs = obs[:, None]

            if len(coords.shape) == 1:
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
                    print("found GPU")
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
            return subprocess.check_output(command).strip()
        elif platform.system() == "Linux":
            command = "cat /proc/cpuinfo"
            all_info = subprocess.check_output(command, shell=True).decode().strip()
            for line in all_info.split("\n"):
                if "model name" in line:
                    return re.sub(".*model name.*:", "", line, 1).lstrip()
        return None

    @abstractmethod
    def predict(self, coords):
        """method to generate prediction at given coords"""
        pass

    @abstractmethod
    def optimise_parameters(self):
        """an inheriting class should define method for optimising (hyper/variational) parameters"""
        pass

    @property
    @abstractmethod
    def param_names(self) -> list:
        """
        any inheriting class should specify a (property) method that returns the names
        of parameters in a list. Each parameter name should have a get_* and set_* method.
        e.g. if param_names = ['A', 'B'] then methods get_A, set_A, get_B, set_B
        should be defined
        """
        ...

    @timer
    def get_parameters(self, *args, return_dict=True):
        """get parameters"""

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
        """set parameters"""
        # TODO: allow for a nan check?
        for k, v in kwargs.items():
            assert k in self.param_names, f"cannot get parameters for: {k}, it's not in param_names: {self.param_names}"
            # TODO: allow for additional arguments to be supplied?
            #  - or should set_paramname() only take in one argument i.e. the parameter values
            getattr(self, f"set_{k}")(v)

    def set_parameter_constraints(self, **kwargs):
        """set parameter constraints"""
        for k, v in kwargs.items():
            assert k in self.param_names, f"cannot get parameters for: {k}, it's not in param_names: {self.param_names}"
            getattr(self, f"set_{k}_constraints")(**v)

    @abstractmethod
    def get_objective_function_value(self):
        # TODO: to be more general let get_marginal_log_likelihood -> get_objective_function?
        pass


