
import pandas as pd
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import Union, Literal, List, Dict


@dataclass_json
@dataclass
class DataConfig:
    """
    This dataclass provides the configuration for data that is consumed by the
    local expert models.
    TODO: Just use local_experts.LocalExpertData?

    Attributes
    ----------
    data_source: str | pandas DataFrame object | None, default None
        The dataframe or path to a file containing the satellite measurement data.
        Specify either as a string, which would be understood as a file name,
        or as a pandas DataFrame object. The data should contain columns specifying
        the coordinates of measurements and columns specifying measurement readings.
    table: str | None, default None
        Used only if data_source is a string pointing to an HDF5 file ("*.h5").
        Should be a valid key/table in the HDF5 file pointing to the dataframe of interest.
    obs_col: str | None, default None
        The name of column in the dataframe specifying the measurement readings.
        TODO: Allow multiple columns to accomodate multioutput GPs?
    coords_col: list of str | None, default None
        The name of columns in the dataframe specifying the coordinate readings
        of locations where measurements were obtained.
    local_select: list of dict | None, default None
        A list of conditions expressed as dictionaries used to specify a subset of data
        that will be used to train each local expert model. The conditions are relative 
        to the local expert locations. Each dictionary should have the following keys:
        - 'col': The name(s) of the column(s) that we want to impose a condition on.
                 This can be a single string or a list of strings.
        - 'comp': The comparison operator, which is one of '==', '!=', '>=', '>', '<=', or '<'.
        - 'val': value to be compared with.
        e.g. local_select = [{"col": "t", "comp": "<=", "val": 4},
                             {"col": "t", "comp": ">=", "val": -4}]
        will select data that is within ±4 days of the expert location.
    global_select: list of dict | None, default None
        A list of conditions expressed as dictionaries used to select a subset of data onto memory.
        This is used if the data size is too large to fit on memory. Selection can be done statically
        and/or dynamically. Static selection is done using the same API as 'local_select' above.
        e.g. global_select = [{"col": "lat", "comp": ">=", "val": 60}] will store on memory only data
        that is above 60 degrees in latitude.
        For dynamic selection, each dictionary should have the following keys:
        - 'loc_col': The 'col' argument of 'local_select' that we use as reference
        - 'src_col': The name(s) of the column(s) in the dataframe that we want to impose a condition on
        - 'func': A lambda function written as string used for the selection criterion.
        e.g. local_select = [{"col": "t", "comp": "<=", "val": 4},
                             {"col": "t", "comp": ">=", "val": -4}] 
             global_select = {"loc_col": "t", "src_col": "date",
                               "func": "lambda x,y: np.datetime64(pd.to_datetime(x+y, unit='D'))"}
        will dynamically load data that is within ±4 days of the reference date.
    row_select: list of dict | None, default None
        Used to select a subset of data AFTER data is initially read into memory.
        Can be same type of input as 'local_select' i.e.
        {"col": "A", "comp": ">=", "val": 0} or use col_funcs that return bool array.
        e.g. {"func": "lambda x: ~np.isnan(x)", "col_args": 1}
    col_select: list of str | None, default None
        If list of str, it will return a subset of columns. All values must be valid.
        If None, all columns will be returned.
    col_funcs: dict | None
        If dict, it will be provided to add_cols method to add or modify columns.
    engine: str | None, default None
        Used to specify the file type of data, if reading from file.
        If None, it will automatically infer engine from the file name of 'data_source'.
    read_kwargs: dict | None, default None
        Keyword arguments for reading in data from source.

    Methods
    -------
    to_dict:

    to_dict_with_dataframe:

    from_dict:

    to_json:

    from_json:

    from_json_with_dataframe:
    
    """
    data_source: Union[str, pd.DataFrame, None] = None
    table:  Union[str, None] = None
    obs_col: Union[str, None] = None
    coords_col: Union[List[str], None] = None
    local_select: Union[List[dict], None] = None
    global_select: Union[List[dict], None] = None
    row_select: Union[List[dict], None] = None # TO check
    col_select: Union[List[str], None] = None # TO check
    col_funcs:  Union[List[str], dict, None] = None # TO check
    engine:  Union[str, None] = None
    read_kwargs: Union[dict, None] = None

    def __post_init__(self):
        """
        If data_source is specified as a pandas dataframe, it will be converted
        into a dictionary for the purpose of saving it in json format.
        """
        if isinstance(self.data_source, pd.DataFrame):
            self.data_source = self.data_source.to_dict()

    def to_dict_with_dataframe(self):
        """
        Convert to dictionary while restoring pandas DataFrame where necessary.
        A naive to_dict() method does not work with pandas DataFrame object.
        """
        config_dict = self.to_dict()
        if isinstance(config_dict['data_source'], dict):
            config_dict['data_source'] = pd.DataFrame.from_dict(config_dict['data_source'])
        return config_dict

    @staticmethod
    def from_json_with_dataframe(json_str: str):
        """
        Instantiate a DataConfig object from a json string.
        Use this method instead of `from_json` if the attribute 'data_source' is a
        pandas dataframe but it can also be used otherwise.
        """
        config = DataConfig.from_json(json_str)
        if isinstance(config.data_source, dict):
            config.data_source = pd.DataFrame.from_dict(config.data_source)
        return config


@dataclass_json
@dataclass
class ModelConfig:
    """
    This dataclass provides the configuration for the local expert models used to
    interpolate data in a local region. The attributes of this class are just the
    arguments passed through the `GPSat.LocalExpertOI.set_model` method.

    Attributes
    ----------
    oi_model: One of pre-implemented models: 'GPflowGPRModel', 'GPflowSGPRModel',
              'GPflowSVGPModel', 'sklearnGPRModel', 'GPflowVFFModel' or 'GPflowASVGPModel'
              | dict | None, default None
        Specify the local expert model used to run optimal interpolation (OI) in a local
        region. Some basic models are implemented already in GPSat in `GPSat.models` and
        can be selected by passing their model class name (e.g. oi_model = 'GPflowGPRModel').
        For custom models, specify a dictionary with the keys:
        - path_to_model: a string specifying the path to a file where the model is implemented.
        - model_name: a string specifying the class name of the model. The model is
                      required to be a subclass of the `GPSat.models.BaseGPRModel` class.
        e.g. oi_model = {'path_to_model': 'path/to/model', 'model_name' = 'CustomModel'}
        will select the model 'CustomModel' in the file 'path/to/model'.
    init_params: dict | None, default None
        A dictionary of keyword arguments used to instantiate the above `oi_model`.
        Note that the keyword arguments depend on the oi_model and the user is expected
        to check the parameters in the __init__ method of their model of choice.
    constraints: dict of dict | None, default None
        Specify constraints on the hyperparameters of the oi_model. The outer dictionary
        has the hyperparameter name as keys and the inner dictionary should have the keys:
        - low: The lower bound for the hyperparameter. Can be float or a list of floats if the 
               hyperparameter is multidimensional. If None, no bound is set.
        - high: The upper bound for the hyperparameter. Can be None, int or a list as before.
        e.g. constraints = {'lengthscale': 'low': 0.1, 'high': 10.} will set the lengthscale
        to be within 0.1 and 10 during optimisation.
    load_params: dict | None, default None
        Dictionary of keyword arguments to be passed to `GPSat.LocalExpertOI.load_params` method.
        This is used to dynamically load parameters (saved in a separate file) when initialising
        models, instead of initialising with the default values. Intended use case is during the 
        inference step where we want to make predictions with a pre-determined set of parameters.
        If None, each local expert model will be instantiated with their default parameters values.
        If not None, the dictionary should contain the following key:
        - file: A string pointing to a HDF5 file containing the parameter values. The file should
                contain keys/tables corresponding to the name of parameters to be loaded.
        Additionally, the dictionary may contain the following keys:
        - param_names: The name of parameters to be loaded in. If not specified, it will load all
                       parameters found in `file`.
        - table_suffix: 
        - param_dict:
        - previos:
        - previous_params:
        - index_adjust:
        - param_dict:
    optim_kwargs: dict | None, default None
        Dictionary of keyword arguments to be passed to the `optimise_parameters` method in the 
        oi_model (see `GPSat.models.BaseGPRModel`). The keyword arguments will vary depending on
        the model and the user is required to check the parameters required in the
        `optimise_parameters` method for their model of choice.
    pred_kwargs: dict | None, default None
        Dictionary of keyword arguments to be passed to the `predict` method in the oi_model
        (see `GPSat.models.BaseGPRModel`). The keyword arguments will vary depending on the model
        and the user is required to check the parameters required in the `predict` method for their
        model of choice.
    params_to_store: 'all' | list of str, default 'all'
        Specify a list of names of model parameters that the user wants to store in the results file.
        By default, it is set to 'all', which will store all parameters defining the model. Instead, one
        can explicitly specify the parameters to store in order to save on memory as it can get quite
        heavy to store all parameters for all local expert models used.
    """
    MODELS = Literal['GPflowGPRModel',
                     'GPflowSGPRModel',
                     'GPflowSVGPModel',
                     'sklearnGPRModel',
                     'GPflowVFFModel',
                     'GPflowASVGPModel']
    
    oi_model: Union[MODELS, None] = None
    init_params: Union[dict, None] = None
    constraints: Union[Dict[str, dict], None] = None
    load_params: Union[dict, None] = None # Provide keyword arguments to `local_expert_oi.load_params`
    optim_kwargs: Union[dict, None] = None # Provide arguments to `model.optimise_parameters`
    pred_kwargs: Union[dict, None] = None # Provide arguments to `model.predict`. Remember to add back.
    params_to_store: Union[Literal['all'], List[str]] = 'all' # Provide parameters to save. If 'all' then all parameters in model are saved.
    # TODO: Deprecate the replacement model functionality?
    replacement_threshold: Union[int, None] = None # Use replacement model if number of data points is below replacement threshold
    replacement_model: Union[MODELS, None] = None
    replacement_init_params: Union[dict, None] = None
    replacement_constraints: Union[dict, None] = None
    replacement_optim_kwargs: Union[dict, None] = None
    replacement_pred_kwargs: Union[dict, None] = None 


@dataclass_json
@dataclass
class ExpertLocsConfig:
    """
    (arguments to `local_expert_oi.set_expert_locations` method)
    ----------
    df:
    file:
    source:
    where:
    add_data_to_col:
    col_funcs:
    keep_cols:
    col_select:
    row_select:
    sort_by:
    reset_index:
    source_kwargs:
    verbose:
    """
    df: Union[pd.DataFrame, None] = None
    file: Union[str, None] = None
    source: Union[str, pd.DataFrame, None] = None
    where: Union[str, None] = None # To check
    add_data_to_col: Union[bool, None] = None # To check
    col_funcs: Union[str,  None] = None # To check
    keep_cols: Union[bool, None] = None # To check
    col_select: Union[List[str], None] = None # To check
    row_select: Union[List[str], None] = None # To check
    sort_by: Union[str, None] = None # To check
    reset_index: bool = False
    source_kwargs: Union[dict, None] = None # To check
    verbose: bool = False

    def __post_init__(self):
        if isinstance(self.df, pd.DataFrame):
            self.df = self.df.to_dict()
        if isinstance(self.source, pd.DataFrame):
            self.source = self.source.to_dict()

    def to_dict_with_dataframe(self):
        """
        Convert to dictionary while restoring pandas DataFrame where necessary.
        A naive to_dict() method does not work with pandas DataFrame object.
        """
        config_dict = self.to_dict()
        if isinstance(config_dict['df'], dict):
            config_dict['df'] = pd.DataFrame.from_dict(config_dict['df'])
        if isinstance(config_dict['source'], dict):
            config_dict['source'] = pd.DataFrame.from_dict(config_dict['source'])
        return config_dict


@dataclass_json
@dataclass
class PredictionLocsConfig:
    """
    (arguments passed to `GPSat.prediction_locations.PredictionLocations` class)
    ----------
    method:
    coords_col:
    expert_loc:
    X_out:
    df:
    df_file:
    max_dist:
    copy_df:
    """
    # For use in prediction_locations.__init__
    method: str = "expert_loc" # Check Literal types ["expert_loc", "from_dataframe", ...?]
    coords_col: Union[str, None] = None # To check
    expert_loc: Union[str, None] = None # To check
    # For use in prediction_locations._shift_arrays
    X_out: Union[str, None] = None # To check.
    # For use in prediction_locations._from_dataframe
    df: Union[pd.DataFrame, None] = None
    df_file: Union[str, None] = None
    max_dist: Union[int, float] = None
    copy_df: bool = False

    def __post_init__(self):
        if isinstance(self.df, pd.DataFrame):
            self.df = self.df.to_dict()

    def to_dict_with_dataframe(self):
        """
        Convert to dictionary while restoring pandas DataFrame where necessary.
        A naive to_dict() method does not work with pandas DataFrame object.
        """
        config_dict = self.to_dict()
        if isinstance(config_dict['df'], dict):
            config_dict['df'] = pd.DataFrame.from_dict(config_dict['df'])
        return config_dict


@dataclass_json
@dataclass
class RunConfig:
    """
    arguments passed to `LocalExpertOI.run` class
    """
    store_path: str
    store_every: int = 10
    check_config_compatible: bool = True
    skip_valid_checks_on: Union[List[int], None] = field(default_factory=list)
    optimise: bool = True
    predict: bool = True
    min_obs: int = 3
    table_suffix: str = ""


@dataclass_json
@dataclass
class LocalExpertConfig:
    data_config: DataConfig
    model_config: ModelConfig
    expert_locs_config: ExpertLocsConfig
    prediction_locs_config: PredictionLocsConfig
    run_config: RunConfig
    comment: Union[str, None] = None


