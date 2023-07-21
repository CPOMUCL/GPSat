
import pandas as pd
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import Union, Literal, List, Dict


@dataclass_json
@dataclass
class DataConfig:
    """
    This dataclass provides the configuration for data that is consumed by the
    local expert models. It provides an API for data loading and selection
    TODO: Just use local_experts.LocalExpertData?

    Attributes
    ----------
    data_source: str | pandas DataFrame object | None, default None
        The dataframe or path to a file containing the satellite measurement data.
        Specify either as a string, understood as a file name, or as a pandas DataFrame object.
        The data should contain columns specifying the coordinates of measurements and columns 
        specifying measurement readings.
    table: str | None, default None
        Used only if data_source is a string pointing to an HDF5 file ("*.h5").
        Should be a valid key/table in the HDF5 file pointing to the dataframe of interest.
    obs_col: str | None, default None
        The name of column in the dataframe specifying the measurement readings.
        TODO: Allow multiple columns to accomodate multioutput GPs?
    coords_col: list of str | None, default None
        The name of columns in the dataframe specifying the coordinate readings
        of locations where measurements were obtained. e.g. coords_col = ["x", "y"].
    local_select: list of dict | None, default None
        A list of conditions expressed as dictionaries used to specify a subset of data
        that will be consumed by the local expert models. Each dictionary should have the following keys:
        - 'col': The name(s) of the column(s) (or variables for xarray objects) that we want to
                 impose conditions on. This can be a single string or a list of strings.
        - 'comp': The comparison operator, which is one of '==', '!=', '>=', '>', '<=', or '<'.
        - 'val': value to be compared with.
        
        Comparisons are made in relation to the local expert locations. So
        e.g. local_select = [{"col": "t", "comp": "<=", "val": 1},
                             {"col": "t", "comp": ">=", "val": -1}]
        will select data that is within ±1 of the "t"-coordinate of the local expert location.
        NOTE: the current expert location is a dataframe with a single row, typically with columns ["x", "y", "t"].
    global_select: list of dict | None, default None
        A list of conditions expressed as dictionaries used to select a subset of data onto memory.
        This is used if the full data size is too large to fit on memory.
        Selection can be done statically and/or dynamically. Static selection is done using the same 
        data selection API as 'local_select' above.
        e.g. global_select = [{"col": "A", "comp": ">=", "val": 10}] will store on memory only data
        whose column "A" is greater than or equal to the value 10. i.e., data["A"] >= 10.

        Dynamic selection works in tandem with 'local_select' to allow data selection that depend on local expert locations.
        (NOTE: in our main script, we loop over the local expert locations)
        Each dictionary should have the following keys:
        - 'loc_col': The 'col' argument of the 'local_select' dictionary that we base our selection criteria on.
        - 'src_col': The name(s) of the column(s) in the data that we apply our selection criteria on.
        - 'func': A lambda function written as a string specifying the selection criterion. The lambda function
                  requires two arguments. The first corresponds to the 'loc_col'-column of the current local expert location
                  and the second corresponds to the 'val' argument of the 'local_select' dictionary.

        Letting 'gs' and 'ls' be shorthands for the 'global select' and 'local select' dicts respectively,
        and 'exp_loc' be the current expert location, then a dictionary
        {"col": gs["src_col"], "comp": ls["comp"], "val": gs["func"](exp_loc[ls["loc_col"]], ls["val"])}
        is dynamically created in the loop over expert locations for data selection.
        e.g. local_select = [{"col": "t", "comp": "<=", "val": 1}] 
             global_select = [{"loc_col": "t", "src_col": "A", "func": "lambda x,y: x+y"}]
        This will dynamically create a data selection dictionary {"col": "A", "comp": "<=", "val": exp_loc["t"]+1}.
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
        If not None, the dictionary should contain the following keys:
        - file: A string pointing to a HDF5 file containing the parameter values. The file should
                contain keys/tables corresponding to the name of parameters to be loaded. Each table
                corresponding to a parameter must have the coordinates of the expert locations and
                the values of the parameter as columns. Default is None (see `param_dict` below).
        - param_names: The name of parameters to be loaded in. e.g. param_names = ["lengthscale"].
                       If None, it will load all parameters found in `file`. Default is None.
        - table_suffix: The suffix attached to parameter name in the keys of the HDFStore, used to
                        specify which version of the parameter to use. For instance, the original
                        lengthscale hyperparameter might be stored under the table `lengthscale` and
                        the smoothed out lengthscale might be stored under `lengthscale_SMOOTHED`.
                        Then, to load in the smoothed lengthscale, we set table_suffix = "_SMOOTHED".
                        Default is "" (i.e. no suffix).
        - param_dict: Instead of loading parameters from a file, we can alternatively specify a dictionary
                      with fixed hyperparameter name-value pairs that will be used to instantiate every
                      local expert models. e.g. param_dict = {'lengthscale' : 1.0}

    optim_kwargs: dict | None, default None
        Dictionary of keyword arguments to be passed to the `optimise_parameters` method in the 
        oi_model (see `GPSat.models.BaseGPRModel`). The keyword arguments will vary depending on
        the model and the user is required to check the arguments required to run the
        `optimise_parameters` method for their model of choice.
    pred_kwargs: dict | None, default None
        Dictionary of keyword arguments to be passed to the `predict` method in the oi_model
        (see `GPSat.models.BaseGPRModel`). The keyword arguments will vary depending on the model
        and the user is required to check the arguments in the `predict` method for their model of choice.
    params_to_store: 'all' | list of str, default 'all'
        Specify a list of names of model parameters that the user wishes to store in the results file.
        Set to 'all' by default, which will store all parameters defining the model. Alternatively, one
        can explicitly specify a subset of parameters to store in order to save memory, as storing
        all parameters for all local expert models can get quite heavy.
    """
    MODELS = Literal['GPflowGPRModel',
                     'GPflowSGPRModel',
                     'GPflowSVGPModel',
                     'sklearnGPRModel',
                     'GPflowVFFModel',
                     'GPflowASVGPModel']
    
    oi_model: Union[MODELS, dict, None] = None
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
    This dataclass provides the configuration for the locations of the local experts.
    The attributes of this class are just the arguments passed to the
    `GPSat.LocalExpertOI.set_expert_locations` method, which itself are mostly parameters
    passed to `GPSat.dataloader.DataLoader.load`.

    Attributes
    ----------
    source: str | pandas dataframe | None, default None
        Specify the dataframe or a path to a file containing the coordinate readings of
        the local expert locations. If specifying by a file, the file should contain tabular
        data (e.g. a csv or netcdf file) whose columns include the xy-coordinates of the expert locs.
    where: dict | list of dict | None, default None
        Used when querying a subset of data from HDFStore, DataFrame, DataSet, DataArray.
        Each dictionary should contain the following keys:
        - col: refers to a 'column' (or variable for xarray objects)
        - comp: type of comparison to apply e.g. "==", "!=", ">=", ">", "<=" or "<"
        - val: value to be compared with

        e.g. where = [ {"col": "A", "comp": ">=", "val": 0}] will select entries where columns "A" is greater than 0.
        NOTE: think of this as a database query, used to read data from the file system into memory
    add_data_to_col: dict | None, default None
        Used if we want to add an extra column to the table with constant values (e.g. the date of expert locs).
        This should be specified as a dictionary with variable name-value pairs to be added to the table.
        e.g. add_data_to_col = {"A": 10.0} will append a column "A" to the table with constant value 10.0.
    col_funcs: dict of dict | None, default None
        Used to add or modify columns in the source table. Specified as a dict of dict whose outer dictionary has
        column names to add/modify as keys and the inner dictionary should have the following keys:
        - func: A python lambda function written as a string that specifies how to modify a column or
                if adding a column, how to use existing columns to generate a new column.
        - col_args: The column name used as arguments to the lambda function. This should be a str or list of str 
                    if multiple arguments are used.

        e.g. col_funcs = {"A" : {"func": "lambda x: x+1", "col_args": "A"},
                          "B" : {"func": "lambda x: 2*x", "col_args": "A"}}}
        will (1) modify column "A" by incrementing the original values by 1, and
             (2) modify/add column "B" by doubling the original values in "A".
    col_select: list of str | None, default None
        This is the same as col_select in `GPSat.config_dataclasses.DataConfig`. Possibly redundant?
    row_select: list of dict | None, default None
        This is the same as row_select in `GPSat.config_dataclasses.DataConfig`. Possibly redundant?
    reset_index: bool, default False
        If True, the index of the output DataFrame will be reset.
    source_kwargs: dict | None, default None
        Set if it requires additional keyword arguments to read data from source file.
        (e.g. keyword arguments passed to pd.read_csv, pd.HDFStore or xr.open_dataset)
    verbose: bool
        Boolean to set verbosity. True for verbose, False otherwise.
    sort_by: str | list of str | None, default None
        Column name to sort rows by. This is passed to `pd.DataFrame.sort_values`.
    """
    source: Union[str, pd.DataFrame, None] = None
    where: Union[dict, List[dict], None] = None
    add_data_to_col: Union[dict, None] = None
    col_funcs: Union[Dict[str, dict],  None] = None
    col_select: Union[List[str], None] = None
    row_select: Union[List[dict], None] = None
    reset_index: bool = False
    source_kwargs: Union[dict, None] = None
    verbose: bool = False
    sort_by: Union[str, List[str], None] = None
    # TODO: Remove the following inputs (kept as legacy)
    df: Union[pd.DataFrame, None] = None
    file: Union[str, None] = None
    keep_cols: Union[bool, None] = None

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
    This dataclass provides the configuration for the locations where predictions are made.
    The attributes of this class are arguments passed throughout the
    `GPSat.prediction_locations.PredictionLocations` class.

    Attributes
    ----------
    method: One of "expert_loc", "from_dataframe" or "from_source", default is "expert_loc"
        Select prediction location specification method. The options are:
        - expert_loc: Use expert locations as prediction locations
        - from_dataframe: Specify prediction locations from a pandas dataframe or a CSV file.
        - from_source: Use locations from other sources (e.g. netcdf or HDF5).

    coords_col: list of str | None, default None
        The column names used to specify location coordinates. e.g. coords_col = ['x', 'y'].
        If None, it will use the coords_col in DataConfig.
    df: pandas dataframe | None, default None
        Used if method = "from_dataframe". Specify the dataframe to be used for prediction locations.
        If None, `df_file` should be specified.
    df_file: str | None, default None
        Used if method = "from_dataframe". Specify path to the CSV file containing the prediction locations.
        If None, `df` should be specified.
    max_dist: int | float | None, default None
        Set inference radius i.e. the radius centered at the expert location where predictions are made.
    load_kwargs: dict | None, default None
        Used if method = "from_source". Specify keyword arguments to be passed to
        `GPSat.dataloader.DataLoader.load` to load prediction locations data from source.
    """
    METHODS = Literal["expert_loc", "from_dataframe", "from_source"]

    # For use in prediction_locations.__init__
    method: METHODS = "expert_loc"
    coords_col: Union[List[str], None] = None # To check
    # For use in prediction_locations._from_dataframe
    df: Union[pd.DataFrame, None] = None
    df_file: Union[str, None] = None
    max_dist: Union[int, float, None] = None
    load_kwargs: Union[dict, None] = None
    # For use in prediction_locations._shift_arrays (remove this functionality for simplicity?)
    X_out: Union[str, None] = None # To check.

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


