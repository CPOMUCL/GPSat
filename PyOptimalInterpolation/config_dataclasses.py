
import pandas as pd
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import Union, Literal, List


@dataclass_json
@dataclass
class DataConfig:
    """
    Just use local_experts.LocalExpertData?
    ...
    """
    data_source: Union[str, None] = None
    table:  Union[str, None] = None
    obs_col: Union[str, None] = None
    coords_col: Union[List[str], None] = None
    local_select: Union[List[dict], None] = None
    global_select: Union[List[dict], None] = None
    row_select: Union[List[dict], None] = None # TO check
    col_select: Union[List[dict], None] = None # TO check
    col_funcs:  Union[List[str], dict, None] = None # TO check
    engine:  Union[str, None] = None
    read_kwargs: Union[dict, None] = None


@dataclass_json
@dataclass
class ModelConfig:
    """
    This dataclass provides configurations for the local expert models
    (arguments to `local_expert_oi.set_model` method)
    ----------
    oi_model:
    init_params:
    constraints:
    load_params:
    optim_kwargs:
    replacement_threshold:
    replacement_model:
    replacement_init_params:
    replacement_constraints:
    replacement_optim_kwargs:
    """
    MODELS = Literal['GPflowGPRModel',
                     'GPflowSGPRModel',
                     'GPflowSVGPModel',
                     'sklearnGPRModel',
                     'GPflowVFFModel',
                     'GPflowASVGPModel']
    
    oi_model: Union[MODELS, None] = None
    init_params: Union[dict, None] = None
    constraints: Union[dict, None] = None
    load_params: Union[dict, None] = None # Provide arguments to `local_expert_oi.load_params`
    optim_kwargs: Union[dict, None] = None # Provide arguments to `model.optimise_parameters`
    replacement_threshold: Union[int, None] = None
    replacement_model: Union[MODELS, None] = None
    replacement_init_params: Union[dict, None] = None
    replacement_constraints: Union[dict, None] = None
    replacement_optim_kwargs: Union[dict, None] = None
    # pred_kwargs: Union[dict, None] = None # Provide arguments to `model.predict`. Remember to add back.
    # replacement_pred_kwargs: Union[dict, None] = None 


@dataclass_json
@dataclass
class ExpertLocsConfig:
    """
    (arguments to `local_expert_oi.set_expert_locations` method)
    """
    df: Union[pd.DataFrame, None] = None
    file: Union[str, None] = None
    source: Union[str, None] = None
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


@dataclass_json
@dataclass
class PredictionLocsConfig:
    """
    (arguments passed to `PyOptimalInterpolation.prediction_locations.PredictionLocations` class)
    """
    # For use in prediction_locations.__init__
    method: str = "expert_loc"
    coords_col: Union[str, None] = None # To check
    expert_loc: Union[str, None] = None # To check
    # For use in prediction_locations._shift_arrays
    X_out: Union[str, None] = None # To check.
    # For use in prediction_locations._from_dataframe
    df: Union[pd.DataFrame, None] = None
    df_file: Union[str, None] = None
    max_dist: Union[int, float] = None
    copy_df: bool = False


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
    comment: Union[str, None] = None


