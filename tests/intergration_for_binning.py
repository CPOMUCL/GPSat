# compare binned values from current methodology with previous to ensure consistent
import pandas as pd
import numpy as np

from GPSat import get_parent_path
from GPSat.dataloader import DataLoader
from GPSat.utils import nested_dict_literal_eval, compare_dataframes
# TODO: move BinData into GPSat - into dataprepper?
# from examples.bin_raw_data_from_hdf5_by_batch import BinData
from GPSat.bin_data import BinData

if __name__ == "__main__":

    # pre-existing binned data (made before changes)
    bin_file = get_parent_path("data", "example", "ABC_binned.h5")

    # --
    # load reference data
    # --
    df = DataLoader.load(source=bin_file, table="data", reset_index=False)

    config = DataLoader.get_attribute_from_table(source=bin_file,
                                                 table="data",
                                                 attribute_name="bin_config")

    config = nested_dict_literal_eval(config)

    # ---
    # bin data
    # ---

    bd = BinData(**config)

    bin_df, stats = bd.bin_data()

    # ---
    # compare dataframe
    # ---

    bc = config['bin_config']
    merge_on = bc['by_cols'] + [bc['x_col'], bc['y_col']]
    columns_to_compare = bc['val_col']
    chk = compare_dataframes(df, bin_df,
                             merge_on=merge_on,
                             columns_to_compare=columns_to_compare)

    assert chk[f'{bc["val_col"]}_abs_diff'].max() == 0

    # ---
    # check configs
    # ---

    # config_names = ['raw_data_config', 'bin_config']
