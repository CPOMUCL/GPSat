# given a config specifying a results file (+ parameters) plot those values
import copy
import warnings
import json
import re

import numpy as np
import pandas as pd

from GPSat import get_config_path, get_parent_path
from GPSat.utils import get_config_from_sysargv, cprint, json_serializable
from GPSat.dataloader import DataLoader

from GPSat.utils import cprint, get_weighted_values, nested_dict_literal_eval
from GPSat.local_experts import get_results_from_h5file
from GPSat.plot_utils import plots_from_config
from matplotlib.backends.backend_pdf import PdfPages


def get_plot_config():
    # read json file provided as first argument
    # cprint('trying to read in configuration from argument', c="OKCYAN")
    config = get_config_from_sysargv()

    # if not json file provide, config will None, get defaults
    if config is None:
        config_file = get_parent_path("configs", "example_plot_from_results.json")
        warnings.warn(f"\n\nconfig is empty / not provided, will just use an example config:\n{config_file}\n\n")
        with open(config_file, "r") as f:
            config = nested_dict_literal_eval(json.load(f))

        # expect the example results to be present
        config['file'] = get_parent_path("results", "example", "ABC_binned_example.h5")


    return config


if __name__ == "__main__":

    # TODO: clean this script up, wrap sections up into functions
    # TODO: be more clear in comments as to what is going on
    # TODO: determine why ocean_only is not working?

    # ---
    # Config / Parameters
    # ---

    # read in config
    config = get_plot_config()

    # ---
    # initialise
    # ---

    cprint("-" * 30, c="BOLD")
    cprint("will attempt to generate plots using the following config:", c="OKCYAN")
    cprint(json.dumps(json_serializable(config), indent=4), c="HEADER")

    # ---
    # parameters
    # ---

    plot_per_row = 3
    num_plots_row_col_size = {i + 1: {"nrows": i // plot_per_row + 1, "ncols": plot_per_row,
                                      "fig_size": ((plot_per_row * 6), (i // plot_per_row + 1) * 6)}
                              for i in range(20)}
    # ---
    # read in the tables from the file
    # ---

    result_file = config['file']
    table_suffix = config['table_suffix']
    table_names = [p if isinstance(p, str) else p["table"] for p in config['plot_what']]
    table_names = np.unique(table_names).tolist()

    dfs, oi_config = get_results_from_h5file(result_file,
                                             merge_on_expert_locations=True,
                                             select_tables=table_names + ["expert_locs", "oi_config", "run_details"],
                                             table_suffix=table_suffix,
                                             add_suffix_to_table=True)

    # in some occasions need to take the weighted combination of observations
    # e.g.take weighted combination of predictions)


    weighted_vals = config.get("weighted_values", {})
    # - here add the table_suffix on
    for k in list(weighted_vals.keys()):
        cprint(f"adding table_suffix: '{table_suffix}' to key in 'weighted_values' for base table: '{k}'", c="OKBLUE")
        weighted_vals[f"{k}{table_suffix}"] = weighted_vals.pop(k)

    # ---
    # plot by
    # ---

    # get value from expert locations, use that for selection criteria

    plot_by = config["plot_by"]
    expert_locs = dfs[f'expert_locs{table_suffix}']

    coords_col = oi_config[0]['data']['coords_col']
    for idx, oic in enumerate(oi_config):
        assert oic['data']['coords_col'] == coords_col, f"expect for all configs to have same coords_cols," \
                                                        f" oi_config[{idx}] has: {oic['data']['coords_col']}"

    assert plot_by in expert_locs, f"plot_by: {plot_by} is not in expert locations table\ncolumns:{expert_locs.columns}"

    # get unique plot_bys
    plot_by_values = np.unique(expert_locs[plot_by])

    # ---
    # determine the plot configurations
    # ---

    # NOTE: these may get expanded - e.g. for length scales
    plot_what = []
    for pw in copy.deepcopy(config["plot_what"]):

        assert "table" in pw, f"plot_what entry: {pw} is missing a 'table' entry"
        assert "template" in pw, f"plot_what entry: {pw} is missing a 'template' entry"

        # get the template
        template_name = pw.pop("template")
        assert template_name in config[
            'templates'], f"template_name: {template_name} is not in config['templates']: {config['templates'].keys()}"

        plt_temp = config['templates'][template_name]

        # let the plot values be the template, plus any overriding values
        pw = {**plt_temp, **pw}

        # if plot values column not specified use the table name
        # NOTE: this does not make senses for plot_xy_from_results_data()
        if "val_col" not in pw:
            pw["val_col"] = pw["table"]

        # add the table suffix
        pw["table"] = f'{pw["table"]}{config["table_suffix"]}'

        # determine the row_select for the dimensions
        plot_what.append(pw)

    plot_tables = np.unique([pw['table'] for pw in plot_what])

    # ---
    # increment over plot by values
    # ---

    image_file = re.sub("\.h5",
                        f"{config.get('table_suffix', '')}_RESULTS.pdf",
                        config["file"])
    cprint(f"plotting to file:\n{image_file}", c="OKBLUE")

    with PdfPages(image_file) as pdf:

        for pvb in plot_by_values:

            cprint("-" * 50, c="OKCYAN")
            cprint(f"on plot_by value: {pvb}", c="OKCYAN")

            # for the given plot by, select
            elocs = expert_locs.loc[expert_locs[plot_by] == pvb, coords_col]

            # select data
            dfs_tmp = {}
            for k in plot_tables:
                # NOTE: merging on floats might have issues
                keep = pd.merge(dfs[k], elocs,
                                on=coords_col,
                                how="left",
                                indicator=True)["_merge"] == "both"

                dfs_tmp[k] = dfs[k].loc[keep, :].copy(True)

            # take weighted combination of some values?
            for k in list(dfs_tmp.keys()):
                if k in weighted_vals:
                    print(k)
                    _ = get_weighted_values(dfs_tmp[k], **weighted_vals[k])
                    _['_dim_0'] = 0
                    dfs_tmp[k] = _

            # get the row select for each plot_what
            plot_configs = []
            for pw in plot_what:
                # copy to leave original unaffected (TODO: confirm this works as expected)
                pw = copy.deepcopy(pw)
                # get the data
                df = dfs_tmp[pw['table']]
                dim_0 = np.unique(df['_dim_0'])
                col = pw.get("val_col", "y_col")

                # plot title: table\n(optional dimension)\n column

                # tmp_col = f" - {col}" if col != pw['table'] else ""

                # TODO: there is a lot of duplication of code here, refactor / clean up
                if len(dim_0) == 1:
                    # plot title
                    plt_title = f"table:{pw['table']}\ncolumn:{col}"

                    pw["load_kwargs"] = {"row_select": {"col": "_dim_0", "comp": "==", "val": 0}}

                    # plot_kwargs specific to data
                    new_plot_kwargs = {"title": plt_title}

                    # combine with existing, letting the existing overwrite the new if there's overlap
                    pw["plot_kwargs"] = {**new_plot_kwargs, **pw.get("plot_kwargs", {})}
                    plot_configs.append(pw)
                else:

                    for jdx, d in enumerate(dim_0):
                        _ = copy.deepcopy(pw)
                        _["load_kwargs"] = {"row_select": {"col": "_dim_0", "comp": "==", "val": jdx}}

                        plt_title = f"table:{pw['table']}\ncolumn:{col} - dim: {d}"
                        # HACK: for lengthscales only
                        if re.search("^lengthscales", col):
                            plt_title = f"table:{pw['table']}\ncolumn:{col} - dim: {coords_col[jdx]}"#f"{ptitle} - {coords_col[jdx]}"

                        new_plot_kwargs = {"title":  plt_title}
                        _["plot_kwargs"] = {**new_plot_kwargs, **_.get("plot_kwargs", {})}
                        plot_configs.append(_)

            plot_configs = plot_configs

            # ---
            # plot each value using the configuration specified
            # ---

            # TODO: format pvb it's a datetime (i.e. to seconds, from nano seconds)
            st = f"\n{plot_by} == {pvb}"

            fig = plots_from_config(plot_configs, dfs_tmp, num_plots_row_col_size, st)
            pdf.savefig(fig)

    cprint("-" * 50, c="OKGREEN")
    cprint(f"wrote plots of results file contents to:\n{image_file}", "OKGREEN")
