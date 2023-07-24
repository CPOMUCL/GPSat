"""
Add package docstring here
"""

# from GPSat.local_experts import LocalGPExpert, GPflowGPRExpert
# from GPSat.global_interpolation import GlobalInterpolation

import os

# import json
# import re

def get_path(*sub_dir):
    """get_path to package"""
    return os.path.join(os.path.dirname(__file__), *sub_dir)

def get_parent_path(*sub_dir):
    return os.path.join(os.path.dirname(get_path()), *sub_dir)

def get_data_path(*sub_dir):
    return get_parent_path('data', *sub_dir)

def get_config_path(*sub_dir):
    return get_parent_path('configs', *sub_dir)
