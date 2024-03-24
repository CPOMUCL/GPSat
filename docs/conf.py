# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os

# add GPSat directory to sys.path
# sys.path.insert(0, os.path.abspath("../GPSat"))
try:
    path_containing_gpsat = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, path_containing_gpsat)
    # sys.path.append(path_containing_gpsat)
except Exception as e:
    print(repr(e))



import GPSat
import pathlib


#sys.path.insert(0, os.path.abspath("../GPSat"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GPSat'
copyright = '2024, CPOM UCL'
author = 'Ronald MacEachern and So Takao'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    "nbsphinx",
    "numpydoc",
    'sphinxemoji.sphinxemoji'
]

[extensions]
todo_include_todos = True
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
pygments_style = 'sphinx'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.sphinx/*']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# allow errors in notebooks
nbsphinx_allow_errors = True
