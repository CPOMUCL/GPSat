.. GPSat Documentation documentation master file, created by
   sphinx-quickstart on Mon Jul 24 09:21:39 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GPSat's documentation!
=================================

``GPSat`` is a Python library to perform spatio-temporal inference using local Gaussian process models.
Its primary use case is optimal interpolation, where the goal is to infer an underlying field,
such as sea surface height, from satellite measurements of the field.

.. \[Put example pictures?\]

Below we highlight what ``GPSat`` can do and what it cannot do:

| |:white_check_mark:| Modelling spatio-temporal data with possibly varying characteristics such as lengthscales.
| |:white_check_mark:| Handling large spatio-temporal data with millions of data points.
| |:white_check_mark:| Recovering along-track signals from noisy satellite measurements.
| |:no_entry:| Modelling data in high dimensions.
| |:no_entry:| Small and sparse data sets.

Benefits of Local GPs for spatial modelling
-------------------------------------------

``GPSat`` harnesses the power of local Gaussian process models, which process small chunks of data at a time. This approach enables the library
to efficiently handle vast amounts of data that would be infeasible with a single Gaussian process.
Moreover, the locality allows for capturing spatial and temporal variations in the data that a single Gaussian process will not be able to learn.

Supported Enhancements
----------------------

- **GPU Acceleration:** ``GPSat`` uses supports GPU usage for accelerated computing, enabling faster training and inference.
- **Sparse GPs:** The library provides support for using sparse Gaussian process models to handle moderately large data per local expert.


.. toctree::
   :maxdepth: 1
   :caption: Getting started

   installation
   cli_examples
   notebooks/gp_regression
   notebooks/using_gpus
   notebooks/1d_local_expert_model_part_1
   notebooks/1d_local_expert_model_part_2
   notebooks/inline_example
   notebooks/dataloader
   notebooks/bin_data

.. toctree::
   :maxdepth: 1
   :caption: Advanced tutorials

.. todo::
   Add tutorials on sparse GP, working from large files, hyperparameter smoothing, custom models, writing to and reading from json config files.

.. todo::
   Add API reference for LocalExpertOI, Postprocessing module, DataLoader, utils.

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   config_classes
   local_experts
   GPSat.models
   postprocessing
   dataloader
   utils
   GPSat



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
