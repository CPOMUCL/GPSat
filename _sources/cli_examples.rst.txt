Command Line Examples
------------

.. not working: want the world purple to be in purple.
.. TODO add css file to handle coloring words
.. role:: purple


Below are examples of how to run modules or example scripts from the command line.
Each module/script can take a path to a ``.json`` file, providing a configuration, as argument.
Most of the examples below do not take in such an argument, and as such will rely on
default configurations, which are printed to the screen in :purple:`purple` for reference.

Read in Flat Files, Store in HDF5 File
======================================

read data from flat files (.csv, .tsv, etc) and store them in a single .h5

.. code-block:: console

    $ python -m GPSat.read_and_store

Plot Raw Observations
=====================

e.g. those stored in the above


.. code-block:: console

    $ python -m examples.plot_observations

Bin Raw Observations
====================

pre-processing data: in this case remove outliers and bin data.

.. code-block:: console

    $ python -m GPSat.bin_data

Run Optimal Interpolation (OI) on Binned Observations
=====================================================

.. code-block:: console

    $ python -m examples.local_expert_oi

Post Process OI Results: Smooth Hyper Parameters
================================================

run post processing: in this case smooth of hyper parameters

.. code-block:: console

    $ python -m GPSat.postprocessing

Run OI by Loading Hyper-Parameters
==================================

in this case load the smoothed hyper parameters from above, don't optimise.

this is done by providing a configuration file, generated in the previous step.

.. code-block:: console

    $ python -m examples.local_expert_oi results/example/ABC_binned_example_SMOOTHED.json


Generate Plots of the Predictions
=================================

selecting the predictions generated when using the smoothed hyper parameters

.. code-block:: console

    $ python -m examples.plot_from_results


