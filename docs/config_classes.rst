Configuration dataclasses
=========================
Experiments on GPSat using the ``LocalExpertOI`` API works by specifying a set of configurations, one for the following components that build
up an experiment:

- :ref:`Data configuration <data_config>`
- :ref:`Model configuration <model_config>`
- :ref:`Expert location configuration <xpert_config>`
- :ref:`Predcition location configuration <pred_config>`

Additionally, we have a dataclass that configures running details:

- :ref:`Run configuration <run_config>`

and a dataclass that configures the entire experiment, which includes
all of the above dataclasses.

- :ref:`Experiment configuration <experiment_config>`

.. _data_config:

Data configuration
------------------
.. autoclass:: GPSat.config_dataclasses.DataConfig
   :undoc-members:
   :show-inheritance:
   :exclude-members:


.. _model_config:

Model configuration
-------------------
.. autoclass:: GPSat.config_dataclasses.ModelConfig
   :undoc-members:
   :show-inheritance:
   :exclude-members:


.. _xpert_config:

Expert location configuration
-----------------------------
.. autoclass:: GPSat.config_dataclasses.ExpertLocsConfig
   :undoc-members:
   :show-inheritance:
   :exclude-members:


.. _pred_config:

Prediction location configuration
---------------------------------
.. autoclass:: GPSat.config_dataclasses.PredictionLocsConfig
   :undoc-members:
   :show-inheritance:
   :exclude-members:


.. _run_config:

Run configuration
-----------------
.. autoclass:: GPSat.config_dataclasses.RunConfig
   :undoc-members:
   :show-inheritance:
   :exclude-members:


.. _experiment_config:

Experiment configuration
------------------------
.. autoclass:: GPSat.config_dataclasses.ExperimentConfig
   :undoc-members:
   :show-inheritance:
   :exclude-members: