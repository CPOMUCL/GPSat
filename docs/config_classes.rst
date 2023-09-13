Configuration dataclasses
=========================
Experiments on ``GPSat`` using the ``LocalExpertOI`` API works by specifying a set of configurations, one for the following components that build
up an experiment:

- :ref:`Data configuration <data_config>`
- :ref:`Model configuration <model_config>`
- :ref:`Expert location configuration <xpert_config>`
- :ref:`Predcition location configuration <pred_config>`

Additionally, we have a class that configures running details:

- :ref:`Run configuration <run_config>`

and a class that configures the entire experiment:

- :ref:`Experiment configuration <experiment_config>`

All of these configurations are subclasses of ``dataclass_json`` (see `dataclasses-json <https://pypi.org/project/dataclasses-json/>`_),
enabling us to read/write configurations to/from JSON objects.
This can be useful for keeping track of experiments and for reproducibility purposes.

Example
-------

Here, we provide an example of a typical ``GPSat`` experiment workflow using the ``LocalExpertOI`` API.
The configurations used to instantiate the ``LocalExpertOI`` object are the configuration dataclasses.

**Note:** We can also provide the configurations in the form of dictionaries for backward compatibility.

.. code-block:: python

   import json
   import numpy as np
   import pandas as pd
   from GPSat.config_dataclasses import *
   from GPSat.local_experts import LocalExpertOI

   # Construct toy data
   obs = np.random.randn(4)
   df = pd.DataFrame(data={'x': [1, 2, 3, 4], 'y': obs})

   # Set expert locations
   xpert_locs = pd.DataFrame(data={'x': [2, 4]})
   pred_locs = pd.DataFrame(data={'x': np.linspace(0, 5, 10)})

   # Set configurations
   data_config = DataConfig(data_source=df,
                            obs_col=['y'],
                            coords_col=['x'])
   model_config = ModelConfig(oi_model="sklearnGPRModel")
   xpert_config = ExpertLocsConfig(source=xpert_locs)
   pred_config = PredictionLocsConfig(method="from_dataframe",
                                      df=pred_locs)

   # Run experiment
   store_path = "/file/to/store"
   locexp = LocalExpertOI(data_config=data_config,
                           model_config=model_config,
                           expert_loc_config=xpert_config,
                           pred_loc_config=pred_config)

   locexp.run(store_path=store_path, check_config_compatible=False)


We can store the experiment configurations to JSON format for reproducibility:

.. code-block:: python

   # Create run configuration
   run_config = RunConfig(store_path=store_path, check_config_compatible=False)

   # Create experiment configuration
   comment = "Configuration for toy experiment"
   experiment_config = ExperimentConfig(data_config,
                                        model_config,
                                        xpert_config,
                                        pred_config,
                                        run_config,
                                        comment)

   # Convert configuration to json format
   config_json = experiment_config.to_json()

   # Save configuration to json file
   with open("example_config.json", "w") as f:
      f.write(config_json)

We can then load this JSON file to reproduce the experiment:

.. code-block:: python

   # Load json file (output is a dict)
   with open('example_config.json', 'r') as f:
      json_object = json.load(f)

   # Convert dict to ExperimentConfig object
   config_json = ExperimentConfig.from_dict(json_object)

   # Run experiment with the same configurations as before
   locexp = LocalExpertOI(config_json.data_config,
                          config_json.model_config,
                          config_json.expert_locs_config,
                          config_json.prediction_locs_config)

   locexp.run(**config_json.run_config.to_dict())


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