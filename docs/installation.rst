Installation
------------

.. todo::
   Make package installable via ``pip`` or ``conda`` to make the process simpler.

1. Clone repository with the command.

.. code-block:: console

   $ git clone https://github.com/CPOMUCL/GPSat.git

2. Set up a virtual environment (best with conda).

.. code-block:: console

   $ conda create -n gpsat_env python=3.8
   $ conda activate gpsat_env

3. Install ``GPSat``.

.. code-block:: console

   $ pip install -e .

4. Install HDF5 and PROJ, and add export paths to ~/.zshrc or ~/.bashrc file.

.. code-block:: console

   $ brew install hdf5
   $ brew install proj
   $ export HDF5_DIR=/opt/homebrew/opt/hdf5
   $ export PROJ_DIR=/opt/homebrew/opt/proj

.. todo::
   Currently command applies for mac-os. Make it more general.

5. Install ``cartopy``.

.. code-block:: console

   $ conda install -c conda-forge cartopy==0.20.2

6. Install packages in ``requirements.txt``.

.. code-block:: console

   $ pip install -r requirements.txt


.. note::
   If using Mac with M1 chip, need to install appropriate tensorflow version (see `tensorflow-metal <https://developer.apple.com/metal/tensorflow-plugin/>`_).
   This worked: ``SYSTEM_VERSION_COMPAT=0 python -m pip install tensorflow-macos``.
   Also found incompatibility of numba and cartopy. Resolved by install cartopy first and then numba. Perhaps reverse order of installation?
   In addition, there seems to be a conflict with mac tensorflow and cartopy. How to resolve this? Solution: conda install matplotlib==3.2.2.