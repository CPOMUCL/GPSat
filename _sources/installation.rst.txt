Installation
------------

Instructions are for Linux and Mac. Windows coming soon.

.. todo::
    Make package installable via ``pip`` or ``conda`` to make the process simpler.

..  Provide Window install instructions
..  Handle package installs through setup.py

1. Clone repository with the command.

using ``https``:

.. code-block:: console

   $ git clone https://github.com/CPOMUCL/GPSat.git

or ``ssh``:

.. code-block:: console

   $ git clone git@github.com:CPOMUCL/GPSat.git

2. Create a virtual environment.

change directory, create virtual environment, activate virtual environment.

.. code-block:: console

    $ cd GPSat
    $ python -m venv venv
    $ source venv/bin/activate

3. Install Packages.

.. code-block:: console

    $ pip install -r requirements.txt

3. (Optional) Install ``GPSat`` in editable model

Changes made to source code will be reflected immediately, useful for development.

.. code-block:: console

   $ pip install -e ./

4. (Mac Specific) Install HDF5

Requires `homebrew <https://brew.sh/>`_.

.. code-block:: console

   $ brew install hdf5

Export paths. To make permanent add to ~/.zshrc or ~/.bashrc file.:

Apple Silicon

.. code-block:: console

   $ export HDF5_DIR=/opt/homebrew/opt/hdf5

Intel

.. code-block:: console

   $ export HDF5_DIR=/usr/local


.. note::
   If using Mac with M1 chip, need to install appropriate tensorflow version (see `tensorflow-metal <https://developer.apple.com/metal/tensorflow-plugin/>`_).
   This worked: ``SYSTEM_VERSION_COMPAT=0 python -m pip install tensorflow-macos``.
   Also found incompatibility of numba and cartopy. Resolved by install cartopy first and then numba. Perhaps reverse order of installation?
   In addition, there seems to be a conflict with mac tensorflow and cartopy. How to resolve this? Solution: conda install matplotlib==3.2.2.