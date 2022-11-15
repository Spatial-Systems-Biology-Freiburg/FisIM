############
Installation
############

Installing via Package Manager
================================

Install the package from `pypi <https://pypi.org/project/FisInMa/>`__ by running

.. code:: bash

    pip install FisInMa

from the command line if using the pip package manager or 

.. code:: bash

    conda install FisInMa

when using conda environments.
Other package managers should also yield the desired effect but have not been tested.

Building from source
====================

It is also possible to clone the `git repository <https://github.com/Spatial-Systems-Biology-Freiburg/FisInMa>`__
and install from there.
This procedure is possibly unsafe since it installs the current development branch and may yield unwanted results.
Thus the preceding procedures are preferred.
To install from the repository follow these steps:
First clone the repo and build the package

.. code:: bash

    git clone github.com/Spatial-Systems-Biology-Freiburg/FisInMa
    cd FisInMa

    python -m build

Now use your desired package manager to install the local file such as pip

.. code:: bash

    pip install dist/fisinma-*.tar.gz

or conda.

.. code:: bash

    conda install dist/fisinma-*.tar.gz
