.. _installation:

Installation
============

hpiPy can be installed using pip:

.. code-block:: bash

   pip install hpipy

Development Installation
------------------------

For development installation, we recommend using `uv <https://github.com/astral-sh/uv>`_:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/reidjohnson/hpipy.git
      cd hpipy

2. Create a virtual environment:

   .. code-block:: bash

      uv venv

3. Install development dependencies:

   .. code-block:: bash

      uv pip install --requirements pyproject.toml --extra dev

4. Run the test suite to verify the installation:

   .. code-block:: bash

      pytest

Dependencies
------------

Core dependencies:

* numpy
* pandas
* pytorch
* scikit-learn
* statsmodels

Optional dependencies for development:

* pytest
* sphinx
* black
* flake8
