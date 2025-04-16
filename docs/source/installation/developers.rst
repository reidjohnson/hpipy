.. _developers:

Developer's Guide
=================

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

      uv run pytest

Dependencies
------------

Optional dependencies for development:

* black
* flake8
* isort
* mypy
* pytest
* pytest-cov
* sphinx
* types-python-dateutil

Test and Coverage
-----------------

Ensure that `pytest` and `pytest-cov` are installed::

  uv run pip install pytest pytest-cov

To test the code::

  uv run pytest hpipy -v

To test the code and produce a coverage report::

  uv run pytest hpipy --cov-report html --cov=hpipy

To test the documentation::

  uv run pytest --doctest-glob="*.rst" --doctest-modules docs

Documentation
-------------

To build the documentation, run::

  uv pip install -r ./docs/sphinx_requirements.txt
  uv run sphinx-build -b html ./docs/source ./docs/_build
