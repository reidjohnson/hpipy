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

      pytest

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

Test and Coverage
-----------------

Ensure that `pytest` and `pytest-cov` are installed::

  pip install pytest pytest-cov

To test the code::

  python -m pytest hpipy -v

To test the code and produce a coverage report::

  python -m pytest hpipy --cov-report html --cov=hpipy

To test the documentation::

  python -m pytest --doctest-glob="*.rst" --doctest-modules docs

Documentation
-------------

To build the documentation, run::

  pip install -r ./docs/sphinx_requirements.txt
  sphinx-build -b html ./docs/source ./docs/_build
