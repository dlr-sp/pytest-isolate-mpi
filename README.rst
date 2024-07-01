==================
pytest-isolate-mpi
==================

pytest-isolate-mpi allows for MPI-parallel tests being executed in a segfault and MPI_Abort safe manner


Features
--------

* TODO

Setting up a New Development Environment
----------------------------------------

1.  Obtain the source code::

        git clone git@gitlab.dlr.de:dlr-sp/pytest-isolate-mpi.git
        cd pytest_isolate_mpi

2.  Install a new Python |venv|_::

        python3 -m venv venv
        source venv/bin/activate
        pip install -e ".[dev]"


    Using ``venv`` is not strictly necessary, but recommended to isolate
    the procect dependencies from the ones of other projects. In the
    ``pip`` step, the development dependencies to run the tests and
    generate the HTML documentation are installed as well. By passing
    the flag ``-e`` the package is installed in editable mode. Changes
    to the source code don't have to be installed to become effective
    when running ``venv/bin/pytest_isolate_mpi``.

.. |venv| replace:: ``venv``
.. _venv: https://docs.python.org/3/library/venv.html

Running pytest-isolate-mpi
--------------------------

After the setup of the development environment, the project can be run
from the project directory with::

    pytest_isolate_mpi data/sample.txt

This requires the virtual environment to be activated. In new shells,
this step has to be repeated once from the project folder with::

    source venv/bin/activate

