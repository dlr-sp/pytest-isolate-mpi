MPI-Parallel, Crash-Safe Tests for Pytest
=========================================

``pytest-isolate-mpi`` is a plugin for `Pytest`_ to enable the test of
MPI-parallel Python software. MPI-parallel tests are executed in forked
MPI jobs with the help of `mpirun`_ to isolate the tests from each
other. Thus, a crash in test only aborts the test, not the whole test
suite. Similarly, deadlocks can be treated with timeouts for tests to
prevent a test suite from never being able to finish.

.. _pytest: https://docs.pytest.org/en/stable/
.. _mpirun: https://docs.open-mpi.org/en/v5.0.x/man-openmpi/man1/mpirun.1.html

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   usage
   api
   changes
   related_work

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
