============
Related Work
============

The following projects were considered:

* pytest
    * industry standard unit testing framework that was chosen as target
* `pytest-mpi <https://github.com/aragilar/pytest-mpi>`_
    * The plugin allows for MPI-parallel execution of tests, i.e. running multiple tests in parallel.
    * It does not implement MPI-parallel tests.
* `pytest-forked <https://github.com/pytest-dev/pytest-forked>`_ / `pytest-isolate <https://github.com/gilfree/pytest-isolate/tree/master>`_
    * The plugins runs tests in a forked subprocess.
    * This gives safety w.r.t. segfaults happening in the subprocess.
    * It does not allow for MPI-parallel execution.
* `testflo <https://github.com/OpenMDAO/testflo>`_
    * `testflo` implements MPI-parallel execution of tests
    * But for unittest and without safety w.r.t. segfaults and MPI_Abort
