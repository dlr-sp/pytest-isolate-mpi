============
Related Work
============

The following projects were considered:
* pytest
  * industry standard unit testing framework that was chosen as target
* `pytest-mpi <https://pypi.org/project/pytest-mpi/>`
  * MPI-parallel execution of tests, but not execution of parallel tests
* `pytest-forked <https://github.com/pytest-dev/pytest-forked>` / `pytest-isolate <https://github.com/gilfree/pytest-isolate/tree/master>`
  * running of tests in a subprocess, albeit without MPI-parallel execution
* `testflo <https://github.com/OpenMDAO/testflo>`
  * MPI-parallel execution of tests, but for unittest and without safety w.r.t. segfaults and MPI_Abort

