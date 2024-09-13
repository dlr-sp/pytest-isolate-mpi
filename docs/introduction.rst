============
Introduction
============

This project was started with the following problem in mind:

*  Having to test MPI-parallel Python programs that link against
   C/C++/Fortran libraries

This poses a set of problems:

* Differing code paths need to be taken into account for the asserts.
* Deadlocks, i.e. all processes waiting on others, might happen and need
  to be accounted for.
* Due to using background libraries that are not memory safe, each
  process might encounter a segfault at any time and
  any place. This leads to one process failing while the others
  (potentially) run on.
* Any process is allowed to call ``MPI_Abort`` at any time and place,
  stopping the execution.

These can be grouped in two categories

* Crashes of the compute environment due to ``MPI_Abort``, segfaults,
  etc.
* Differing control flows that lead to some parts of the code being
  executed on just one process

    * E.g. an if being triggered on one process, but not on the other
      and, in turn, an assert triggering


To counter these, this code was designed as follows:

* The main process gathers the tests.
* The main process uses ``mpirun`` to generate a parallel, forked
  environment
* Only the forked environment runs in parallel
* Communication with the processes happens via file IO

    * e.g. the results of the tests are written to file by the processes
      and read by the master process.

These decisions have the following benefits:

* In case of the tests actually running through, the results of the
  multiple processes can, then, be gathered on the main process and
  joined, leading to a unified output of the test results.
* The forked environment allows to tolerate MPI_Abort and segfaults
  happening, as the main process is not touched.
* in case of the tests catastrophically failing (segfault,
  ``MPI_Abort``), the usage of file IO leads to the
  ``stdout``/``stderr`` surviving the processes and output for the tests
  being captured by the main process.

