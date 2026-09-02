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
  anywhere. This leads to one process failing while the others
  (potentially) keep running.
* Any process is allowed to call ``MPI_Abort`` at any time and place,
  stopping execution.

These can be grouped into two categories:

* Crashes of the compute environment due to ``MPI_Abort``, segfaults,
  etc.
* Differing control flows that lead to some parts of the code being
  executed on just one process

    * e.g. an ``if`` branch being executed on one process, but not on
      the other, and, in turn, an assert being triggered


To counter these, this code was designed as follows:

* The main process gathers the tests.
* The main process uses ``mpirun`` to generate a parallel, forked
  environment.
* Only the forked environment runs in parallel.
* Communication with the processes happens via file IO.

    * e.g. the results of the tests are written to file by the processes
      and read by the main process.

These decisions have the following benefits:

* In case the tests actually run through, the results of the
  multiple processes can be gathered by the main process and
  joined, leading to a unified output of the test results.
* The forked environment allows tolerating ``MPI_Abort`` and
  segfaults, as the main process is not affected.
* In case the tests catastrophically fail (segfault,
  ``MPI_Abort``), the use of file IO ensures that the
  ``stdout``/``stderr`` output survives the processes and is captured
  by the main process.

