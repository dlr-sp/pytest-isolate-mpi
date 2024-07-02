============
Introduction
============

This project was started with the following problem in mind:

*  Having to test MPI-parallel Python programs that link against C/C++/Fortran libraries

This poses a set of problems:

* Differing code paths need to be taken into account for the asserts.
* Deadlocks, i.e. all processes waiting on others, might happen and need to be accounted for.
* Due to linking
