Changelog
=========

Version 0.4
-----------

- Two command line options for the use of independent Python executables
  and Pytest configurations in the main session and subsessions have
  been added. This enhances the ability for use with e.g. Apptainer,
  allowing a main session outside of a container to spawn containerized
  subsessions. (`#33`_)

.. _#33:  https://github.com/dlr-sp/pytest-isolate-mpi/pull/33

Version 0.3
-----------

- A command line option to disable MPI and/or process isolation has been
  added. This particularly useful to debug MPI-parallel test cases.
  (`#24`_)

- Command line options to set a default test timeout and test timeout
  unit for all MPI-parallel tests have been added. (`#20`_)

.. _#20: https://github.com/dlr-sp/pytest-isolate-mpi/issues/20
.. _#24: https://github.com/dlr-sp/pytest-isolate-mpi/issues/24

Version 0.2
-----------

- An option to customize the command used to launch Pytest in MPI has
  been added. This enables test runs on HPC environments in which
  individual tests are scheduled as jobs via the HPC batch system.
  (`#10`_)

- An unhandled edge case when using a session-scoped fixture in
  non-parametrized tests was fixed. (`#14`_)

- Session-scoped fixtures are now only cached within the MPI-parallel
  Pytest sub sessions. This allows the use of session-scoped fixtures
  which cannot be pickled for non-MPI tests.

- Most of Pytest's CLI options are now passed the MPI-parallel
  sub sessions. (`#10`_)

.. _#10:  https://github.com/dlr-sp/pytest-isolate-mpi/issues/10
.. _#11:  https://github.com/dlr-sp/pytest-isolate-mpi/issues/11
.. _#14:  https://github.com/dlr-sp/pytest-isolate-mpi/pull/14

Version 0.1
-----------

- Initial release.

