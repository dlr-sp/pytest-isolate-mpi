Changelog
=========

Version 0.4
-----------
- Fixed VS Code test execution for MPI tests by preventing
  ``vscode_pytest`` from being forwarded to MPI subprocesses and by
  keeping report node IDs stable across MPI ranks. (`#39`_)

- Fixed compatibilty with ``pytest-forked`` so that non-MPI tests can still be
  executed with ``--forked`` when ``pytest-isolate-mpi`` is installed. (`#38`_)

- Compatibility with Pytest >= 9.1 has been restored. Pytest's
  ``tmp_path_factory``, which is no longer picklable since Pytest 9.1, is
  serialized with a dedicated recipe so it remains cached and all
  subsessions keep sharing one base temporary directory. Session-scoped
  fixtures whose result still cannot be pickled are skipped by the fixture
  cache and re-evaluated in each subsession, instead of crashing the run.
  The cache file is only created once the fixture result has been
  serialized successfully, so a failed write can no longer leave a
  truncated file behind that would make subsequent subsessions fail with
  ``EOFError``. (`#36`_)

- Python 3.10 or newer and Pytest 7.3 or newer are now required, matching
  the versions covered by the CI test matrix.

- The ``mpi`` marker now accepts an optional ``threads`` argument.
  When set, ``OMP_NUM_THREADS`` is configured for each isolated MPI process.

- Two command line options for the use of independent Python executables
  and Pytest configurations in the main session and subsessions have
  been added. This enhances the ability for use with e.g. Apptainer,
  allowing a main session outside of a container to spawn containerized
  subsessions. (`#33`_)

.. _#33:  https://github.com/dlr-sp/pytest-isolate-mpi/pull/33
.. _#36:  https://github.com/dlr-sp/pytest-isolate-mpi/pull/36
.. _#38:  https://github.com/dlr-sp/pytest-isolate-mpi/pull/38
.. _#39:  https://github.com/dlr-sp/pytest-isolate-mpi/pull/39

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

