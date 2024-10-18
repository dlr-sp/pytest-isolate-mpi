Changelog
=========

Version 0.2
-----------

- An unhandled edge case when using a session-scoped fixture in
  non-parametrized tests was fixed. (`#14`_)

- Session-scoped fixtures are now only cached within the MPI-parallel
  Pytest sub sessions. This allows the use of fixtures which cannot be
  pickled for non-MPI tests.

- Most of Pytest's CLI options are now passed the MPI-parallel
  sub sessions.

.. _#14:  https://github.com/dlr-sp/pytest-isolate-mpi/pull/14

Version 0.1
-----------

- Initial release.

