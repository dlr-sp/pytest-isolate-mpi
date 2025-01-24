MPI-Parallel, Crash-Safe Tests for Pytest
=========================================

``pytest-isolate-mpi`` is a plugin for `Pytest`_ to enable the test of
MPI-parallel Python software. MPI-parallel tests are executed in forked
MPI jobs with the help of `mpirun`_ to isolate the tests from each
other. Thus, a crash in test only aborts the test, not the whole test
suite. Similarly, deadlocks can be treated with timeouts for tests to
prevent a test suite from never being able to finish::

    import pytest
    
    
    @pytest.mark.mpi(ranks=2, timeout=10, unit="s")
    def test_with_mpi(mpi_ranks):
        assert True  # replace with actual, MPI-parallel test code


.. _pytest: https://docs.pytest.org/en/stable/
.. _mpirun: https://docs.open-mpi.org/en/v5.0.x/man-openmpi/man1/mpirun.1.html

Installation
------------

``pytest-isolate-mpi`` is available on `pypi.org`_ and can be installed
with ``pip``::

    pip install pytest-mpi-isolate


.. _pypi.org: https://pypi.org/project/pytest-isolate-mpi/

Documentation
-------------

For the full documentation, please see
https://pytest-isolate-mpi.readthedocs.io/.

Contributing
------------

Please refer to the `Contributor Guide`_ for
instructions on how to contribute to ``pytest-isolate-mpi``.

.. _Contributor Guide: https://pytest-isolate-mpi.readthedocs.io/en/latest/contributing.html

License
-------

This work is licensed under the conditions of the BSD-3-Clause license,
see `LICENSE <LICENSE>`_.

The software is provided as is.  We sincerely welcome your feedback on
issues, bugs and possible improvements.  Please use the issue tracker of
the project for the corresponding communication or make a fork.  Our
priority and time line for working on the issues depend on the project
and its follow ups.  This may lead to issue and tickets, which are not
pursued.  In case you need an urgent fix, please contact us directly for
discussing possible forms of collaboration (direct contribution,
projects, contracting, ...): `Institute of Software Methods for Product
Virtualization <https://www.dlr.de/sp>`_.


