=====
Usage
=====

Writing MPI-Parallel Tests
--------------------------

To create a MPI-parallel test, its test function must be marked with the
``mpi`` mark:

.. literalinclude:: ../examples/test_basic.py
    :linenos:

The number of MPI processes to be used for the test must be set via the
required ``ranks`` argument. All MPI tests need to have an ``mpi_ranks``
parameter as shown in the example.

For any test carrying the ``mpi`` mark, ``pytest-isolate-mpi`` will
launch an MPI job with the requested amount of processes. In this MPI
job,  a ``pytest`` session runs this particular tests. Each MPI process
produces its own test report which is collected in the main process. To
distinguish the reports form each MPI process, ``pytest-isolate-mpi``
extends the node IDs of the test reports to contain the source rank
where the report is originating from. For instance the test above would
result in (with ``--verbose`` passed to ``pytest``):

.. literalinclude:: ../examples/test_basic.py.out
    :linenos:
    :language: output
    
By having an dedicated report for each MPI process, failing ranks can be
easily identified:

.. literalinclude:: ../examples/test_one_failing_rank.py
    :linenos:

This test will always fail an MPI process 0:

.. literalinclude:: ../examples/test_one_failing_rank.py.out
    :linenos:
    :language: output

All tests not marked with the ``mpi`` mark are executed as usual in the
main ``pytest`` session.

Parametrizing the Number of MPI Processes
-----------------------------------------

By passing a list to ``ranks`` argument to the ``mpi`` mark, a test is
run multiple times with each requested number of MPI processes in turn

.. literalinclude:: ../examples/test_number_of_processes_matches_ranks.py
    :linenos:

Here, for each parametrization a matching number of test reports is
produced:

.. literalinclude:: ../examples/test_number_of_processes_matches_ranks.py.out
    :language: output

Enforcing a Maximum Runtime for MPI Tests
-----------------------------------------

``pytest-isolate-mpi`` allows to set a maximum runtime for MPI-parallel
tests with the ``timeout`` argument of the ``mpi`` mark:

.. literalinclude:: ../examples/test_mpi_deadlock.py
    :linenos:

``timeout`` sets maximum permissible runtime before the test is
forcefully terminated. With the optional ``unit`` argument one can set
the time unit for the duration. Supported are ``"s"`` for seconds,
``"m"`` for minutes and ``h`` for hours. If not specified explicitly,
the default unit is seconds.

By setting an timeout for an MPI-parallel test, deadlocks in this test
will no longer prevent the completion of the test suite:

.. literalinclude:: ../examples/test_mpi_deadlock.py.out
    :language: output

MPI Fixtures
------------

``pytest-isolate-mpi`` offers a selection of fixtures for the
development of MPI-parallel tests:

comm
    The MPI communicator available for the MPI-parallel test, i.g. 
    :obj:`mpi4py.MPI.COMM_WORLD`.

    See also :func:`pytest_isolate_mpi.fixtures.comm_fixture`.
    

mpi_tmpdir
    Wraps Pytest builtin ``tmpdir`` fixture so that it can be used under
    MPI from all MPI processes.

    See also :func:`pytest_isolate_mpi.fixtures.mpi_tmpdir_fixture`.

mpi_tmp_path
    Wraps Pytest builtin ``tmp_path`` fixture so that it can be used under
    MPI from all MPI processes.

    See also :func:`pytest_isolate_mpi.fixtures.mpi_tmp_path_fixture`.


Limitations
-----------

Reports for Crashed MPI Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If a Pytest session running a single MPI-parallel test exits
prematurely, it may fail to write its test report to its predetermined
location. In this case, ``pytest-isolate-mpi`` can no longer provide a
per-process test report for the failed ranks. Instead,
``pytest-isolate-mpi`` will instead produce the output of ``mpirun``
which will contain the full output of all parallel-run Pytest sessions
and ``mpirun`` itself:

.. literalinclude:: ../examples/test_one_aborting_rank.py.out
    :language: output

Fixture Scopes
~~~~~~~~~~~~~~

Pytest allows to reuse fixtures between tests with the help of `fixture
scopes`_. Since ``pytest-isolate-mpi`` executes each MPI-parallel test
in a Pytest sub session, support for session scopes other than the
default ``function`` scope is limited for MPI-parallel tests:

.. _fixture scopes: https://docs.pytest.org/en/stable/how-to/fixtures.html#fixture-scopes

* ``session``: ``pytest-isolate-mpi`` will store the result of
  session-scoped fixture functions in a cache file. This file will be
  read back then the fixture is requested by subsequent tests. The file
  is managed per MPI communicator size and rank so each MPI process
  caches its own dedicated fixture. Sharing fixtures between tests of
  differently sized communicators and non-MPI/MPI tests is not possible.
  Fixtures are serialized with the :mod:`pickle` module. Beware of that
  not all Python objects support pickling.

* ``class``, ``module``, and ``package``: Fixtures for these scopes are
  re-created for each MPI-parallel tests. Such fixtures effectively
  behave as they were function-scoped.

For non-MPI tests, fixture scopes behave as usual even if
``pytest-isolate-mpi`` is employed in the project.


Percentage of Completed Tests During Pytest Run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As ``pytest-isolate-mpi`` produces one test protocol per MPI-process
while not increasing the test count, the reported percentages for test
run completion are incorrect.


Troubleshooting
---------------

Test Collection Fails with ``function uses no argument 'mpi_ranks'``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pytest-isolate-mpi`` `parametrizes`_ all MPI tests with regards to the
chosen number of MPI processes. As such, all test marked using the
``pytest.mark.mpi()`` marker must accept the argument ``mpi_ranks``,
even if the test makes no use of this information::


    @pytest.mark.mpi(ranks=2)
    def test_pass(mpi_ranks):  # Argument required
        assert True



If at least one MPI test misses this argument, the test collection fails.

.. _parametrizes: https://docs.pytest.org/en/stable/how-to/parametrize.html#pytest-mark-parametrize-parametrizing-test-functions


