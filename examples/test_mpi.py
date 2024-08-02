import os
from pathlib import Path


import pytest
import time


from pytest_isolate_mpi._constants import ENVIRONMENT_VARIABLE_TO_HIDE_INNARDS_OF_PLUGIN


@pytest.mark.mpi(ranks=2)
def test_pass(mpi_ranks):
    """Simple passing test."""
    assert True


@pytest.mark.mpi(ranks=2)
def test_fail(mpi_ranks):
    """Simple failing test."""
    assert False


@pytest.mark.mpi(ranks=2)
@pytest.mark.xfail
def test_xfail(mpi_ranks):
    """Simple xfailing test."""
    assert False


@pytest.mark.mpi(ranks=2)
def test_one_failing_rank(mpi_ranks, comm):
    assert comm.rank != 0


@pytest.mark.mpi(ranks=2)
def test_one_aborting_rank(mpi_ranks, comm):
    if comm.rank == 0:
        os._exit(127)


@pytest.mark.mpi(ranks=[1, 2, 3])
def test_number_of_processes_matches_ranks(mpi_ranks, comm):
    """Simple test that checks whether we run on multiple processes."""
    assert comm.size == mpi_ranks


@pytest.mark.mpi(ranks=2)
@pytest.mark.mpi_timeout(timeout=5, unit='s')
def test_timeout(mpi_ranks, comm):
    rank = comm.rank
    for _ in range(10):
        print(f"Timeout: sleeping (1) on rank `{rank}`")
        time.sleep(1)


@pytest.mark.mpi(ranks=2)
@pytest.mark.mpi_timeout(timeout=10, unit='s')
def test_mpi_deadlock(mpi_ranks, comm):
    if comm.rank == 0:
        comm.Barrier()


@pytest.mark.mpi(ranks=[1, 2, 3])
@pytest.mark.skip(reason="Testing whether skipping works")
def test_skip(mpi_ranks):
    """This test checks whether skipping the tests is used correctly.
    The skip should be computed on the process running pytest and not lead to anything being done in a forked
    environment."""
    # This will always fail in case of us actually executing the test, such that we can test whether it works
    assert False


@pytest.mark.mpi(ranks=2)
def test_mpi_tmp_path(mpi_ranks, mpi_tmp_path):
    assert isinstance(mpi_tmp_path, Path) and mpi_tmp_path.exists()


def test_no_mpi():
    """Simple test checking non-isolated non-MPI test remain possible."""
    assert ENVIRONMENT_VARIABLE_TO_HIDE_INNARDS_OF_PLUGIN not in os.environ
