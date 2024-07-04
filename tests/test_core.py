from mpi4py import MPI
import pytest
import pytest_isolate_mpi
import time


@pytest.mark.mpi(ranks=[2])
def test_fail(mpi_ranks):
    """Failing test -- to check whether XFAIL works.
    TODO: use xfail and test whether it works
    """
    assert False


@pytest.mark.mpi(ranks=[1, 2, 3])
def test_number_of_processes_matches_ranks(mpi_ranks):
    """Simple test that checks whether we run on multiple processes."""
    print(mpi_ranks)
    num_ranks = MPI.COMM_WORLD.Get_size()
    assert num_ranks == mpi_ranks


@pytest.mark.mpi(ranks=[1, 3])
@pytest.mark.mpi_timeout(timeout=5, unit='s')
def test_timeout(mpi_ranks):
    rank = MPI.COMM_WORLD.Get_rank()
    for _ in range(10):
        print(f"Timeout: sleeping (1) on rank `{rank}`")
        time.sleep(1)


@pytest.mark.mpi(ranks=[2, 3])
def test_abort(mpi_ranks):
    time.sleep(1)
    rank = MPI.COMM_WORLD.Get_rank()
    assert rank != 1

    while True:
        print(f"Sleep(1) on rank `{rank}`")
        time.sleep(1)


@pytest.mark.mpi(ranks=[1, 2, 3])
@pytest.mark.skip(reason="Testing whether skipping works")
def test_skip(mpi_ranks):
    """This test checks whether skipping the tests is used correctly.
    The skip should be computed on the process running pytest and not lead to anything being done in a forked
    environment."""
    # This will always fail in case of us actually executing the test, such that we can test whether it works
    assert False
