from mpi4py import MPI
import pytest
import pytest_isolate_mpi
import time


@pytest.mark.mpi(ranks=[1, 2, 3])
def test_sum(mpi_ranks):
    assert False


@pytest.mark.mpi(ranks=[1, 2, 3])
def test_true(mpi_ranks):
    assert True

@pytest.mark.mpi(ranks=[1, 2, 3])
def test_abort(mpi_ranks):
    time.sleep(1)
    rank = MPI.COMM_WORLD.Get_rank()
    assert rank != 1
    for _ in range(3):
        print(f"Sleep(1) on rank `{rank}`")
        time.sleep(1)
