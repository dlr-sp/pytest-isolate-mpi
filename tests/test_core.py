import pytest
import pytest_isolate_mpi


@pytest.mark.mpi(ranks=[1, 2, 3])
def test_sum(mpi_ranks):
    assert False


@pytest.mark.mpi(ranks=[1, 2, 3])
def test_true(mpi_ranks):
    assert True
