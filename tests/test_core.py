import pytest


@pytest.mark.mpi(ranks=[1, 2, 3])
def test_sum(mpi_ranks):
    assert False
