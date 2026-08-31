import pytest


@pytest.mark.mpi(ranks=[2, 3])
def test_with_mpi(comm):
    """Simple passing test"""
    assert comm.size in {2, 3}
