import pytest


@pytest.mark.mpi(procs=3)
def test_sum():
    assert True
