import pytest


@pytest.mark.mpi(ranks=2)
def test_with_mpi():
    """Simple passing test"""
    assert True  # replace with actual test code
