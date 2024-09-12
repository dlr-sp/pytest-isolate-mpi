import pytest


@pytest.mark.mpi(ranks=2)
def test_with_mpi(mpi_ranks):  # pylint: disable=unused-argument
    """Simple passing test"""
    assert True  # replace with actual test code
