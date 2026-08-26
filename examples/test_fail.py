import pytest


@pytest.mark.mpi(ranks=2)
def test_fail():
    """Simple failing test."""
    assert False
