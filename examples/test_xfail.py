import pytest


@pytest.mark.mpi(ranks=2)
@pytest.mark.xfail
def test_xfail():
    """Simple xfailing test."""
    assert False
