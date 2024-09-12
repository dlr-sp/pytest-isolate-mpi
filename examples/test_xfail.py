import pytest


@pytest.mark.mpi(ranks=2)
@pytest.mark.xfail
def test_xfail(mpi_ranks):  # pylint: disable=unused-argument
    """Simple xfailing test."""
    assert False
