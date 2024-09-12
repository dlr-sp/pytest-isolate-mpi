import pytest


@pytest.mark.mpi(ranks=2)
def test_fail(mpi_ranks):  # pylint: disable=unused-argument
    """Simple failing test."""
    assert False
