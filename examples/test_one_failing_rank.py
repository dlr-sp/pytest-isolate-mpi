import pytest


@pytest.mark.mpi(ranks=2)
def test_one_failing_rank(mpi_ranks, comm):  # pylint: disable=unused-argument
    """In case of just one process failing an assert, the test counts
    as failed and the outputs are gathered from the processes."""
    assert comm.rank != 0
