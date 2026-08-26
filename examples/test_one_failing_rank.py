import pytest


@pytest.mark.mpi(ranks=2)
def test_one_failing_rank(comm):
    """In case of just one process failing an assert, the test counts
    as failed and the outputs are gathered from the processes."""
    assert comm.rank != 0
