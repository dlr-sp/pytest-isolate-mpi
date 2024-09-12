import pytest


@pytest.mark.mpi(ranks=[1, 2, 3])
def test_number_of_processes_matches_ranks(mpi_ranks, comm):
    """Simple test that checks whether we run on multiple processes."""
    assert comm.size == mpi_ranks
