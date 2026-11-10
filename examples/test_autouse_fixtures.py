import pytest


@pytest.mark.mpi(ranks=[1, 2, 3])
def test_autouse_fixtures(mpi_rank, mpi_ranks, comm):
    """Check that the `mpi_rank` and `mpi_ranks` fixtures match the actual MPI communicator."""
    assert comm.rank == mpi_rank and comm.size == mpi_ranks
