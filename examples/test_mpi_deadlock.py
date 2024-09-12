import pytest


@pytest.mark.mpi(ranks=2, timeout=10, unit="s")
def test_mpi_deadlock(mpi_ranks, comm):  # pylint: disable=unused-argument
    """Only the first process enters the barrier, all others move on
    and complete the test this leads to a deadlock.  pytest-isolate-mpi
    handles this with timeouts"""
    if comm.rank == 0:
        comm.Barrier()
