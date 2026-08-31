import os

import pytest


@pytest.mark.mpi(ranks=2)
def test_one_aborting_rank(comm):
    """In case of one process aborting, MPI_Finalize is not called.
    This is handled by pytest-isolate-mpi and counts as a failed test."""
    if comm.rank == 0:
        os._exit(127)  # this bypasses the Python shutdown process
