import threading

import pytest


@pytest.fixture(scope="session")
def unpicklable():
    return threading.Lock()  # locks cannot be pickled


@pytest.mark.mpi(ranks=2)
@pytest.mark.usefixtures("unpicklable")
def test_unpicklable_fixture_is_not_cached(mpi_ranks, comm):
    # First subsession: evaluates the fixture, the plugin skips caching it.
    assert comm.size == mpi_ranks


@pytest.mark.mpi(ranks=2)
@pytest.mark.usefixtures("unpicklable")
def test_unpicklable_fixture_is_reevaluated(mpi_ranks, comm):
    # Later subsession: finds no cache file and evaluates the fixture again.
    assert comm.size == mpi_ranks
