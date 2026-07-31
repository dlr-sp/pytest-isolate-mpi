# pylint: disable=unused-argument,redefined-outer-name
import threading

import pytest


@pytest.fixture(scope="session")
def unpicklable():
    return threading.Lock()  # locks cannot be pickled


@pytest.mark.mpi(ranks=2)
def test_unpicklable_fixture_is_not_cached(mpi_ranks, comm, unpicklable):
    # First subsession: evaluates the fixture, the plugin skips caching it.
    assert unpicklable is not None


@pytest.mark.mpi(ranks=2)
def test_unpicklable_fixture_is_reevaluated(mpi_ranks, comm, unpicklable):
    # Later subsession: finds no cache file and evaluates the fixture again.
    assert unpicklable is not None
