# pylint: disable=unused-argument,redefined-outer-name
"""Session fixtures whose values cannot be pickled are not cached across
subsessions but re-evaluated per test, like ``comm``.

Each MPI test runs in its own forked subsession, so this scenario inherently
needs two tests: the first exercises the cache write path (the plugin must
skip caching the unpicklable value instead of crashing), the second exercises
the cache read path in a later subsession (no cache file, so the fixture is
evaluated again).
"""

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
