import threading

import pytest


@pytest.fixture(scope="session")
def unpicklable():
    return threading.Lock()  # locks cannot be pickled


# The two tests below share one cache file, since its path is keyed by the fixture name
# and the size/rank combination rather than by the test. Each runs in its own subsession,
# so in order they cover the write side of the cache and then the read side.


@pytest.mark.mpi(ranks=2)
@pytest.mark.usefixtures("unpicklable")
def test_unpicklable_fixture_is_not_cached(mpi_ranks, comm):
    # Write side: the fixture is evaluated, pickling its result fails, and the plugin
    # skips caching instead of failing the run.
    assert comm.size == mpi_ranks


@pytest.mark.mpi(ranks=2)
@pytest.mark.usefixtures("unpicklable")
def test_unpicklable_fixture_is_reevaluated(mpi_ranks, comm):
    # Read side, one subsession later: with no cache file to load, the fixture is simply
    # evaluated again. Fails with an EOFError if the skipped write left an empty file.
    assert comm.size == mpi_ranks
