import threading

import pytest


@pytest.fixture(scope="session")
def unpicklable():
    return threading.Lock()


@pytest.mark.mpi(ranks=2)
def test_uses_unpicklable_first(mpi_ranks, comm, unpicklable):  # pylint: disable=unused-argument,redefined-outer-name
    assert unpicklable is not None


@pytest.mark.mpi(ranks=2)
def test_uses_unpicklable_second(mpi_ranks, comm, unpicklable):  # pylint: disable=unused-argument,redefined-outer-name
    # Runs in a separate subsession and would load the cached fixture result.
    assert unpicklable is not None
