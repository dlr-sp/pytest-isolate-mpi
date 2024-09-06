import os
import time
from pathlib import Path
import random

import pytest

from pytest_isolate_mpi._constants import ENVIRONMENT_VARIABLE_TO_HIDE_INNARDS_OF_PLUGIN


@pytest.mark.mpi(ranks=2)
def test_pass(mpi_ranks):  # pylint: disable=unused-argument
    """Simple passing test."""
    assert True


@pytest.mark.mpi(ranks=2)
def test_fail(mpi_ranks):  # pylint: disable=unused-argument
    """Simple failing test."""
    assert False


@pytest.mark.mpi(ranks=2)
@pytest.mark.xfail
def test_xfail(mpi_ranks):  # pylint: disable=unused-argument
    """Simple xfailing test."""
    assert False


@pytest.mark.mpi(ranks=2)  # pylint: disable=unused-argument
def test_one_failing_rank(mpi_ranks, comm):  # pylint: disable=unused-argument
    assert comm.rank != 0


@pytest.mark.mpi(ranks=2)
def test_one_aborting_rank(mpi_ranks, comm):  # pylint: disable=unused-argument
    if comm.rank == 0:
        os._exit(127)


@pytest.mark.mpi(ranks=[1, 2, 3])
def test_number_of_processes_matches_ranks(mpi_ranks, comm):
    """Simple test that checks whether we run on multiple processes."""
    assert comm.size == mpi_ranks


@pytest.mark.mpi(ranks=2, timeout=5, unit="s")
def test_timeout(mpi_ranks, comm):  # pylint: disable=unused-argument
    rank = comm.rank
    for _ in range(10):
        print(f"Timeout: sleeping (1) on rank `{rank}`")
        time.sleep(1)


@pytest.mark.mpi(ranks=2, timeout=10, unit="s")
def test_mpi_deadlock(mpi_ranks, comm):  # pylint: disable=unused-argument
    if comm.rank == 0:
        comm.Barrier()


@pytest.mark.mpi(ranks=[1, 2, 3])
@pytest.mark.skip(reason="Testing whether skipping works")
def test_skip(mpi_ranks):  # pylint: disable=unused-argument
    """This test checks whether skipping the tests is used correctly.
    The skip should be computed on the process running pytest and not lead to anything being done in a forked
    environment."""
    # This will always fail in case of us actually executing the test, such that we can test whether it works
    assert False


@pytest.mark.mpi(ranks=2)
def test_mpi_tmp_path(mpi_ranks, mpi_tmp_path):  # pylint: disable=unused-argument
    assert isinstance(mpi_tmp_path, Path) and mpi_tmp_path.exists()


def test_no_mpi():
    """Simple test checking non-isolated non-MPI test remain possible."""
    assert ENVIRONMENT_VARIABLE_TO_HIDE_INNARDS_OF_PLUGIN not in os.environ


@pytest.fixture(scope='session', params=['a', 'b'], ids=['A', 'B'])
def first_fixture(request):
    return request.param


@pytest.fixture(scope='session', params=['x', 'y'], ids=['X', 'Y'])
def second_fixture(first_fixture, request):
    return first_fixture, request.param


@pytest.fixture(params=['u', 'v'], ids=['U', 'V'])
def third_fixture(request):
    return request.param


@pytest.fixture(scope='session')
def expensive_fixture(second_fixture, comm):
    time.sleep(4)
    val = random.random()
    print(f'expensive fixture in rank {comm.rank} of size {comm.size} with parameter {second_fixture} calculated {val}')
    return val


@pytest.mark.mpi(ranks=[1, 2])
def test_cache_first(mpi_ranks, expensive_fixture):  # pylint: disable=unused-argument
    # This test calls the expensive fixture first.
    print(f"got {expensive_fixture}")


@pytest.mark.mpi(ranks=[1, 2], timeout=2, unit="s")
def test_cache_second(mpi_ranks, expensive_fixture, third_fixture):  # pylint: disable=unused-argument
    # This test calls the expensive fixture second and uses the cache, avoiding timeout.
    print(f"got {expensive_fixture} and {third_fixture}")
