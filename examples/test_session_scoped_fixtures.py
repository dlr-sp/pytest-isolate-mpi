import pytest

from pytest_isolate_mpi._helpers import ExpensiveComputation


@pytest.fixture(scope="session", name="first", params=["a", "b"], ids=["A", "B"])
def first_fixture(request):
    return request.param


@pytest.fixture(scope="session", name="second", params=["x", "y"], ids=["X", "Y"])
def second_fixture(first, request):
    return first, request.param


@pytest.fixture(name="third", params=["u", "v"], ids=["U", "V"])
def third_fixture(request):
    return request.param


@pytest.fixture(scope="session", name="computation")
def expensive_fixture(second, comm):
    computation = ExpensiveComputation(comm)
    print(
        f"expensive fixture in rank {comm.rank} of size {comm.size} with parameter {second} "
        f"calculated {computation.value}"
    )
    return computation


@pytest.mark.mpi(ranks=[1, 2])
def test_cache_first(mpi_ranks, comm, computation):  # pylint: disable=unused-argument
    # This test calls the expensive fixture first.
    assert computation.was_cached is False
    assert computation.computed_in_rank_of_size == (comm.rank, comm.size)
    print(f"got {computation.value}")


@pytest.mark.mpi(ranks=[1, 2])
def test_cache_second(mpi_ranks, comm, computation, third):  # pylint: disable=unused-argument
    # This test uses the cache.
    assert computation.was_cached is True
    assert computation.computed_in_rank_of_size == (comm.rank, comm.size)
    print(f"got {computation.value} and {third}")
