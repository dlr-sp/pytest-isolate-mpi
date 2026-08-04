import pytest

# Each test below runs in its own subsession. The first subsession stores the
# session-scoped ``tmp_path_factory`` in the fixture cache; the second only
# sees the same base temporary directory if the factory was restored from the
# cache instead of being evaluated anew.


@pytest.mark.mpi(ranks=2)
def test_tmp_path_factory_is_cached(mpi_ranks, comm, tmp_path_factory):  # pylint: disable=unused-argument
    marker = tmp_path_factory.getbasetemp() / f"marker-rank-{comm.rank}"
    marker.touch()
    assert marker.exists()


@pytest.mark.mpi(ranks=2)
def test_tmp_path_factory_is_shared(mpi_ranks, comm, tmp_path_factory):  # pylint: disable=unused-argument
    assert (tmp_path_factory.getbasetemp() / f"marker-rank-{comm.rank}").exists()
