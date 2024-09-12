import pytest


@pytest.mark.mpi(ranks=[1, 2, 3])
@pytest.mark.skip(reason="Testing whether skipping works")
def test_skip(mpi_ranks):  # pylint: disable=unused-argument
    """This test checks whether skipping the tests is used correctly.
    The skip should be computed on the process running pytest and not lead to anything being done in a forked
    environment."""
    # This will always fail in case of us actually executing the test, such that we can test whether it works
    assert False
