import os

import pytest


@pytest.mark.mpi(ranks=2)
def test_with_mpi(mpi_ranks):  # pylint: disable=unused-argument
    """This test checks whether command line arguments to the mpi executable, as given in the ini-file,
    are passed correctly"""
    assert os.getenv("foo") == "bar"
