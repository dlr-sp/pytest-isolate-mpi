import os

from pytest_isolate_mpi._constants import ENVIRONMENT_VARIABLE_TO_HIDE_INNARDS_OF_PLUGIN


def test_no_mpi():
    """Simple test checking non-isolated non-MPI test remain possible."""
    assert ENVIRONMENT_VARIABLE_TO_HIDE_INNARDS_OF_PLUGIN not in os.environ
