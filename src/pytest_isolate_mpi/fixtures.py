"""MPI-specific fixtures."""

from pathlib import Path

import py
import pytest


@pytest.fixture(scope="session", name="comm")
def comm_fixture() -> "mpi4py.MPI.Comm":
    """Provides the MPI communicator available to the test."""
    try:
        from mpi4py import MPI  # pylint: disable=import-outside-toplevel
    except ImportError:
        pytest.fail("mpi4py needs to be installed to run this test")
    return MPI.COMM_WORLD


@pytest.fixture(name="mpi_tmpdir")
def mpi_tmpdir_fixture(tmpdir, comm) -> py.path.local:
    """
    Wraps Pytest builtin ``tmpdir`` fixture so that it can be used under
    MPI from all MPI processes.

    This fixture ensures that only one process handles the creation of temporary
    folders broadcasts its path to the other processes.
    """
    # we only want to put the file inside one tmpdir, this creates the name
    # under one process, and passes it on to the others
    name = str(tmpdir) if comm.rank == 0 else None
    name = comm.bcast(name, root=0)
    return py.path.local(name)


@pytest.fixture(name="mpi_tmp_path")
def mpi_tmp_path_fixture(tmp_path, comm) -> Path:
    """
    Wraps Pytest builtin ``tmp_path`` fixture so that it can be used under
    MPI from all MPI processes.

    This fixture ensures that only one process handles the creation of temporary
    folders broadcasts its path to the other processes.
    """
    # we only want to put the file inside one tmpdir, this creates the name
    # under one process, and passes it on to the others
    name = str(tmp_path) if comm.rank == 0 else None
    name = comm.bcast(name, root=0)
    return Path(name)
