"""MPI-specific fixtures."""

from pathlib import Path

import py
import pytest


@pytest.fixture(name="comm")
def comm_fixture():
    try:
        from mpi4py import MPI  # pylint: disable=import-outside-toplevel
    except ImportError:
        pytest.fail("mpi4py needs to be installed to run this test")
    return MPI.COMM_WORLD


@pytest.fixture(name="mpi_file_name")
def mpi_file_name_fixture(tmpdir, request):
    """
    Provides a temporary file name which can be used under MPI from all MPI
    processes.

    This function avoids the need to ensure that only one process handles the
    naming of temporary files.
    """
    try:
        from mpi4py import MPI  # pylint: disable=import-outside-toplevel
    except ImportError:
        pytest.fail("mpi4py needs to be installed to run this test")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # we only want to put the file inside one tmpdir, this creates the name
    # under one process, and passes it on to the others
    name = str(tmpdir.join(str(request.node) + ".hdf5")) if rank == 0 else None
    name = comm.bcast(name, root=0)
    return name


@pytest.fixture(name="mpi_tmpdir")
def mpi_tmpdir_fixture(tmpdir):
    """
    Wraps `pytest.tmpdir` so that it can be used under MPI from all MPI
    processes.

    This function avoids the need to ensure that only one process handles the
    naming of temporary folders.
    """
    try:
        from mpi4py import MPI  # pylint: disable=import-outside-toplevel
    except ImportError:
        pytest.fail("mpi4py needs to be installed to run this test")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # we only want to put the file inside one tmpdir, this creates the name
    # under one process, and passes it on to the others
    name = str(tmpdir) if rank == 0 else None
    name = comm.bcast(name, root=0)
    return py.path.local(name)


@pytest.fixture(name="mpi_tmp_path")
def mpi_tmp_path_fixture(tmp_path):
    """
    Wraps `pytest.tmp_path` so that it can be used under MPI from all MPI
    processes.

    This function avoids the need to ensure that only one process handles the
    naming of temporary folders.
    """
    try:
        from mpi4py import MPI  # pylint: disable=import-outside-toplevel
    except ImportError:
        pytest.fail("mpi4py needs to be installed to run this test")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # we only want to put the file inside one tmpdir, this creates the name
    # under one process, and passes it on to the others
    name = str(tmp_path) if rank == 0 else None
    name = comm.bcast(name, root=0)
    return Path(name)
