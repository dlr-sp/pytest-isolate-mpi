from __future__ import annotations

import enum

import pytest


@enum.unique
class MPIMarkerEnum(str, enum.Enum):
    """
    Enum containing all the markers used by pytest-mpi

    FIXME: Once we are on Python 3.11, use StrEnum
    """

    mpi = "mpi"
    mpi_skip = "mpi_skip"
    mpi_xfail = "mpi_xfail"
    mpi_break = "mpi_break"
    mpi_timeout = "mpi_timeout"


VERBOSE_MPI_ARG = "--verbose-mpi"
IS_FORKED_MPI_ARG = "--is-forked-by-main-pytest"
ENVIRONMENT_VARIABLE_TO_HIDE_INNARDS_OF_PLUGIN = "PYTEST_ISOLATE_MPI_IS_FORKED"
TIME_UNIT_CONVERSION = {
    's': lambda timeout: timeout,
    'm': lambda timeout: timeout * 60,
    'h': lambda timeout: timeout * 3600,
}

MPI_ENV_HINTS = [
    "OMPI_COMM_WORLD_SIZE",
    "MV2_COMM_WORLD_RANK",
    "PMI_RANK",
    "ALPS_APP_PE",
    "PMIX_RANK",
    "PALS_NODEID",
]

MPI_MARKERS = {
    MPIMarkerEnum.mpi_skip: pytest.mark.skip(
        reason="test does not work under mpi"
    ),
    MPIMarkerEnum.mpi_break: pytest.mark.skip(
        reason="test does not work under mpi"
    ),
    MPIMarkerEnum.mpi_xfail: pytest.mark.xfail(
        reason="test fails under mpi"
    ),
}
