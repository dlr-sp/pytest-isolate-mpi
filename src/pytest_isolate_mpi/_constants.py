from __future__ import annotations

import enum

@enum.unique
class MPIMarkerEnum(str, enum.Enum):
    """
    Enum containing all the markers used by pytest-mpi

    FIXME: Once we are on Python 3.11, use StrEnum
    """

    MPI = "mpi"


VERBOSE_MPI_ARG = "--verbose-mpi"
IS_FORKED_MPI_ARG = "--is-forked-by-main-pytest"
ENVIRONMENT_VARIABLE_TO_HIDE_INNARDS_OF_PLUGIN = "PYTEST_ISOLATE_MPI_IS_FORKED"
TIME_UNIT_CONVERSION = {
    "s": lambda timeout: timeout,
    "m": lambda timeout: timeout * 60,
    "h": lambda timeout: timeout * 3600,
}

MPI_ENV_HINTS = [
    "OMPI_COMM_WORLD_SIZE",
    "MV2_COMM_WORLD_RANK",
    "PMI_RANK",
    "ALPS_APP_PE",
    "PMIX_RANK",
    "PALS_NODEID",
]
