from pathlib import Path

import pytest


@pytest.mark.mpi(ranks=2)
def test_mpi_tmp_path(mpi_tmp_path):
    assert isinstance(mpi_tmp_path, Path) and mpi_tmp_path.exists()
