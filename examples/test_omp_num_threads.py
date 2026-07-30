import os

import pytest


@pytest.mark.mpi(ranks=2, threads=4)
def test_omp_num_threads(mpi_ranks):
    assert mpi_ranks == 2
    assert os.environ["OMP_NUM_THREADS"] == "4"


@pytest.mark.mpi(ranks=1, threads=1)
def test_single_omp_thread(mpi_ranks):
    assert mpi_ranks == 1
    assert os.environ["OMP_NUM_THREADS"] == "1"
