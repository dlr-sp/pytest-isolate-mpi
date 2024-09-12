import time

import pytest


@pytest.mark.mpi(ranks=2, timeout=5, unit="s")
def test_timeout(mpi_ranks, comm):  # pylint: disable=unused-argument
    rank = comm.rank
    # we sleep 10 times to be larger than the timeout set for this test
    for _ in range(10):
        print(f"Timeout: sleeping (1) on rank `{rank}`")
        time.sleep(1)
