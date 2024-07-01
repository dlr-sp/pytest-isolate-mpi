import pytest

import pytest_isolate_mpi

@pytest.mark.mpi(procs=3)
def test_sum():
    assert False

