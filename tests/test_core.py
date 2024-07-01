import pytest
import numpy as np


from pytest_isolate_mpi import core


@pytest.mark.parametrize(['array', 'expected_result'], [
    (np.zeros((1, 1)), 0),
    (np.arange(5), 10),
    (np.arange(9).reshape(3, 3), 36),
])
def test_sum(array, expected_result):
    assert core.sum_array(array) == expected_result
