import pytest


@pytest.mark.parametrize(
    ["test", "outcomes", "lines"],
    [
        pytest.param("test_pass", {"passed": 2}, [], id="test_pass"),
        pytest.param(
            "test_fail", {"failed": 2}, [rf"FAILED .*test_fail\[2\]\[rank={i}\].*" for i in range(2)], id="test_fail"
        ),
        pytest.param("test_xfail", {"xfailed": 2}, [], id="test_xfail"),
        pytest.param(
            "test_one_failing_rank",
            {"passed": 1, "failed": 1},
            [r"FAILED .*test_one_failing_rank\[2\]\[rank=0\].*"],
            id="test_one_failing_rank",
        ),
        pytest.param("test_one_aborting_rank", {"passed": 1, "failed": 1}, [], id="test_one_aborting_rank"),
        pytest.param(
            "test_number_of_processes_matches_ranks", {"passed": 6}, [], id="test_number_of_processes_matches_ranks"
        ),
        pytest.param(
            "test_timeout",
            {"failed": 1},
            [r"Timeout occurred for test_mpi.py::test_timeout\[2\]: exceeded run time limit of 5s\."],
            id="test_timeout",
        ),
        pytest.param(
            "test_mpi_deadlock",
            {"failed": 1, "passed": 1},
            [r"Timeout occurred for test_mpi.py::test_mpi_deadlock\[2\]: exceeded run time limit of 10s\."],
            id="test_mpi_deadlock",
        ),
        pytest.param("test_skip", {"skipped": 6}, [], id="test_skip"),
        pytest.param("test_mpi_tmp_path", {"passed": 2}, [], id="test_mpi_tmp_path"),
        pytest.param("test_no_mpi", {"passed": 1}, [], id="test_no_mpi"),
        pytest.param("test_cache", {"passed": 24}, [], id="test_cache"),
    ],
)
def test_outcomes(pytester, test, outcomes, lines):
    pytester.copy_example("test_mpi.py")
    result = pytester.runpytest("-v", "-rA", "-k", test)
    result.assert_outcomes(**outcomes)
    if lines:
        result.stdout.re_match_lines(lines, consecutive=True)
