import pytest


@pytest.mark.parametrize(
    ["test", "outcomes", "lines"],
    [
        pytest.param("test_basic", {"passed": 2}, [], id="test_basic"),
        pytest.param(
            "test_fail",
            {"failed": 2},
            [r"FAILED .*test_fail\[2-rank0\].*", r"FAILED .*test_fail\[2-rank1\].*"],
            id="test_fail",
        ),
        pytest.param("test_xfail", {"xfailed": 2}, [], id="test_xfail"),
        pytest.param(
            "test_one_failing_rank",
            {"passed": 1, "failed": 1},
            [r"FAILED .*test_one_failing_rank\[2-rank0\].*"],
            id="test_one_failing_rank",
        ),
        pytest.param("test_one_aborting_rank", {"passed": 1, "failed": 1}, [], id="test_one_aborting_rank"),
        pytest.param(
            "test_number_of_processes_matches_ranks", {"passed": 6}, [], id="test_number_of_processes_matches_ranks"
        ),
        pytest.param(
            "test_timeout",
            {"failed": 2},
            [
                r"Timeout occurred for test_timeout.py::test_timeout\[2-rank0\]: exceeded run time limit of 5s\.",
                r"_+ test_timeout\[2-rank1\] _+",
                r"Timeout occurred for test_timeout.py::test_timeout\[2-rank1\]: exceeded run time limit of 5s\.",
            ],
            id="test_timeout",
        ),
        pytest.param(
            "test_mpi_deadlock",
            {"failed": 1, "passed": 1},
            [
                r"Timeout occurred for test_mpi_deadlock.py::test_mpi_deadlock\[2-rank0\]: "
                r"exceeded run time limit of 10s\."
            ],
            id="test_mpi_deadlock",
        ),
        pytest.param("test_skip", {"skipped": 6}, [], id="test_skip"),
        pytest.param("test_mpi_tmp_path", {"passed": 2}, [], id="test_mpi_tmp_path"),
        pytest.param("test_no_mpi", {"passed": 1}, [], id="test_no_mpi"),
        pytest.param("test_session_scoped_fixtures", {"passed": 36}, [], id="test_cache"),
        pytest.param("test_omp_num_threads", {"passed": 3}, [], id="test_omp_num_threads"),
        pytest.param("test_unpicklable_session_fixture", {"passed": 4}, [], id="test_unpicklable_session_fixture"),
        pytest.param("test_tmp_path_factory_cached", {"passed": 4}, [], id="test_tmp_path_factory_cached"),
        pytest.param(
            "test_mpi_parametrization_without_fixture", {"passed": 5}, [], id="test_mpi_parametrization_without_fixture"
        ),
    ],
)
def test_outcomes(pytester, test, outcomes, lines):
    pytester.copy_example(f"{test}.py")
    result = pytester.runpytest("-v", "-rA")
    result.assert_outcomes(**outcomes)
    if lines:
        result.stdout.re_match_lines(lines, consecutive=True)


@pytest.mark.parametrize(
    ["test", "outcomes", "lines"],
    [
        pytest.param("test_basic", {"passed": 1}, [], id="test_basic"),
        pytest.param("test_fail", {"failed": 1}, [r"FAILED .*test_fail\[2\].*"], id="test_fail"),
        pytest.param("test_xfail", {"xfailed": 1}, [], id="test_xfail"),
        pytest.param(
            "test_one_failing_rank",
            {"passed": 0, "failed": 1},
            [r"FAILED .*test_one_failing_rank\[2\].*"],
            id="test_one_failing_rank",
        ),
        pytest.param(
            "test_number_of_processes_matches_ranks",
            {"passed": 1, "failed": 2},
            [],
            id="test_number_of_processes_matches_ranks",
        ),
        pytest.param("test_skip", {"skipped": 3}, [], id="test_skip"),
        pytest.param("test_mpi_tmp_path", {"passed": 1}, [], id="test_mpi_tmp_path"),
        pytest.param("test_no_mpi", {"passed": 1}, [], id="test_no_mpi"),
        pytest.param("test_session_scoped_fixtures", {"passed": 8, "failed": 16}, [], id="test_cache"),
        pytest.param(
            "test_mpi_parametrization_without_fixture", {"failed": 2}, [], id="test_mpi_parametrization_without_fixture"
        ),
    ],
)
def test_outcomes_no_isolation(pytester, test, outcomes, lines):
    pytester.copy_example(f"{test}.py")
    result = pytester.runpytest("-v", "-rA", "--no-mpi-isolation")
    result.assert_outcomes(**outcomes)
    if lines:
        result.stdout.re_match_lines(lines, consecutive=True)
