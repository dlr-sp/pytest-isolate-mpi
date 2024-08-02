import pytest


@pytest.mark.parametrize(["test", "outcomes", "lines"], [
    ("test_pass", {"passed": 2}, []),
    ("test_fail", {"failed": 2}, [rf"FAILED .*test_fail\[2\]\[rank={i}\].*" for i in range(2)]),
    ("test_xfail", {"xfailed": 2}, []),
    ("test_one_failing_rank", {"passed": 1, "failed": 1}, [r"FAILED .*test_one_failing_rank\[2\]\[rank=0\].*"]),
    ("test_one_aborting_rank", {"passed": 1, "failed": 1}, []),
    ("test_number_of_processes_matches_ranks", {"passed": 6}, []),
    ("test_timeout", {"failed": 1}, [r"Timeout occurred for test_mpi.py::test_timeout\[2\]: "
                                     r"exceeded run time limit of 5s\."]),
    ("test_skip", {"skipped": 6}, []),
    ("test_mpi_tmp_path", {"passed": 2}, []),
    ("test_no_mpi", {"passed": 1}, []),
])
def test_outcomes(pytester, test, outcomes, lines):
    pytester.copy_example("test_mpi.py")
    result = pytester.runpytest("-k", test)
    result.assert_outcomes(**outcomes)
    if lines:
        result.stdout.re_match_lines(lines, consecutive=True)
