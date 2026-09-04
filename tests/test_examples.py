import os

import pytest
from types import SimpleNamespace
from pytest_isolate_mpi import _plugin


@pytest.mark.parametrize(
    ["test", "outcomes", "lines"],
    [
        pytest.param("test_basic", {"passed": 2}, [], id="test_basic"),
        pytest.param("test_fail", {"failed": 2}, [r"FAILED .*test_fail\[2\].*"] * 2, id="test_fail"),
        pytest.param("test_xfail", {"xfailed": 2}, [], id="test_xfail"),
        pytest.param(
            "test_one_failing_rank",
            {"passed": 1, "failed": 1},
            [r"FAILED .*test_one_failing_rank\[2\].*"],
            id="test_one_failing_rank",
        ),
        pytest.param("test_one_aborting_rank", {"passed": 1, "failed": 1}, [], id="test_one_aborting_rank"),
        pytest.param(
            "test_number_of_processes_matches_ranks", {"passed": 6}, [], id="test_number_of_processes_matches_ranks"
        ),
        pytest.param(
            "test_timeout",
            {"failed": 1},
            [r"Timeout occurred for test_timeout.py::test_timeout\[2\]: exceeded run time limit of 5s\."],
            id="test_timeout",
        ),
        pytest.param(
            "test_mpi_deadlock",
            {"failed": 1, "passed": 1},
            [r"Timeout occurred for test_mpi_deadlock.py::test_mpi_deadlock\[2\]: exceeded run time limit of 10s\."],
            id="test_mpi_deadlock",
        ),
        pytest.param("test_skip", {"skipped": 6}, [], id="test_skip"),
        pytest.param("test_mpi_tmp_path", {"passed": 2}, [], id="test_mpi_tmp_path"),
        pytest.param("test_no_mpi", {"passed": 1}, [], id="test_no_mpi"),
        pytest.param("test_session_scoped_fixtures", {"passed": 36}, [], id="test_cache"),
        pytest.param("test_omp_num_threads", {"passed": 3}, [], id="test_omp_num_threads"),
        pytest.param("test_unpicklable_session_fixture", {"passed": 4}, [], id="test_unpicklable_session_fixture"),
        pytest.param("test_tmp_path_factory_cached", {"passed": 4}, [], id="test_tmp_path_factory_cached"),
    ],
)
def test_outcomes(pytester, test, outcomes, lines):
    pytester.copy_example(f"{test}.py")
    result = pytester.runpytest("-v", "-rA")
    result.assert_outcomes(**outcomes)
    if lines:
        result.stdout.re_match_lines(lines, consecutive=True)


def test_vscode_nodeid_remains_unchanged(monkeypatch, tmp_path):
    nodeid = "test_example.py::test_mpi[2]"
    report = SimpleNamespace(
        nodeid=nodeid,
        location=("test_example.py", 0, "test_mpi[2]"),
    )
    item = SimpleNamespace(
        config=SimpleNamespace(option=SimpleNamespace(plugins=["vscode_pytest"])),
    )

    monkeypatch.setattr(
        _plugin.runner,
        "runtestprotocol",
        lambda *_args, **_kwargs: [report],
    )
    monkeypatch.setenv("PYTEST_MPI_REPORTS_PATH", str(tmp_path))

    _plugin.MPIPlugin()._mpi_runtestprococol_inner(item)

    assert report.nodeid == nodeid


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
    ],
)
def test_outcomes_no_isolation(pytester, test, outcomes, lines):
    pytester.copy_example(f"{test}.py")
    result = pytester.runpytest("-v", "-rA", "--no-mpi-isolation")
    result.assert_outcomes(**outcomes)
    if lines:
        result.stdout.re_match_lines(lines, consecutive=True)


@pytest.mark.parametrize(
    "threads",
    [
        pytest.param("0", id="zero"),
        pytest.param("-1", id="negative"),
        pytest.param("2.5", id="float"),
        pytest.param("'4'", id="string"),
        pytest.param("True", id="boolean"),
    ],
)
def test_invalid_thread_count(pytester, threads):
    pytester.makepyfile(f"""
        import pytest

        @pytest.mark.mpi(ranks=2, threads={threads})
        def test_invalid_threads(mpi_ranks):
            assert True
        """)

    result = pytester.runpytest("-v")
    combined_output = result.stdout.str() + result.stderr.str()

    assert result.ret != pytest.ExitCode.OK
    assert "Number of OpenMP threads must be a positive integer" in combined_output


def test_thread_count_does_not_modify_outer_environ(
    pytester,
    monkeypatch,
):
    monkeypatch.setenv("OMP_NUM_THREADS", "9")

    pytester.makepyfile("""
        import os

        import pytest

        @pytest.mark.mpi(ranks=2, threads=3)
        def test_inner_environ(mpi_ranks):
            assert os.environ["OMP_NUM_THREADS"] == "3"
        """)

    result = pytester.runpytest("-v")

    result.assert_outcomes(passed=2)
    assert os.environ["OMP_NUM_THREADS"] == "9"


def test_non_mpi_test_respects_forked(pytester):
    pytester.makepyfile("""
        import os

        parent_pid = os.getpid()

        def test_runs_in_fork():
            assert os.getpid() != parent_pid
        """)

    result = pytester.runpytest("--forked", "-v")
    result.assert_outcomes(passed=1)
