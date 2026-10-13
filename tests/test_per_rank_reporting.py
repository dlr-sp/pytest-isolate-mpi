import re


def test_number_of_reports_matches_collected_tests(pytester):
    pytester.makepyfile("""
        import pytest

        def test_plain():
            pass

        @pytest.mark.mpi(ranks=2)
        def test_mpi():
            pass
    """)

    result = pytester.runpytest("-v")

    result.assert_outcomes(passed=3)
    result.stdout.fnmatch_lines(["*collected 3 items"])

    percentages = [int(match) for match in re.findall(r"\[\s*(\d+)%\]", result.stdout.str())]
    assert percentages == [33, 66, 100]


def test_rank_tests_are_collected(pytester):
    pytester.makepyfile("""
        import pytest

        @pytest.mark.mpi(ranks=[2, 3])
        def test_mpi():
            pass
    """)

    result = pytester.runpytest("--collect-only", "-q")

    result.stdout.fnmatch_lines(
        [
            "test_rank_tests_are_collected.py::test_mpi[[]2-rank0]",
            "test_rank_tests_are_collected.py::test_mpi[[]2-rank1]",
            "test_rank_tests_are_collected.py::test_mpi[[]3-rank0]",
            "test_rank_tests_are_collected.py::test_mpi[[]3-rank1]",
            "test_rank_tests_are_collected.py::test_mpi[[]3-rank2]",
        ]
    )


def test_mpi_rank_fixture_matches_comm_rank(pytester):
    pytester.makepyfile("""
        import pytest

        @pytest.mark.mpi(ranks=3)
        def test_ranks(comm, mpi_rank):
            assert mpi_rank == comm.rank
    """)

    result = pytester.runpytest("-v")

    result.assert_outcomes(passed=3)


def test_mpi_rank_fixture_defaults_to_zero_for_plain_tests(pytester):
    pytester.makepyfile("""
        def test_plain(mpi_rank):
            assert mpi_rank == 0
    """)

    result = pytester.runpytest("-v")

    result.assert_outcomes(passed=1)


def test_mpi_rank_fixture_defaults_to_zero_without_isolation(pytester):
    pytester.makepyfile("""
        import pytest

        @pytest.mark.mpi(ranks=2)
        def test_mpi(mpi_rank):
            assert mpi_rank == 0
    """)

    result = pytester.runpytest("-v", "--no-mpi-isolation")

    result.assert_outcomes(passed=1)


def test_rank_tests_are_executed_in_a_single_mpi_job(pytester):
    pytester.makepyfile("""
        import pytest

        @pytest.mark.mpi(ranks=3)
        def test_mpi(comm, mpi_rank):
            assert mpi_rank == comm.rank
            with open(f"rank-{comm.rank}.log", mode="a") as f:
                f.write("executed\\n")
    """)

    result = pytester.runpytest("-v")

    result.assert_outcomes(passed=3)
    # The test body must have run exactly once per MPI process, i.e., one
    # line per rank and none of the ranks ran more than once.
    logs = {path.name: path.read_text().splitlines() for path in pytester.path.glob("rank-*.log")}
    assert set(logs) == {"rank-0.log", "rank-1.log", "rank-2.log"}
    assert all(lines == ["executed"] for lines in logs.values())


def test_user_parametrization_keeps_rank_parametrization_adjacent(pytester):
    pytester.makepyfile("""
        import pytest

        @pytest.mark.parametrize("value", ["a", "b"])
        @pytest.mark.mpi(ranks=2)
        def test_mpi(value):
            pass
    """)

    result = pytester.runpytest("--collect-only", "-q")

    result.stdout.fnmatch_lines(
        [
            "test_user_parametrization_keeps_rank_parametrization_adjacent.py::test_mpi[[]a-2-rank0]",
            "test_user_parametrization_keeps_rank_parametrization_adjacent.py::test_mpi[[]a-2-rank1]",
            "test_user_parametrization_keeps_rank_parametrization_adjacent.py::test_mpi[[]b-2-rank0]",
            "test_user_parametrization_keeps_rank_parametrization_adjacent.py::test_mpi[[]b-2-rank1]",
        ]
    )


def test_subset_of_rank_tests_can_be_selected(pytester):
    pytester.makepyfile("""
        import pytest

        @pytest.mark.mpi(ranks=2)
        def test_mpi(comm):
            assert comm.size == 2
    """)

    result = pytester.runpytest("-v", "test_subset_of_rank_tests_can_be_selected.py::test_mpi[2-rank1]")

    result.assert_outcomes(passed=1)


def test_single_failing_rank_is_attributed_to_its_rank_test(pytester):
    pytester.makepyfile("""
        import pytest

        @pytest.mark.mpi(ranks=2)
        def test_mpi(comm):
            assert comm.rank != 0
    """)

    result = pytester.runpytest("-v", "-rA")

    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(
        [
            "PASSED test_single_failing_rank_is_attributed_to_its_rank_test.py::test_mpi[[]2-rank1]*",
            "FAILED test_single_failing_rank_is_attributed_to_its_rank_test.py::test_mpi[[]2-rank0]*",
        ]
    )


def test_timeout_is_reported_for_every_rank(pytester):
    pytester.makepyfile("""
        import time

        import pytest

        @pytest.mark.mpi(ranks=2, timeout=1)
        def test_sleepy():
            time.sleep(30)
    """)

    result = pytester.runpytest("-v")

    result.assert_outcomes(failed=2)
    result.stdout.fnmatch_lines(
        [
            "Timeout occurred for test_timeout_is_reported_for_every_rank.py::test_sleepy[2-rank0]: "
            "exceeded run time limit of 1s.",
            "Timeout occurred for test_timeout_is_reported_for_every_rank.py::test_sleepy[2-rank1]: "
            "exceeded run time limit of 1s.",
        ]
    )


def test_mpi_tests_without_ranks_argument_get_a_single_rank_test(pytester):
    pytester.makepyfile("""
        import pytest

        @pytest.mark.mpi
        def test_mpi(comm, mpi_rank):
            assert comm.size == 1
            assert mpi_rank == 0
    """)

    result = pytester.runpytest("-v")

    result.assert_outcomes(passed=1)
