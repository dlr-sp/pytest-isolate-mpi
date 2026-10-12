import os

import pytest


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
