import pytest


@pytest.mark.parametrize(
    ["mixed", "outcomes"],
    [
        pytest.param(["plain", "mpi"], {"passed": 5}, id="plain_then_mpi"),
        pytest.param(["mpi", "plain", "mpi"], {"passed": 7}, id="mpi_plain_mpi"),
        pytest.param(["mpi", "plain"], {"passed": 5}, id="mpi_then_plain"),
    ],
)
def test_module_torn_down_after_trailing_mpi_test(pytester, mixed, outcomes):
    tests = []
    for index, kind in enumerate(mixed):
        marker = "@pytest.mark.mpi(ranks=2)\n" if kind == "mpi" else ""
        tests.append(f"{marker}def test_{kind}_{index}():\n    pass\n")
    pytester.makepyfile(
        test_a_mixed="import pytest\n\n\n" + "\n\n".join(tests),
        test_b_plain_only="def test_first():\n    pass\n\n\ndef test_second():\n    pass\n",
    )
    result = pytester.runpytest("-v", "-rA")
    result.assert_outcomes(**outcomes)


def test_teardown_error_after_trailing_mpi_test(pytester):
    pytester.makepyfile("""
        import pytest

        @pytest.fixture(scope="module")
        def failing_teardown():
            yield
            raise RuntimeError("teardown failed")

        def test_plain(failing_teardown):
            pass

        @pytest.mark.mpi(ranks=2)
        def test_mpi():
            pass
        """)

    result = pytester.runpytest("-v")

    result.assert_outcomes(passed=3, errors=1)
    result.stdout.no_fnmatch_line("*INTERNALERROR*")
