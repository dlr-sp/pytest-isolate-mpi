def test_non_mpi_test_respects_forked(pytester):
    pytester.makepyfile("""
        import os

        parent_pid = os.getpid()

        def test_runs_in_fork():
            assert os.getpid() != parent_pid
        """)

    result = pytester.runpytest("--forked", "-v")
    result.assert_outcomes(passed=1)
