def test_pass(pytester):
    pytester.copy_example("test_mpi.py")
    result = pytester.runpytest("-k", "test_pass")
    result.assert_outcomes(passed=2tui)
