"""
Support for testing python code with MPI and pytest
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import subprocess
import enum
import pickle
import shutil
import sys
import warnings
from typing import Any, Union
from tempfile import TemporaryDirectory

import collections
import os
import pytest


from _pytest import runner
from _pytest.reports import TestReport

from ._constants import ENVIRONMENT_VARIABLE_TO_HIDE_INNARDS_OF_PLUGIN
from ._constants import IS_FORKED_MPI_ARG
from ._constants import MPI_ENV_HINTS
from ._constants import MPI_MARKERS
from ._constants import TIME_UNIT_CONVERSION
from ._constants import VERBOSE_MPI_ARG
from ._fixtures import comm           # pylint: disable=unused-import
from ._fixtures import mpi_file_name  # pylint: disable=unused-import
from ._fixtures import mpi_tmpdir     # pylint: disable=unused-import
from ._fixtures import mpi_tmp_path   # pylint: disable=unused-import


# list of env variables copied from HPX


@dataclasses.dataclass(init=False)
class MPIConfiguration:
    """Configuration defining how to execute an MPI-parallel subprocess"""
    mpirun_executable: str
    flag_for_processes: str
    flag_for_passing_environment_variables: str

    def __init__(self):
        """TODO: Make this configurable by the pytest ini files."""
        self.mpirun_executable = self.__get_mpirun_executable()
        self.flag_for_processes = "-n"
        self.flag_for_passing_environment_variables = "-x"

        if not self.mpirun_executable:
            pytest.exit(
                "failed to find mpirun/mpiexec required for starting MPI tests",
                pytest.ExitCode.USAGE_ERROR)

    def extend_command_for_parallel_execution(
        self, cmd: list[str], ranks: int, env_mod: dict[str, Union[str, int]]
    ) -> list[str]:
        """Extend a given command sequence by the mpirun call for the given number of ranks and environment modification
        given by env.

        """
        parallel_cmd = [
            self.mpirun_executable,
            self.flag_for_processes,
            str(ranks),
        ] \
        + self.get_arguments_for_passing_environment_variables(env_mod) \
        +  cmd

        return parallel_cmd

    def get_arguments_for_passing_environment_variables(self, env_mod: dict[str, Union[str, int]]):

        args = []
        for name, value in env_mod:
            args += [self.flag_for_passing_environment_variables, f"{name}={value}"]

        return args


    @staticmethod
    def __get_mpirun_executable() -> str:
        mpirun = ""
        if shutil.which("mpirun") is not None:
            mpirun = "mpirun"
        elif shutil.which("mpiexec") is not None:
            mpirun = "mpiexec"

        return mpirun


def assemble_sub_pytest_cmd(option: argparse.Namespace, nodeid: str):
    cmd = [
        sys.executable,
        "-m", "mpi4py",
        "-m", "pytest",
        "--capture", option.capture
    ]
    if option.debug:
        cmd += ["--debug"]
    if option.verbose:
        cmd += [f"-{option.verbose * 'v'}"]
    # TODO: to be continued, go through `pytest --help` and look for relevant options to pass on
    cmd += [nodeid]
    return cmd



# copied from pytest-forked
def report_process_crash(item, result):
    from _pytest._code import getfslineno

    path, lineno = getfslineno(item)
    info = "%s:%s: running the test CRASHED with signal %d" % (
        path,
        lineno,
        result.signal,
    )
    from _pytest import runner

    # pytest >= 4.1
    has_from_call = getattr(runner.CallInfo, "from_call", None) is not None
    if has_from_call:
        call = runner.CallInfo.from_call(lambda: 0 / 0, "???")
    else:
        call = runner.CallInfo(lambda: 0 / 0, "???")
    call.excinfo = info
    rep = runner.pytest_runtest_makereport(item, call)
    if result.out:
        rep.sections.append(("captured stdout", result.out))
    if result.err:
        rep.sections.append(("captured stderr", result.err))

    xfail_marker = item.get_closest_marker("xfail")
    if not xfail_marker:
        return rep

    rep.outcome = "skipped"
    rep.wasxfail = f"reason: {xfail_marker.kwargs['reason']}; pytest-isolate-mpi reason: {info}"
    warnings.warn(
        "pytest-forked xfail support is incomplete at the moment and may "
        "output a misleading reason message",
        RuntimeWarning,
    )

    return rep


class MPIPlugin:
    """
    pytest plugin to assist with testing MPI-using code
    """

    _is_forked_mpi_environment: bool = False
    _verbose_mpi_info: bool = False
    _mpi_configuration: MPIConfiguration = MPIConfiguration()
    _session: Any = None

    def _add_markers(self, item):
        """
        Add markers to tests when run under MPI.
        """
        for label, marker in MPI_MARKERS.items():
            if label in item.keywords:
                item.add_marker(marker)

    def pytest_configure(self, config):
        """
        Hook setting config object (always called at least once)
        """
        self._is_forked_mpi_environment = bool(os.environ.get(ENVIRONMENT_VARIABLE_TO_HIDE_INNARDS_OF_PLUGIN, ''))
        self._verbose_mpi_info = config.getoption(VERBOSE_MPI_ARG)
        self._mpi_configuration: MPIConfiguration = MPIConfiguration()

        # double check whether MPI environment variables are residing in the forked env
        if not self._is_forked_mpi_environment:
            for env in MPI_ENV_HINTS:
                if os.getenv(env):
                    pytest.exit(
                        "forked MPI tests cannot be run in an MPI environment",
                        pytest.ExitCode.USAGE_ERROR)

    def pytest_generate_tests(self, metafunc):
        """Extend the marker @pytest.mark.mpi such that we have parametrization of the tests w.r.t. # ranks."""
        for mark in metafunc.definition.iter_markers(name="mpi"):
            ranks = mark.kwargs.get("ranks")
            if ranks is not None:
                if isinstance(ranks, collections.abc.Sequence):
                    list_of_ranks = ranks
                elif isinstance(ranks, int):
                    list_of_ranks = [ranks]
                else:
                    list_of_ranks = []
                    pytest.exit(
                        "Range of MPI ranks must be an integer or an integer sequence",
                        pytest.ExitCode.USAGE_ERROR
                    )

                for rank in list_of_ranks:
                    if not isinstance(rank, int) or rank <= 0:
                        pytest.exit(
                            "Number of MPI ranks must be a positive integer",
                            pytest.ExitCode.USAGE_ERROR
                    )

                metafunc.parametrize("mpi_ranks", list_of_ranks)

    # TODO: remove the whole method?
    # def pytest_collection_modifyitems(self, config, items):
    #     """
    #     Skip tests depending on what options are chosen
    #     """
    #     for item in items:
    #         self._add_markers(item)


    def pytest_runtest_setup(self, item):
        """
        Hook for doing additional MPI-related checks on mpi marked tests
        """
        for mark in item.iter_markers(name="mpi"):
            if mark.args:
                raise ValueError("mpi mark does not take positional args")

            # in our outer pytest run, we do not need to do any further checks
            if not self._is_forked_mpi_environment:
                continue

            # check whether we have the correct number of cores
            try:
                from mpi4py import MPI
            except ImportError:
                pytest.fail("MPI tests require that mpi4py be installed")

            # TODO: remove this? (we fork with the required number of tests anyway!)
            comm = MPI.COMM_WORLD
            min_size = mark.kwargs.get('min_size')
            if min_size is not None and comm.size < min_size:
                pytest.skip(
                    f"Test requires {min_size} MPI processes, only {comm.size} MPI processes specified, skipping test"
                )

    @pytest.hookimpl(trylast=True)
    def pytest_sessionstart(self, session):
        self._session = session

    @pytest.hookimpl
    def pytest_sessionfinish(self, session):
        self._session = None

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_protocol(self, item):
        ihook = item.ihook
        ihook.pytest_runtest_logstart(
            nodeid=item.nodeid, location=item.location)

        if self._is_forked_mpi_environment:
            reports = self._mpi_runtestprococol_inner(item)
        else:
            if "mpi_ranks" in item.fixturenames:
                reports = self._mpi_runtestprotocol(item)
            else:
                reports = runner.runtestprotocol(item, log=False)

        for rep in reports:
            ihook.pytest_runtest_logreport(report=rep)

        ihook.pytest_runtest_logfinish(
            nodeid=item.nodeid, location=item.location)

        return True

    def _mpi_runtestprococol_inner(self, item):
        try:
            from mpi4py import MPI, rc, get_config
        except ImportError:
            pytest.fail("MPI tests require that mpi4py be installed")

        comm = MPI.COMM_WORLD
        reports = runner.runtestprotocol(item, log=False)
        for report in reports:
            if report.location is not None:
                fspath, lineno, domain = report.location
                report.location = fspath, lineno, f'{domain}[rank={comm.rank}]'
                report.nodeid = f'{report.nodeid}[rank={comm.rank}]'
            setattr(report, 'rank', comm.rank)
        with open(os.path.join(os.environ['PYTEST_MPI_REPORTS_PATH'], f'{comm.rank}'), mode='wb') as f:
            pickle.dump(reports, f)
        return reports

    def _mpi_runtestprotocol(self, item):
        mpi_ranks = 1
        for fixture in item.fixturenames:
            if fixture == "mpi_ranks" and "mpi_ranks" in item.callspec.params:
                mpi_ranks = item.callspec.params["mpi_ranks"]

        timeout = None
        timeout_in = ("NaN", "N/A")
        for mark in item.iter_markers(name="mpi_timeout"):
            if mark.args:
                raise ValueError("mpi mark does not take positional args")
            unit = mark.kwargs.get("unit", "s")
            timeout = mark.kwargs.get("timeout")
            timeout_in = (unit, timeout)
            timeout = TIME_UNIT_CONVERSION[unit](timeout)

        cmd = self._mpi_configuration.extend_command_for_parallel_execution(
            cmd=assemble_sub_pytest_cmd(self._session.config.option, item.nodeid),
            ranks=mpi_ranks,
            env_mod={},
        )

        if self._verbose_mpi_info:
            print(f"dispatching command: {cmd}")

        reports = []
        mpi_proc_result = None
        timeout_expired = False

        with TemporaryDirectory() as tmpdir:
            run_env = copy.copy(os.environ)
            run_env[ENVIRONMENT_VARIABLE_TO_HIDE_INNARDS_OF_PLUGIN] = "1"
            run_env['PYTEST_MPI_REPORTS_PATH'] = tmpdir

            try:
                # FIXME: disable capturing if -s is passed to pytest
                mpi_proc_result = subprocess.run(
                    cmd, env=run_env, universal_newlines=True, timeout=timeout, capture_output=True,
                )
            except subprocess.TimeoutExpired:
                timeout_expired = True

            found_all_reports = True
            for i in range(mpi_ranks):
                try:
                    with open(os.path.join(tmpdir, f'{i}'), mode='rb') as f:
                        reports += pickle.load(f)
                except FileNotFoundError:
                    found_all_reports = False

            if not found_all_reports:
                if timeout_expired:
                    msg = f"Timeout occurred for {item.nodeid}: exceeded run time limit of " \
                          f"{timeout_in[1]}{timeout_in[0]}."
                else:
                    msg = f"At least one MPI process has exited prematurely."
                rep = TestReport(
                    nodeid=item.nodeid,
                    location=item.location,
                    outcome="failed",
                    when="call",
                    keywords={},
                    longrepr=msg
                )
                if mpi_proc_result is not None:
                    if mpi_proc_result.stdout is not None:
                        rep.sections.append(("Captured stdout", mpi_proc_result.stdout))
                    if mpi_proc_result.stderr is not None:
                        rep.sections.append(("Captured stderr", mpi_proc_result.stderr))
                reports.append(rep)

        return reports

    def pytest_terminal_summary(self, terminalreporter, exitstatus, config):  # pylint: disable=unused-argument
        """Hook for printing MPI info at the end of the run"""
        if self._verbose_mpi_info:
            self._report_mpi_information(terminalreporter)

    def _report_mpi_information(self, terminalreporter):
        terminalreporter.section("MPI Information")
        try:
            from mpi4py import MPI, rc, get_config
        except ImportError:
            terminalreporter.write("Unable to import mpi4py")
        else:
            comm = MPI.COMM_WORLD
            terminalreporter.write(f"rank: {comm.rank}\n")
            terminalreporter.write(f"size: {comm.size}\n")

            terminalreporter.write(f"MPI version: {'.'.join([str(v) for v in MPI.Get_version()])}\n")
            terminalreporter.write(f"MPI library version: {MPI.Get_library_version()}\n")

            vendor, vendor_version = MPI.get_vendor()
            terminalreporter.write(f"MPI vendor: {vendor} {'.'.join([str(v) for v in vendor_version])}\n")

            terminalreporter.write("mpi4py rc:\n")
            for name, value in vars(rc).items():
                terminalreporter.write(f" {name}: {value}\n")

            terminalreporter.write("mpi4py config:\n")
            for name, value in get_config().items():
                terminalreporter.write(f" {name}: {value}\n")



def pytest_configure(config):
    """
    Add pytest-mpi to pytest (see pytest docs for more info)
    """
    config.addinivalue_line(
        "markers", "mpi: Tests that require being run with MPI/mpirun"
    )
    config.addinivalue_line(
        "markers", "mpi_break: Tests that cannot run under MPI/mpirun "
        "(deprecated)"
    )
    config.addinivalue_line(
        "markers", "mpi_skip: Tests to skip when running MPI/mpirun"
    )
    config.addinivalue_line(
        "markers", "mpi_xfail: Tests that fail when run under MPI/mpirun"
    )
    config.addinivalue_line(
        "markers", "mpi_timeout: Add a timeout to the test execution, units: [(s), m, h]"
    )
    config.pluginmanager.register(MPIPlugin())


def pytest_addoption(parser):
    """
    Add pytest-mpi options to pytest cli
    """
    group = parser.getgroup("mpi", description="support for MPI-enabled code")
    group.addoption(
        VERBOSE_MPI_ARG, action="store_true", default=False,
        help="Include detailed MPI information in output."
    )

    # Argument for explicitly running a forked session.
    # This requires two interactions to limit users from using it, aka: Explicit intent to use it is required.
    if ENVIRONMENT_VARIABLE_TO_HIDE_INNARDS_OF_PLUGIN in os.environ:
        group.addoption(
            IS_FORKED_MPI_ARG, action="store_true", default=False,
            help="Whether we are running in an already forked environment. INTERNAL USE ONLY!."
        )
