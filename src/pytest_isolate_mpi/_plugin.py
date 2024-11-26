"""
Support for testing python code with MPI and pytest
"""

from __future__ import annotations

import argparse
import dataclasses
import subprocess
import pickle
import shutil
import sys
from tempfile import TemporaryDirectory

import collections
import os
import pytest


from _pytest import runner
from _pytest.main import Session
from _pytest.reports import TestReport

from ._constants import ENVIRONMENT_VARIABLE_TO_HIDE_INNARDS_OF_PLUGIN
from ._constants import MPIMarkerEnum
from ._constants import MPI_ENV_HINTS
from ._constants import TIME_UNIT_CONVERSION
from ._constants import VERBOSE_MPI_ARG
from ._fixturecache import _load_fixture_result
from ._fixturecache import _cache_fixture_result
from .fixtures import comm_fixture  # pylint: disable=unused-import
from .fixtures import mpi_tmpdir_fixture  # pylint: disable=unused-import
from .fixtures import mpi_tmp_path_fixture  # pylint: disable=unused-import


@dataclasses.dataclass(init=False)
class MPIConfiguration:
    """Configuration defining how to execute an MPI-parallel subprocess"""

    mpi_executable: str
    mpi_option_for_processes: str
    mpi_command_line_arguments: list[str]

    def __init__(
            self,
            mpi_executable: str = None,
            mpi_option_for_processes="-n",
            mpi_command_line_args: list[str] = None
    ):
        if not mpi_executable:
            self.mpi_executable = self.__deduce_mpirun_executable()
        else:
            if shutil.which(mpi_executable) is None:
                pytest.exit(f"failed to find mpi executable {mpi_executable} as defined in ini-file.",
                            pytest.ExitCode.USAGE_ERROR)
            self.mpi_executable = mpi_executable
        self.mpi_option_for_processes = mpi_option_for_processes
        self.mpi_command_line_arguments = mpi_command_line_args if mpi_command_line_args else []

    def extend_command_for_parallel_execution(
        self, cmd: list[str], ranks: int
    ) -> list[str]:
        """Extend a given command sequence by the mpirun call for the given number of ranks."""
        parallel_cmd = (
            [
                self.mpi_executable,
                self.mpi_option_for_processes,
                str(ranks),
            ]
            + self.mpi_command_line_arguments
            + cmd
        )

        return parallel_cmd

    @staticmethod
    def __deduce_mpirun_executable() -> str:
        if shutil.which("mpirun") is not None:
            return "mpirun"
        elif shutil.which("mpiexec") is not None:
            return "mpiexec"
        pytest.exit("failed to find mpirun/mpiexec required for starting MPI tests", pytest.ExitCode.USAGE_ERROR)


def assemble_sub_pytest_cmd(option: argparse.Namespace, nodeid: str):
    cmd = [sys.executable, "-m", "mpi4py", "-m", "pytest", "--capture", option.capture]
    if option.debug:
        cmd += ["--debug"]
    if option.verbose:
        cmd += [f"-{option.verbose * 'v'}"]
    # TODO: to be continued, go through `pytest --help` and look for relevant options to pass on
    cmd += [nodeid]
    return cmd


class MPIPlugin:
    """
    pytest plugin to assist with testing MPI-using code
    """

    _is_forked_mpi_environment: bool = False
    _verbose_mpi_info: bool = False
    _mpi_configuration: MPIConfiguration
    _session: Session | None = None
    _cache_tempdir: TemporaryDirectory | None = None

    def pytest_configure(self, config):
        """
        Hook setting config object (always called at least once)
        """
        self._is_forked_mpi_environment = bool(os.environ.get(ENVIRONMENT_VARIABLE_TO_HIDE_INNARDS_OF_PLUGIN, ""))
        self._verbose_mpi_info = config.getoption(VERBOSE_MPI_ARG)

        self._mpi_configuration = MPIConfiguration(
            mpi_executable=config.getini("mpi_executable"),
            mpi_option_for_processes=config.getini("mpi_option_for_processes"),
            mpi_command_line_args=config.getini("mpi_command_line_args")
        )

        # double check whether MPI environment variables are residing in the forked env
        if not self._is_forked_mpi_environment:
            for env in MPI_ENV_HINTS:
                if os.getenv(env):
                    pytest.exit("forked MPI tests cannot be run in an MPI environment", pytest.ExitCode.USAGE_ERROR)

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
                        "Range of MPI ranks must be an integer or an integer sequence", pytest.ExitCode.USAGE_ERROR
                    )

                for rank in list_of_ranks:
                    if not isinstance(rank, int) or rank <= 0:
                        pytest.exit("Number of MPI ranks must be a positive integer", pytest.ExitCode.USAGE_ERROR)

                metafunc.parametrize("mpi_ranks", list_of_ranks)  # maybe make this scope='session'?

    def pytest_runtest_setup(self, item):
        """
        Hook for doing additional MPI-related checks on mpi marked tests
        """
        for mark in item.iter_markers(name="mpi"):
            if mark.args:
                raise ValueError("mpi mark does not take positional args")

    @pytest.hookimpl(trylast=True)
    def pytest_sessionstart(self, session):
        self._session = session
        if "PYTEST_MPI_CACHE_PATH" not in os.environ:
            self._cache_tempdir = TemporaryDirectory()  # pylint: disable=consider-using-with
            os.environ["PYTEST_MPI_CACHE_PATH"] = self._cache_tempdir.name

    @pytest.hookimpl
    def pytest_sessionfinish(self, session):  # pylint: disable=unused-argument
        if self._cache_tempdir is not None:
            self._cache_tempdir.cleanup()
        self._session = None

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_protocol(self, item):
        ihook = item.ihook
        ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)

        if self._is_forked_mpi_environment:
            reports = self._mpi_runtestprococol_inner(item)
        else:
            if "mpi_ranks" in item.fixturenames:
                reports = self._mpi_runtestprotocol(item)
            else:
                reports = runner.runtestprotocol(item, log=False)

        for rep in reports:
            ihook.pytest_runtest_logreport(report=rep)

        ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)

        return True

    def _mpi_runtestprococol_inner(self, item):
        try:
            from mpi4py import MPI  # pylint: disable=import-outside-toplevel
        except ImportError:
            pytest.fail("MPI tests require that mpi4py be installed")

        comm = MPI.COMM_WORLD
        reports = runner.runtestprotocol(item, log=False)
        for report in reports:
            if report.location is not None:
                fspath, lineno, domain = report.location
                report.location = fspath, lineno, f"{domain}[rank={comm.rank}]"
                report.nodeid = f"{report.nodeid}[rank={comm.rank}]"
            setattr(report, "rank", comm.rank)
        if not os.path.isdir(os.environ["PYTEST_MPI_REPORTS_PATH"]):
            raise RuntimeError(
                "Could not find temporary directory. If you are using SLURM running on multiple compute nodes, this is "
                "likely because each node has its own local `/tmp`. Try setting `$TMPDIR` to a single directory "
                "outside the compute nodes."
            )
        with open(os.path.join(os.environ["PYTEST_MPI_REPORTS_PATH"], f"{comm.rank}"), mode="wb") as f:
            pickle.dump(reports, f)
        return reports

    def _mpi_runtestprotocol(self, item):  # pylint: disable=too-many-locals,too-many-branches
        mpi_ranks = 1
        for fixture in item.fixturenames:
            if fixture == "mpi_ranks" and "mpi_ranks" in item.callspec.params:
                mpi_ranks = item.callspec.params["mpi_ranks"]

        timeout = None
        timeout_in = ("NaN", "N/A")
        for mark in item.iter_markers(name="mpi"):
            timeout = mark.kwargs.get("timeout", None)
            if timeout is not None:
                unit = mark.kwargs.get("unit", "s")
                timeout_in = (unit, timeout)
                timeout = TIME_UNIT_CONVERSION[unit](timeout)

        cmd = self._mpi_configuration.extend_command_for_parallel_execution(
            cmd=assemble_sub_pytest_cmd(self._session.config.option, item.nodeid),
            ranks=mpi_ranks,
        )

        if self._verbose_mpi_info:
            print(f"dispatching command: {cmd}")

        reports = []
        mpi_proc_result = None
        timeout_expired = False

        with TemporaryDirectory() as tmpdir:
            run_env = os.environ.copy()
            run_env[ENVIRONMENT_VARIABLE_TO_HIDE_INNARDS_OF_PLUGIN] = "1"
            run_env["PYTEST_MPI_REPORTS_PATH"] = tmpdir

            try:
                # FIXME: disable capturing if -s is passed to pytest
                mpi_proc_result = subprocess.run(
                    cmd,
                    env=run_env,
                    universal_newlines=True,
                    timeout=timeout,
                    capture_output=True,
                    check=False,
                    cwd=self._session.config.rootdir,
                )
            except subprocess.TimeoutExpired:
                timeout_expired = True

            found_all_reports = True
            for i in range(mpi_ranks):
                try:
                    with open(os.path.join(tmpdir, f"{i}"), mode="rb") as f:
                        reports += pickle.load(f)
                except FileNotFoundError:
                    found_all_reports = False

            if not found_all_reports:
                if timeout_expired:
                    msg = (
                        f"Timeout occurred for {item.nodeid}: exceeded run time limit of "
                        f"{timeout_in[1]}{timeout_in[0]}."
                    )
                else:
                    msg = "At least one MPI process has exited prematurely."
                rep = TestReport(
                    nodeid=item.nodeid, location=item.location, outcome="failed", when="call", keywords={}, longrepr=msg
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
            from mpi4py import MPI, rc, get_config  # pylint: disable=import-outside-toplevel
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

    @pytest.hookimpl
    def pytest_fixture_setup(self, fixturedef, request):
        return _load_fixture_result(fixturedef, request)

    @pytest.hookimpl
    def pytest_fixture_post_finalizer(self, fixturedef, request):
        return _cache_fixture_result(fixturedef, request)


def pytest_configure(config):
    """
    Add pytest-mpi to pytest (see pytest docs for more info)
    """
    config.addinivalue_line("markers", f"{MPIMarkerEnum.MPI.value}: Tests that require being run with MPI/mpirun")
    config.pluginmanager.register(MPIPlugin())


def pytest_addoption(parser):
    """
    Add pytest-mpi options to pytest cli and ini file
    """
    group = parser.getgroup("mpi", description="support for MPI-enabled code")
    group.addoption(
        VERBOSE_MPI_ARG, action="store_true", default=False, help="Include detailed MPI information in output."
    )
    parser.addini("mpi_executable", type="string", default=None,
                  help="mpi executable (e.g. 'mpirun', 'mpiexec', 'srun')")
    parser.addini("mpi_option_for_processes", type="string", default="-n",
                  help="mpi option for number of processes ('-n')")
    parser.addini("mpi_command_line_args", type="args", default=None,
                  help="mpi command line arguments (e.g. '--bind-to core')")
