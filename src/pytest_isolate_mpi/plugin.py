"""
Support for testing python code with MPI and pytest
"""
from enum import Enum
from pathlib import Path
from subprocess import Popen
from tempfile import mkstemp

import collections
import os
import py
import pytest
import sys
import warnings

from _pytest import runner

from . import _version
__version__ = _version.get_versions()['version']


VERBOSE_MPI_ARG = "--verbose-mpi"

IS_FORKED_MPI_ARG = "--is-forked-by-main-pytest"


# list of env variables copied from HPX
MPI_ENV_HINTS = [
    "OMPI_COMM_WORLD_SIZE",
    "MV2_COMM_WORLD_RANK",
    "PMI_RANK",
    "ALPS_APP_PE",
    "PMIX_RANK",
    "PALS_NODEID",
]


class MPIMarkerEnum(str, Enum):
    """
    Enum containing all the markers used by pytest-mpi
    """
    mpi = "mpi"
    mpi_skip = "mpi_skip"
    mpi_xfail = "mpi_xfail"
    mpi_break = "mpi_break"


MPI_MARKERS = {
    MPIMarkerEnum.mpi_skip: pytest.mark.skip(
        reason="test does not work under mpi"
    ),
    MPIMarkerEnum.mpi_break: pytest.mark.skip(
        reason="test does not work under mpi"
    ),
    MPIMarkerEnum.mpi_xfail: pytest.mark.xfail(
        reason="test fails under mpi"
    ),
}


# copied from xdist remote
def serialize_report(rep):
    import py

    d = rep.__dict__.copy()
    if hasattr(rep.longrepr, "toterminal"):
        d["longrepr"] = str(rep.longrepr)
    else:
        d["longrepr"] = rep.longrepr
    for name in d:
        if isinstance(d[name], py.path.local):
            d[name] = str(d[name])
        elif name == "result":
            d[name] = None  # for now
    return d


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
    rep.wasxfail = (
        "reason: {xfail_reason}; "
        "pytest-forked reason: {crash_info}".format(
            xfail_reason=xfail_marker.kwargs["reason"],
            crash_info=info,
        )
    )
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
        self._is_forked_mpi_environment = config.getoption(IS_FORKED_MPI_ARG)
        self._verbose_mpi_info = config.getoption(VERBOSE_MPI_ARG)

        if not self._is_forked_mpi_environment:
            for env in MPI_ENV_HINTS:
                if os.getenv(env):
                    pytest.exit(
                        "forked MPI tests cannot be run in an MPI environment",
                        pytest.ExitCode.USAGE_ERROR)

            from distutils import spawn

            self._mpirun_exe = None
            if spawn.find_executable("mpirun") is not None:
                self._mpirun_exe = "mpirun"
            elif spawn.find_executable("mpiexec") is not None:
                self._mpirun_exe = "mpiexec"

            # FIXME: should this be here?
            if not self._mpirun_exe:
                pytest.exit(
                    "failed to find mpirun/mpiexec required for starting MPI tests",
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

    def pytest_collection_modifyitems(self, config, items):
        """
        Skip tests depending on what options are chosen
        """

        for item in items:
            self._add_markers(item)

    def pytest_terminal_summary(self, terminalreporter, exitstatus, *args):
        """
        Hook for printing MPI info at the end of the run
        """
        # pylint: disable=unused-argument
        if self._verbose_mpi_info:
            terminalreporter.section("MPI Information")
            try:
                from mpi4py import MPI, rc, get_config
            except ImportError:
                terminalreporter.write("Unable to import mpi4py")
            else:
                comm = MPI.COMM_WORLD
                terminalreporter.write("rank: {}\n".format(comm.rank))
                terminalreporter.write("size: {}\n".format(comm.size))

                terminalreporter.write("MPI version: {}\n".format(
                    '.'.join([str(v) for v in MPI.Get_version()])
                ))
                terminalreporter.write("MPI library version: {}\n".format(
                    MPI.Get_library_version()
                ))

                vendor, vendor_version = MPI.get_vendor()
                terminalreporter.write("MPI vendor: {} {}\n".format(
                    vendor, '.'.join([str(v) for v in vendor_version])
                ))

                terminalreporter.write("mpi4py rc: \n")
                for name, value in vars(rc).items():
                    terminalreporter.write(" {}: {}\n".format(name, value))

                terminalreporter.write("mpi4py config:\n")
                for name, value in get_config().items():
                    terminalreporter.write(" {}: {}\n".format(name, value))

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
        # TODO: only do this if we are set to very verbose
        print('starting test session:', session)
        self._session = session

    @pytest.hookimpl
    def pytest_sessionfinish(self, session):
        # TODO: only do this if we are set to very verbose
        print('terminating test session:', session)
        self._session = None

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_protocol(self, item):
        ihook = item.ihook
        ihook.pytest_runtest_logstart(
            nodeid=item.nodeid, location=item.location)

        if self._is_forked_mpi_environment:
            reports = runner.runtestprotocol(item, log=False)
        else:
            reports = self.mpi_runtestprotocol(item)

        for rep in reports:
            ihook.pytest_runtest_logreport(report=rep)

        ihook.pytest_runtest_logfinish(
            nodeid=item.nodeid, location=item.location)

        return True

    def mpi_runtestprotocol(self, item):
        mpi_ranks = 1
        for fixture in item.fixturenames:
            if fixture == "mpi_ranks" and "mpi_ranks" in item.callspec.params:
                mpi_ranks = item.callspec.params["mpi_ranks"]

        cmd = [
            self._mpirun_exe, "-n", str(mpi_ranks),
            sys.executable, "-m", "pytest", "--debug", IS_FORKED_MPI_ARG,
            # "--no-header",
            item.nodeid
        ]

        print("dispatching command:", cmd)

        out_fd, out_path = mkstemp()
        out = os.fdopen(out_fd, 'w')
        # out = sys.stdout

        err_fd, err_path = mkstemp()
        err = os.fdopen(err_fd, 'w')

        proc = Popen(cmd,
                     # stdout=out, stderr=err,
                     env=os.environ,
                     universal_newlines=True)

        proc.wait()

        with open(err_path, 'rb') as f:
            err_msg = f.read()
            print('err_msg:', err_msg)

        err.close()
        os.remove(err_path)

        # TODO: improve this, this is a quick hack
        return [
            runner.TestReport(
                nodeid=item.nodeid,
                location=item.location,
                outcome="failed",  # TODO: improve this
                when="call",  # TODO: improve this
                keywords=[],
                longrepr=item.nodeid
            )
        ]
        # import marshal
        # report_dumps = marshal.loads(err_msg)
        # [runner.TestReport(**x) for x in report_dumps]

        # ff = py.process.ForkedFunc(runforked)
        # result = ff.waitfinish()
        # if result.retval is not None:
        #     report_dumps = marshal.loads(result.retval)
        #     return [runner.TestReport(**x) for x in report_dumps]
        # else:
        #     if result.exitstatus == EXITSTATUS_TESTEXIT:
        #         pytest.exit(f"forked test item {item} raised Exit")
        #     return [report_process_crash(item, result)]


@pytest.fixture
def mpi_file_name(tmpdir, request):
    """
    Provides a temporary file name which can be used under MPI from all MPI
    processes.

    This function avoids the need to ensure that only one process handles the
    naming of temporary files.
    """
    try:
        from mpi4py import MPI
    except ImportError:
        pytest.fail("mpi4py needs to be installed to run this test")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # we only want to put the file inside one tmpdir, this creates the name
    # under one process, and passes it on to the others
    name = str(tmpdir.join(str(request.node) + '.hdf5')) if rank == 0 else None
    name = comm.bcast(name, root=0)
    return name


@pytest.fixture
def mpi_tmpdir(tmpdir):
    """
    Wraps `pytest.tmpdir` so that it can be used under MPI from all MPI
    processes.

    This function avoids the need to ensure that only one process handles the
    naming of temporary folders.
    """
    try:
        from mpi4py import MPI
    except ImportError:
        pytest.fail("mpi4py needs to be installed to run this test")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # we only want to put the file inside one tmpdir, this creates the name
    # under one process, and passes it on to the others
    name = str(tmpdir) if rank == 0 else None
    name = comm.bcast(name, root=0)
    return py.path.local(name)


@pytest.fixture
def mpi_tmp_path(tmp_path):
    """
    Wraps `pytest.tmp_path` so that it can be used under MPI from all MPI
    processes.

    This function avoids the need to ensure that only one process handles the
    naming of temporary folders.
    """
    try:
        from mpi4py import MPI
    except ImportError:
        pytest.fail("mpi4py needs to be installed to run this test")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # we only want to put the file inside one tmpdir, this creates the name
    # under one process, and passes it on to the others
    name = str(tmp_path) if rank == 0 else None
    name = comm.bcast(name, root=0)
    return Path(name)
#
# @pytest.fixture
# def run_as_subprocesses(ranks):
#     return "horst"


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
    config.pluginmanager.register(MPIPlugin())


def pytest_addoption(parser):
    """
    Add pytest-mpi options to pytest cli
    """
    group = parser.getgroup("mpi", description="support for MPI-enabled code")
    # TODO: hide behind a magic environment flag
    group.addoption(
        IS_FORKED_MPI_ARG, action="store_true", default=False,
        help="Whether we are running in an already forked environment. INTERNAL USE ONLY!."
    )
    group.addoption(
        VERBOSE_MPI_ARG, action="store_true", default=False,
        help="Include detailed MPI information in output."
    )
