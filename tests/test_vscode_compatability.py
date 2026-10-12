from types import SimpleNamespace

from pytest_isolate_mpi import _plugin


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

    _plugin.MPIPlugin()._mpi_runtestprococol_inner(item)  # pylint: disable=protected-access

    assert report.nodeid == nodeid
