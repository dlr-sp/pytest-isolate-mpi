import pickle
from types import SimpleNamespace

from pytest_isolate_mpi import _plugin


def test_nodeid_remains_unchanged(monkeypatch, tmp_path):
    nodeid = "test_example.py::test_mpi[2-rank0]"
    report = SimpleNamespace(
        nodeid=nodeid,
        location=("test_example.py", 0, "test_mpi[2-rank0]"),
    )
    item = SimpleNamespace()

    monkeypatch.setattr(
        _plugin.runner,
        "runtestprotocol",
        lambda *_args, **_kwargs: [report],
    )
    monkeypatch.setenv("PYTEST_MPI_REPORTS_PATH", str(tmp_path))

    _plugin.MPIPlugin()._mpi_runtestprococol_inner(item)  # pylint: disable=protected-access

    assert report.nodeid == nodeid
    assert report.location == ("test_example.py", 0, "test_mpi[2-rank0]")
    with open(tmp_path / "0", mode="rb") as f:
        [pickled_report] = pickle.load(f)
    assert pickled_report.nodeid == nodeid
