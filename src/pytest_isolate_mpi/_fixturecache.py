"""
Functionalities for caching fixture results across MPI-sessions.
"""

from __future__ import annotations

import io
import os
import pickle
from contextlib import ExitStack

import pytest
from _pytest.fixtures import FixtureDef
from _pytest.fixtures import SubRequest


def _restore_tmp_path_factory(state: dict) -> pytest.TempPathFactory:
    """Recreates a ``TempPathFactory`` from its pickled state."""
    factory = pytest.TempPathFactory.__new__(pytest.TempPathFactory)
    factory.__dict__.update(state)
    # Cleanup of the base temp directory stays with the session which created
    # it; a restored factory only needs an empty stack to be operational.
    factory._exit_stack = ExitStack()  # pylint: disable=protected-access
    return factory


class _CachePickler(pickle.Pickler):
    """Pickler which knows how to serialize Pytest's ``TempPathFactory``."""

    def reducer_override(self, obj):
        if isinstance(obj, pytest.TempPathFactory):
            state = {name: value for name, value in obj.__dict__.items() if name != "_exit_stack"}
            return _restore_tmp_path_factory, (state,)
        return NotImplemented


def _pickle_dumps(obj) -> bytes:
    """Pickles ``obj`` with the fixture-result recipes of ``_CachePickler``."""
    buffer = io.BytesIO()
    _CachePickler(buffer).dump(obj)
    return buffer.getvalue()


def _load_fixture_result(fixturedef: FixtureDef, request: SubRequest):
    """Loads fixture result from cache file, if it exists."""
    if fixturedef.scope == "session" and fixturedef.argname != "comm":
        cache_file_path = _get_cache_file_path(fixturedef, request)
        if os.path.isfile(cache_file_path):
            try:
                with open(cache_file_path, mode="rb") as f:
                    res = pickle.load(f)
            except Exception:  # pylint: disable=broad-exception-caught
                # ignore old corrupted files and re-evaluate the
                # fixture rather than failing the test.
                return None
            fixturedef.cached_result = (res, None, None)
            return True  # cache is loaded, do not call the fixture function
    return None  # continue calling the fixture function


def _cache_fixture_result(fixturedef: FixtureDef, request: SubRequest):
    """Saves fixture result to cache file, if it does not exists."""
    if fixturedef.scope == "session" and fixturedef.argname != "comm":
        cache_file_path = _get_cache_file_path(fixturedef, request)
        if not os.path.isfile(cache_file_path):
            res = fixturedef.cached_result[0]
            try:
                payload = _pickle_dumps(res)

            # A fixture that cannot be cached must never fail the test run.
            except Exception:  # pylint: disable=broad-exception-caught
                return  # skip caching; the fixture is re-evaluated in each subsession

            os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
            with open(cache_file_path, mode="wb") as f:
                f.write(payload)


def _get_cache_file_path(fixturedef: FixtureDef, request: SubRequest) -> str:
    """Returns a cache file path for a fixture call with size/rank combination."""
    comm = request.getfixturevalue("comm")
    # each MPI size/rank combination gets its own folder
    cache_dir = os.path.join(os.environ["PYTEST_MPI_CACHE_PATH"], f"size-{comm.size}", f"rank-{comm.rank}")
    identifier = _get_identifier(fixturedef, request)
    cache_file_path = os.path.join(cache_dir, identifier)
    return cache_file_path


def _get_identifier(fixturedef: FixtureDef, request: SubRequest) -> str:
    # pylint: disable=protected-access
    """Return a unique but minimal identifier string for a fixture call."""
    # all the fixtures
    fixturedefs: dict[str, FixtureDef] = {arg: f[0] for arg, f in request._arg2fixturedefs.items() if arg != "request"}
    # the test's parametrization as dictionary of indices:
    if hasattr(request._pyfuncitem, "callspec"):
        test_indices: dict[str, int] = request._pyfuncitem.callspec.indices
    else:
        test_indices: dict[str, int] = {}
    fixture_name: str = fixturedef.argname  # this fixture's name

    def get_dependent_names(fixturedef: FixtureDef) -> list[str]:
        """recursively finds all the fixture names, that `fixturedef` depends on."""
        dependent_names: list[str] = list(n for n in fixturedef.argnames if n != "request")
        for dependent_name in dependent_names:
            dependent_fixture = fixturedefs[dependent_name]
            dependent_names.extend(get_dependent_names(dependent_fixture))
        return dependent_names

    identifier = f"{fixture_name}"
    for name in (fixture_name, *get_dependent_names(fixturedef)):
        if name in test_indices:
            identifier += f"~{name}-{test_indices[name]}"
    return identifier
