"""
Internal helpers for testing only, do not use in main code
"""

from __future__ import annotations

from dataclasses import dataclass
import random

import pytest


def _fix_plural(**kwargs):
    """
    Work around error -> errors change in pytest 6
    """
    if int(pytest.__version__[0]) >= 6:
        return kwargs
    if "errors" in kwargs:
        errors = kwargs.pop("errors")
        kwargs["error"] = errors
    return kwargs


@dataclass
class ExpensiveComputation:
    """Mock for an expensive to compute state."""

    value: float
    computed_in_rank_of_size: tuple[int, int]
    was_cached: bool = False

    def __init__(self, comm):
        self.value = random.random()  # the expensive computation
        self.computed_in_rank_of_size = comm.rank, comm.size

    def __getstate__(self):  # gets called when instance is being pickled
        self.was_cached = True
        return self.__dict__.copy()
