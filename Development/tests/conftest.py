"""Pytest configuration for WEPPCLIFF Development/tests."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-slow", action="store_true", default=False, help="run slow KMQE tests")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: long-running integration (KMQE gap-fill)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-slow", default=False):
        return
    skip = pytest.mark.skip(reason="pass --run-slow to execute slow KMQE fair-compare tests")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)