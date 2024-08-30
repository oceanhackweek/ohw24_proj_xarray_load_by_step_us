from __future__ import annotations

import importlib.metadata

import load_by_step as m


def test_version():
    assert importlib.metadata.version("load_by_step") == m.__version__
