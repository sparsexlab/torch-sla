"""Consistency tests for the backend registry.

These guard against the class of bug where a new backend is added to
``BACKEND_METHODS`` (so it becomes selectable) but is forgotten in the
user-facing discovery helpers ``get_available_backends()`` and
``show_backends()`` -- leaving it invisible to ``torch_sla.show_backends()``.
"""
from __future__ import annotations

import torch_sla.backends as be


# Every backend that advertises solver methods must be a "known" backend, i.e.
# have a description + availability probe wired into the discovery helpers.
def test_every_method_backend_is_described():
    described = set(be._BACKEND_DESCRIPTIONS)
    with_methods = set(be.BACKEND_METHODS)
    missing = with_methods - described
    assert not missing, (
        f"backends {sorted(missing)} advertise methods but are missing from "
        "_BACKEND_DESCRIPTIONS (so show_backends()/get_available_backends() "
        "will silently drop them)"
    )


def test_availability_probe_covers_all_backends():
    # _backend_availability() must return one (name, ok) pair per described
    # backend, in the canonical order.
    names = [name for name, _ in be._backend_availability()]
    assert names == list(be._BACKEND_DESCRIPTIONS)


def test_default_method_is_a_supported_method():
    # Each backend's DEFAULT_METHODS entry must be one it actually supports.
    for backend, default in be.DEFAULT_METHODS.items():
        assert default in be.BACKEND_METHODS[backend], (
            f"default method {default!r} for backend {backend!r} is not in "
            f"its supported methods {be.BACKEND_METHODS[backend]}"
        )


def test_show_backends_lists_every_backend(capsys):
    be.show_backends()
    out = capsys.readouterr().out
    for name in be.BACKEND_METHODS:
        assert name in out, f"show_backends() omitted backend {name!r}"


def test_get_available_backends_subset_of_known():
    known = set(be._BACKEND_DESCRIPTIONS)
    assert set(be.get_available_backends()) <= known
