"""Pytest configuration: expose SuiteSparse fixtures from ``_suitesparse``.

Without this, fixtures defined in ``tests/_suitesparse.py`` would not be
discovered by pytest (only fixtures in ``conftest.py`` or fixtures
explicitly imported into test modules are visible).
"""
from tests._suitesparse import (  # noqa: F401
    suitesparse_any,
    suitesparse_complex,
    suitesparse_complex_small,
)
