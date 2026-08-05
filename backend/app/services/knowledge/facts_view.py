"""Deterministic file→module grouping.

The one piece of module-derivation still in use: write_back.py tags a delivery
note's `related` links by the top-level module each touched file belongs to.
Pure path logic, no LLM.
"""

from __future__ import annotations


def _parts(relative_path: str) -> list[str]:
    parts = relative_path.replace("\\", "/").split("/")
    if parts and parts[0] in ("src", "lib"):
        parts = parts[1:]
    return parts


def module_of(relative_path: str) -> str:
    """Group a file into a module by its top-level package directory.

    src/orders/service.py -> 'orders'   (a leading 'src' is ignored)
    main.py               -> 'root'
    """
    parts = _parts(relative_path)
    return parts[0] if len(parts) > 1 else "root"
