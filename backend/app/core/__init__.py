"""Use-case layer: everything the product can *do*, independent of how it is driven.

The operations live here as plain functions over a ``Session``, and the terminal
client is a thin adapter over them:

    cli/commands.py     →  core.tasks.approve(db, task_id)

The rule this layer follows is that it knows nothing about its caller. No Rich
console, no argparse, no prompt objects. Failures are raised as ``CoreError``
with an HTTP-shaped status code — kept because the status is a compact
*severity vocabulary* (404 missing, 409 wrong state, 403 refused), which the
terminal maps to a message. Below this sits ``services/``, which does the actual
work.
"""

from .errors import CoreError

__all__ = ["CoreError"]
