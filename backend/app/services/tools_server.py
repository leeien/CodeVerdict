"""The loopback endpoint the agents' index shim talks to.

Not a server in the product sense — there is no UI and nothing to visit. The
Dev and jury agents run as separate processes, so they cannot call into this one,
and they cannot open the index themselves because the vector store is
single-writer. They post to this instead, and the process that already owns the
graph answers.

It starts itself, once, the first time a run installs the shim. It used to be
started only by an explicit `/serve`, which meant that in an ordinary terminal
session the shim was written into the working copy pointing at nothing: every
tool call failed and the agent quietly fell back to grep. Nobody saw it, because
the shim is fail-open by design.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_base_url: str = ""


def _free_port(port: int, host: str) -> int:
    for candidate in range(port, port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, candidate))
                return candidate
            except OSError:
                continue
    return port


def ensure_running(host: str = "127.0.0.1", port: int = 8017) -> str:
    """Start the endpoint if it isn't up, and return its base URL ("" on failure).

    Never raises. The shim is fail-open — an agent with no tools still works,
    just less well — so a port that cannot be bound must not take down a run.
    """
    global _base_url
    with _lock:
        if _base_url:
            return _base_url
        try:
            import uvicorn

            from ..main import app

            chosen = _free_port(port, host)
            config = uvicorn.Config(app, host=host, port=chosen, log_level="warning")
            server = uvicorn.Server(config)
            # uvicorn installs signal handlers by default, which it cannot do off
            # the main thread — and it must not steal Ctrl-C from the prompt.
            server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
            # Daemon, so quitting takes the endpoint with it. A stray server
            # outliving its parent is the "stale process serving stale imports"
            # trap that costs an afternoon when the code underneath it moves.
            threading.Thread(target=server.run, name="codeverdict-tools",
                             daemon=True).start()

            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and not getattr(server, "started", False):
                time.sleep(0.05)
            if not getattr(server, "started", False):
                logger.warning("tools endpoint did not come up within 10s")
                return ""

            # The shim resolves the same way; keep the two in agreement.
            os.environ["PORT"] = str(chosen)
            os.environ.setdefault("HOST", host)
            _base_url = f"http://{host}:{chosen}"
            logger.info("tools endpoint on %s", _base_url)
            return _base_url
        except Exception as exc:  # noqa: BLE001
            logger.warning("tools endpoint unavailable: %s", exc)
            return ""
