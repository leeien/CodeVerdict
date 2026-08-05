"""Tiny in-process job runner.

Long-running work (KB indexing, agent runs) is pushed onto a thread pool so the
HTTP request returns immediately and the UI can poll for progress. Each job
opens its own DB Session. For production this is where you'd swap in a real
queue (Celery / RQ / Arq).
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="job")


def submit(fn: Callable, *args, **kwargs) -> None:
    _executor.submit(_guard, fn, *args, **kwargs)


def _guard(fn: Callable, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 — never let a worker thread die silently
        import traceback

        print(f"[background] job {getattr(fn, '__name__', fn)} failed: {exc}")
        traceback.print_exc()
