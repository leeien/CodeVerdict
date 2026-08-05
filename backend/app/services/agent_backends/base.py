"""Adapter contract for headless agentic coding tools.

Every backend exposes ``run(cwd, prompt, on_event, *, model, timeout)`` and
returns the common dict ``{text, tokens_in, tokens_out, cost, error}``. The tool
is invoked non-interactively, may edit files in ``cwd``, and streams progress
into ``on_event``. ``services/claude_agent.py`` (the Claude Code CLI runner) is
the reference implementation this contract was lifted from.

Observability is honest: ``tokens_in``/``tokens_out``/``cost`` are ``None`` when
the tool does not report them — never a fake zero.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from collections.abc import Callable

Event = Callable[[str, str], None]  # (severity, message)

# How long a cached availability probe stays valid (seconds).
_DETECT_TTL = 60.0


def new_result() -> dict:
    """The common result dict. Tokens/cost start as None (= unknown); adapters
    fill them in only from the tool's OWN reported usage."""
    return {"text": "", "tokens_in": None, "tokens_out": None, "cost": None, "error": None}


def short(x, n: int = 160) -> str:
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)
    s = " ".join(s.split())
    return s[:n]


class AgentBackend:
    """Base class for one headless coding-tool adapter.

    Subclasses set ``id``/``name`` and implement ``run()``; ``executable()``
    resolves the CLI binary (usually from a settings field). Detection
    (installed? which version?) is shared and cached here.
    """

    id: str = ""
    name: str = ""
    # argv appended to the executable to print a version string.
    version_args: tuple[str, ...] = ("--version",)
    # Set for tools that exist but have no scriptable/headless interface.
    headless: bool = True
    no_headless_reason: str = ""
    # One-line "how to enable this" shown in Settings when the tool is
    # unavailable — the install command plus how to authenticate it.
    connect_hint: str = ""
    # One-click install from the Settings screen: the argv to run (empty = no
    # auto-install), and the prerequisite binary it needs on PATH (checked first
    # so the user gets "install Node.js" instead of a cryptic spawn error).
    install_cmd: tuple[str, ...] = ()
    install_requires: str = ""

    def __init__(self) -> None:
        self._detect_cache: dict | None = None
        self._detect_at = 0.0
        self._detect_lock = threading.Lock()

    # -- to override --------------------------------------------------------
    def executable(self) -> str:
        raise NotImplementedError

    def run(self, cwd: str, prompt: str, on_event: Event, *,
            model: str | None = None, timeout: int = 1800) -> dict:
        raise NotImplementedError

    def chat(self, system: str, user: str = "", *, model: str | None = None,
             timeout: int = 180, json_mode: bool = False,
             messages: list[dict] | None = None) -> dict:
        """Pure-chat completion through the agentic CLI: run in an empty scratch
        dir with a no-tools instruction (same trick as claude_agent.chat)."""
        from pathlib import Path

        from ...config import settings

        scratch = Path(settings.repos_dir).expanduser() / "_chat"
        scratch.mkdir(parents=True, exist_ok=True)
        parts = [system]
        if json_mode:
            parts.append("Respond with a single valid JSON object and NOTHING else — "
                         "no prose, no markdown fences.")
        parts.append("Do not use any tools (no file reads, searches, or commands). "
                     "Answer directly from the information below.")
        if messages is not None:
            for m in messages:
                parts.append(f"[{m.get('role', 'user')}]\n{m.get('content', '')}")
        else:
            parts.append(user)
        prompt = "\n\n".join(p for p in parts if p)
        res = self.run(str(scratch), prompt, lambda sev, msg: None, model=model, timeout=timeout)
        return {"text": res["text"], "tokens_in": res["tokens_in"],
                "tokens_out": res["tokens_out"], "cost": res["cost"],
                "model": model or "", "error": res["error"]}

    # -- install / detection ------------------------------------------------
    def installable(self) -> bool:
        return bool(self.install_cmd) and self.headless

    def reset_detection(self) -> None:
        """Drop the cached probe so the next detect() re-checks PATH."""
        with self._detect_lock:
            self._detect_cache = None

    def install(self, timeout: int = 600) -> dict:
        """Run the tool's official installer and re-detect. Returns
        {ok, output, detect}; never raises. Admin-triggered from Settings."""
        if not self.installable():
            return {"ok": False, "output": f"{self.name} has no auto-install", "detect": self.detect()}
        if self.install_requires and shutil.which(self.install_requires) is None:
            return {"ok": False,
                    "output": f"'{self.install_requires}' is not installed — install it first "
                              f"(it provides the package manager this installer needs)",
                    "detect": self.detect()}
        try:
            p = subprocess.run(list(self.install_cmd), capture_output=True, text=True,
                               timeout=timeout)
            out = ((p.stdout or "") + "\n" + (p.stderr or "")).strip()
            ok = p.returncode == 0
        except subprocess.TimeoutExpired:
            out, ok = f"installer timed out after {timeout}s", False
        except OSError as exc:
            out, ok = f"installer could not run: {exc}", False
        self.reset_detection()
        det = self.detect()
        # Trust the re-probe over the exit code: installed-and-found is what matters.
        return {"ok": ok and det["available"], "output": out[-4000:], "detect": det}

    # -- shared plumbing ----------------------------------------------------
    def resolve(self) -> str | None:
        """Absolute path of the CLI, or None when not installed."""
        exe = self.executable()
        return shutil.which(exe) if exe else None

    def detect(self) -> dict:
        """{available, version, reason} — cached for a minute so settings-page
        polls don't fork a subprocess per render."""
        with self._detect_lock:
            now = time.monotonic()
            if self._detect_cache is not None and now - self._detect_at < _DETECT_TTL:
                return self._detect_cache
            self._detect_cache = self._detect_uncached()
            self._detect_at = now
            return self._detect_cache

    def _detect_uncached(self) -> dict:
        if not self.headless:
            return {"available": False, "version": "",
                    "reason": self.no_headless_reason or "no scriptable/headless interface",
                    "connect_hint": ""}  # nothing the operator can do — it's an IDE
        path = self.resolve()
        if path is None:
            return {"available": False, "version": "",
                    "reason": f"'{self.executable()}' not found on PATH",
                    "connect_hint": self.connect_hint}
        version = ""
        try:
            p = subprocess.run([path, *self.version_args], capture_output=True,
                               text=True, timeout=10)
            version = short((p.stdout or p.stderr or "").strip(), 80)
        except Exception:  # noqa: BLE001 — a broken --version still means installed
            pass
        return {"available": True, "version": version, "reason": "", "connect_hint": ""}

    def stream(self, cmd: list[str], cwd: str, on_line: Callable[[str], None],
               on_event: Event, result: dict, *, stdin_text: str | None = None,
               timeout: int = 1800, env: dict | None = None) -> None:
        """Run ``cmd`` and feed each stdout line to ``on_line``, with the same
        watchdog hard-kill as claude_agent.run_claude (a hung tool can't wedge
        the pipeline thread). Sets result['error'] on spawn/timeout/exit-code
        failures — parse errors inside on_line are the adapter's business."""
        try:
            proc = subprocess.Popen(
                cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
            )
        except (FileNotFoundError, OSError) as exc:
            result["error"] = f"{self.id} CLI not runnable: {exc}"
            on_event("error", result["error"])
            return
        try:
            if stdin_text is not None:
                proc.stdin.write(stdin_text)
            proc.stdin.close()
        except Exception as exc:  # noqa: BLE001
            result["error"] = f"failed to send prompt: {exc}"
            on_event("error", result["error"])
            proc.kill()
            return

        watchdog_fired = threading.Event()

        def _kill() -> None:
            watchdog_fired.set()
            proc.kill()

        watchdog = threading.Timer(timeout, _kill)
        watchdog.start()
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                if line.strip():
                    on_line(line)
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            result["error"] = "agent timed out"
            on_event("error", result["error"])
        finally:
            watchdog.cancel()
        if watchdog_fired.is_set() and not result["error"]:
            result["error"] = f"agent timed out after {timeout}s (watchdog)"
            on_event("error", result["error"])
        if proc.returncode not in (0, None) and not result["error"]:
            result["error"] = f"{self.id} exited with code {proc.returncode}"
