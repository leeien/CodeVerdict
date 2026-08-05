"""Brand: the palette, the wordmark, and the small glyph vocabulary.

Two rules hold this together.

**Never set a background.** The terminal already has one, and it may be light or
dark — a hardcoded background is the single fastest way to make a CLI look
broken on half of the machines it runs on. Body text likewise stays the
terminal's own foreground colour. Only *accents* are coloured, and every accent
is picked to hold contrast on both a white and a near-black background.

**Degrade, don't disappear.** Colours are truecolor hex; Rich maps them down to
256/16 colours automatically. Glyphs have ASCII fallbacks for terminals that
can't render them, chosen by ``supports_unicode()`` rather than assumed.
"""

from __future__ import annotations

import os
import sys

from rich.console import Console
from rich.theme import Theme

# ── Palette ───────────────────────────────────────────────────────────────────
# Drawn from the mark: the gold of the scales, the wood of the bench, the black
# of the robe. Mid-tone by design so nothing washes out on a light terminal or
# muddies on a dark one.
GOLD = "#C8A24A"      # primary accent — the scales
GOLD_DIM = "#8A7231"  # the same accent, receded
OAK = "#9A6B43"       # the bench; secondary structure
INK = "#4A4A52"       # rules and borders
MUTED = "#8A8A94"     # de-emphasised text, equally readable either way
GREEN = "#4E9A51"
AMBER = "#CC8B29"
RED = "#C4453B"
BLUE = "#4A7EB5"
VIOLET = "#8C6BB1"

# The palette is defined once, as literal Rich style strings, because it has to
# serve two renderers that resolve styles differently. Rich looks names up in a
# Theme; Textual renders widgets through its own pipeline and never consults
# one — a named style like "sev.critical" silently comes out unstyled there.
# So: names for the REPL console (via THEME below), and `s()` to resolve a name
# to its literal string anywhere Textual will do the drawing.
STYLES: dict[str, str] = {
    # Structure
    "brand": f"bold {GOLD}",
    "brand.dim": GOLD_DIM,
    "rule": INK,
    "muted": MUTED,
    "heading": "bold",
    "key": f"bold {GOLD}",           # ticket keys, TASK-101
    "path": BLUE,                    # file paths and symbols
    "url": f"underline {BLUE}",

    # Verdict / status vocabulary — one colour per meaning, used everywhere
    "ok": GREEN,
    "warn": AMBER,
    "err": RED,
    "info": BLUE,
    "pending": MUTED,
    "running": GOLD,

    # Speakers in the transcript
    "you": "bold",
    "pm": VIOLET,
    "planner": BLUE,
    "dev": GOLD,
    "qa": GREEN,
    "jury": OAK,

    # Diff
    "diff.add": GREEN,
    "diff.del": RED,
    "diff.hunk": MUTED,
    "diff.file": f"bold {BLUE}",
    "diff.lineno": MUTED,

    # Severity, as the jury reports it
    "sev.critical": f"bold {RED}",
    "sev.high": RED,
    "sev.medium": AMBER,
    "sev.low": MUTED,
}

THEME = Theme(STYLES)


def s(name: str) -> str:
    """Resolve a palette name to its literal style string.

    Use this anywhere Textual does the rendering. Passing a bare name there is
    not an error — it just quietly loses the colour, which is worse.
    """
    return STYLES.get(name, name)


def supports_unicode() -> bool:
    """Whether we can draw the nice glyphs.

    Checked rather than assumed: a Windows console under a legacy code page or
    a CI log capturing to ASCII will raise or mojibake on the box-drawing and
    emoji-adjacent characters below.
    """
    if os.environ.get("CODEVERDICT_ASCII"):
        return False
    encoding = getattr(sys.stdout, "encoding", None) or ""
    if "utf" not in encoding.lower():
        return False
    try:
        "⚖ ◆ ▸ ✓ ✗ ─".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


class Glyphs:
    """The symbol vocabulary, resolved once against terminal capability."""

    def __init__(self, unicode: bool) -> None:
        self.scales = "⚖" if unicode else "*"
        self.prompt = "›" if unicode else ">"
        self.speaker = "◆" if unicode else "*"
        self.step = "▸" if unicode else ">"
        self.ok = "✓" if unicode else "+"
        self.fail = "✗" if unicode else "x"
        self.pending = "·" if unicode else "."
        self.running = "◇" if unicode else "~"
        self.arrow = "→" if unicode else "->"
        self.bullet = "•" if unicode else "-"
        self.ellipsis = "…" if unicode else "..."
        self.vbar = "│" if unicode else "|"


# ── Wordmark ──────────────────────────────────────────────────────────────────
# 68 columns — fits an 80-column terminal with room for the frame.
_WORDMARK = r"""
 ██████╗ ██████╗ ██████╗ ███████╗     ██╗██╗   ██╗██████╗ ██╗   ██╗
██╔════╝██╔═══██╗██╔══██╗██╔════╝     ██║██║   ██║██╔══██╗╚██╗ ██╔╝
██║     ██║   ██║██║  ██║█████╗       ██║██║   ██║██████╔╝ ╚████╔╝
██║     ██║   ██║██║  ██║██╔══╝  ██   ██║██║   ██║██╔══██╗  ╚██╔╝
╚██████╗╚██████╔╝██████╔╝███████╗╚█████╔╝╚██████╔╝██║  ██║   ██║
 ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝ ╚════╝  ╚═════╝ ╚═╝  ╚═╝   ╚═╝
"""

TAGLINE = "a jury of agents for your codebase"


def wordmark(width: int, unicode: bool) -> list[str]:
    """The banner, sized to the terminal.

    Below ~72 columns the block wordmark wraps into noise, so narrow terminals
    get the lockup instead of a broken picture.
    """
    if width >= 72 and unicode:
        return _WORDMARK.strip("\n").splitlines()
    if unicode:
        return ["⟨ ⚖ ⟩  C O D E J U R Y"]
    return ["<*>  C O D E J U R Y"]


def console(**kwargs) -> Console:
    """The one Console the terminal client draws through."""
    return Console(theme=THEME, highlight=False, soft_wrap=False, **kwargs)
