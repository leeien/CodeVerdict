"""Full-screen panels.

Three things in CodeVerdict are genuinely two-dimensional and do not belong in a
scrolling transcript:

* **the jury's review** — four independent opinions, a synthesis, a diff and a
  plan, which a reader needs to move *between* rather than read in sequence;
* **settings** — 59 fields across 7 groups, with dependent provider→model
  pickers that only make sense side by side;
* **the jury roster** — an ordered list edited in place.

Everything else stays in the conversation. These open over it, take the
alternate screen, and hand control back where they found it, so the scrollback
is still the record of the session when they close.
"""

from __future__ import annotations

BRAND_CSS = """
/* Design tokens only — no literal colours for surfaces or text, so the panel
   inherits whichever Textual theme is active and stays legible in both. */
Screen { background: $surface; }

#title { text-style: bold; padding: 0 1; }
.subtle { color: $text-muted; }
.pad { padding: 1 2; }

Tabs { dock: top; }
Footer { background: $panel; }

.verdict-approved { color: $success; text-style: bold; }
.verdict-changes  { color: $warning; text-style: bold; }
.verdict-bad      { color: $error;   text-style: bold; }
"""
