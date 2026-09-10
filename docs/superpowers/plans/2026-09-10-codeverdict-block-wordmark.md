# Verdict Block Wordmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render a prominent five-row `VERDICT` block wordmark in wide terminals while retaining the readable narrow-terminal fallback.

**Architecture:** Keep `app.cli.theme.wordmark(width, unicode)` as the single rendering boundary. Replace only the wide `_WORDMARK` data with a compact `VERDICT` block-letter representation; preserve the existing threshold and Unicode/ASCII fallback branches.

**Tech Stack:** Python 3.12, Rich, pytest.

## Global Constraints

- Wide Unicode wordmarks must visibly spell `VERDICT` in five rows.
- Every wide output row must remain at or below 72 characters.
- Narrow Unicode and ASCII output must continue to spell `VERDICT` on one line.
- Do not alter startup, provider, repository, or agent pipeline behaviour.

---

### Task 1: Replace the legacy wide-terminal wordmark

**Files:**
- Modify: `backend/app/cli/theme.py:136-159`
- Modify: `tests/test_cli.py:149-159`

**Interfaces:**
- Consumes: `wordmark(width: int, unicode: bool) -> list[str]`.
- Produces: A five-row wide wordmark and one-row narrow fallback, both visibly branded `CODEVERDICT`.

- [x] **Step 1: Write the failing test**

Extend `TestTheme.test_wordmark_degrades_on_a_narrow_terminal` with the observable wide-render assertions:

```python
assert len(wide) == 5
assert all("█" in line for line in wide)
```

Retain the existing narrow-output assertions that independently verify both Unicode and ASCII fallbacks spell `CODEVERDICT`.

- [x] **Step 2: Run the test to verify it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_cli.py::TestTheme::test_wordmark_degrades_on_a_narrow_terminal -q
```

Expected: FAIL because the legacy logo has six rows rather than the new five-row compact layout.

- [x] **Step 3: Write the minimal implementation**

Replace `_WORDMARK` in `backend/app/cli/theme.py` with the following five-row, block-letter `CODEVERDICT` banner. Keep `wordmark()` unchanged: it returns `_WORDMARK` only for `width >= 72 and unicode`, otherwise it uses the existing single-line Unicode/ASCII fallback strings.

```python
_WORDMARK = r"""
 ███  ███  ███  ███  █ █ █ ███ ███ ███  █  ███ ███
█    █ █  █ █  █    █ █ █ █   █ █  █   █   █   █ █
█    █ █  █ █  ██   █ █ █ ██  ███  █   █   █   █ █
█    █ █  █ █  █     █ █  █   █ █  █   █   █   █ █
 ███ ███  ███  ███    █  ███ ███ ███ ███ ███  █
"""
```

- [x] **Step 4: Run theme tests to verify the result**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_cli.py::TestTheme -q
```

Expected: PASS; wide and narrow branding, style resolution, and terminal fallback behaviour remain valid.

- [x] **Step 5: Visually verify startup**

Run:

```powershell
.\.venv\Scripts\codeverdict.exe
```

Expected: the startup screen displays a five-row `CODEVERDICT` block logo. Type `/quit` after inspection.
