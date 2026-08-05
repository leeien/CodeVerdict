"""A delivery that exhausted its revision rounds while still blocked must never
look like an approved one.

Found live on `rich`: the panel's Reliability juror correctly flagged an early
`return` inside the render generator that skipped the table's bottom border. The
Dev agent did not resolve it in two rounds, and the task landed in the `pr` lane
with the same status and card as a clean delivery.
"""

from __future__ import annotations

from app.services import prompts

BANNER = "DELIVERED WITHOUT APPROVAL"


class TestPrBody:
    def test_an_unapproved_delivery_is_flagged_in_the_first_line(self):
        body = prompts.scope_pr_body("Scope", [{"key": "T-1", "title": "t"}], "qa ok",
                                     f"⚠️ {BANNER} — 1 finding(s) remain UNRESOLVED")
        assert body.startswith("> ⚠️ **This PR was NOT approved")
        assert "before\n> merging" in body or "before merging" in body

    def test_an_approved_delivery_gets_no_warning(self):
        body = prompts.scope_pr_body("Scope", [{"key": "T-1", "title": "t"}], "qa ok",
                                     "## Jury review — APPROVED, no blocking findings")
        assert "NOT approved" not in body
        assert body.startswith("## Scope")

    def test_the_review_text_still_rides_along_in_full(self):
        review = f"⚠️ {BANNER} — do not merge.\n\nJuror said: bottom border unrendered."
        body = prompts.scope_pr_body("Scope", [{"key": "T-1", "title": "t"}], "qa", review)
        assert "bottom border unrendered" in body
