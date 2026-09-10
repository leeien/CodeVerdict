# CodeVerdict Terminal Wordmark

## Goal

Replace the legacy `CODEJURY` branding in the terminal banner with a prominent
`VERDICT` wordmark while keeping the startup screen readable in narrow
terminals.

## Design

- At wide terminal widths, render `VERDICT` as a compact, five-row block
  wordmark using the existing Unicode-capable CLI path and brand colour.
- At narrow widths, retain the existing one-line fallback, spelling
  `V E R D I C T` in both Unicode and ASCII modes.
- Keep the existing tagline and all runtime behaviour unchanged.

## Acceptance Criteria

1. A wide Unicode terminal's rendered wordmark visibly spells `VERDICT`.
2. The wide wordmark fits within the established 72-column rendering budget.
3. Narrow Unicode and ASCII fallbacks visibly spell `CODEVERDICT`.
4. CLI theme tests cover wide and narrow branding output.
