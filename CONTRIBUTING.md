# Contributing

Thanks for your interest in CodeVerdict. Contributions of all sizes are welcome —
bug reports, docs fixes, and features alike.

## Getting set up

```bash
git clone https://github.com/leeien/CodeVerdict
cd codeverdict
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"       # add ,treesitter for exact non-Python symbol maps
cp .env.example .env          # add an API key
```

Run the app with `./run.sh` (or `codeverdict`), and open http://localhost:8017.

## Before you open a pull request

Run both checks locally — CI runs the same ones on Python 3.11 and 3.12:

```bash
ruff check backend/app tests
pytest
```

The test suite makes **no network calls** — the LLM/agent boundary is mocked and the
knowledge base uses its TF-IDF fallback, so it runs anywhere in a few seconds. If you
change pipeline logic (verdict parsing, the revise loop, settings coercion, auth), add
or update a test for it.

## Guidelines

- **Match the surrounding style.** The codebase favors small, well-commented services;
  comments explain *why*, not *what*. Keep routers thin and put logic in `services/`.
- **Keep it dependency-light.** New runtime dependencies need a good reason — a lot of
  this project is deliberately built on the standard library (auth, static analysis,
  the TF-IDF fallback).
- **Don't break graceful degradation.** Features should still work when the embedding
  stack, an optional integration, or a provider key is absent.
- **Never commit secrets.** `.env`, `.secret.key`, and `*.db` are gitignored — keep it
  that way. See [SECURITY.md](SECURITY.md).
- **One logical change per PR**, with a clear description of what and why.

## Reporting bugs / requesting features

Open an issue using the templates under
[`.github/ISSUE_TEMPLATE`](.github/ISSUE_TEMPLATE). For bugs, include the steps to
reproduce, what you expected, and what happened (with logs where relevant).

By contributing, you agree that your contributions are licensed under the project's
[MIT License](LICENSE).
