# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue for a security vulnerability. Instead, use
GitHub's [private vulnerability reporting](https://github.com/leeien/CodeVerdict/security/advisories/new)
(Security -> Report a vulnerability). Reports will be acknowledged within a few days and
keep you updated on the fix.

## Scope and operational notes

CodeVerdict runs LLM agents that **clone repositories and execute code and tests
from them**. That is inherent to what it does, and it shapes how it should be deployed:

- **Treat any instance as sensitive.** It holds provider API keys and can push commits
  and open pull requests. Do not expose it to an untrusted network. `run.sh` binds to
  `127.0.0.1` by default; only bind `0.0.0.0` behind a trusted network or a proxy that
  handles TLS and access control.
- **Only run it against repositories you trust**, unless you sandbox execution. The
  QA/Dev stages run tests from the target repo. The Docker image runs as a non-root
  user, but that is not a substitute for a real sandbox; containerized per-repo test
  execution is on the roadmap.
- **Demo mode is on by default** so the PR stage is a dry-run until you deliberately
  opt in and authenticate `gh`.

## How secrets are handled

- The first-boot `admin` password is generated randomly (or taken from
  `ADMIN_PASSWORD`) — there is no guessable default.
- API keys saved from the Settings screen are encrypted at rest with Fernet, keyed by
  `CODEVERDICT_SECRET_KEY` or a generated `.secret.key` file (chmod 600).
- `.env`, `.secret.key`, and `*.db` are gitignored. **Never commit them.** If you ever
  do, rotate the exposed credentials immediately — removing them in a later commit is
  not enough.

## Supported versions

This is an actively developed project; fixes land on `main`. Please report against the
latest `main`.
