# CodeVerdict

CodeVerdict is a terminal-based coding agent for working with existing codebases.
It turns a natural-language request into an engineering task, retrieves relevant
code, plans the change, invokes a coding agent, runs checks, and reviews the
result before delivery.

The project is designed for users who want a controllable development workflow:
each stage can use a different LLM provider or coding CLI, and code changes remain
behind a human approval gate.

## Highlights

- **Structured workflow**: Knowledge → PM → Planner → Dev → QA → Review.
- **Repository-aware retrieval**: Uses a code graph, semantic search, and keyword
  search to locate relevant files and symbols.
- **Configurable models**: Supports coding CLIs and OpenAI-compatible APIs,
  including DeepSeek.
- **Review jury**: Multiple independent review roles can evaluate a change and
  produce a consolidated verdict.
- **Safe delivery**: Supports demo mode, test execution, revision limits, and a
  manual approval step before changes are delivered.

## Requirements

- Python 3.11 or later
- Git
- At least one model provider:
  - a logged-in coding CLI such as Claude Code, Codex, Cursor, Aider, or Gemini CLI;
    or
  - an API key for an LLM provider such as DeepSeek, OpenAI, Groq, OpenRouter, or
    a local OpenAI-compatible endpoint.

Optional tools:

- `ripgrep` for faster search
- `codebase-memory-mcp` for the complete code graph
- GitHub CLI (`gh`) for creating pull requests

## Install

Clone the repository and install the project dependencies:

```bash
git clone https://github.com/leeien/CodeVerdict.git
cd CodeVerdict
pip install -e ".[semantic,treesitter]"
```

Start the interactive terminal:

```bash
codeverdict
```

On macOS or Linux, you may also use:

```bash
./run.sh
```

On Windows PowerShell:

```powershell
cd D:\path\to\CodeVerdict
.\.venv\Scripts\codeverdict.exe
```

If the virtual environment does not exist yet, create it and install first:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[semantic,treesitter]"
codeverdict
```

## Configure a model provider

Copy the example environment file before adding API keys:

```bash
copy .env.example .env
```

For DeepSeek, set the following values in `.env`. Do not commit this file or share
your API key.

```dotenv
CUSTOM_API_KEY=your_deepseek_api_key
CUSTOM_BASE_URL=https://api.deepseek.com

KNOWLEDGE_PROVIDER=custom
KNOWLEDGE_MODEL=deepseek-v4-flash
PM_PROVIDER=custom
PM_MODEL=deepseek-v4-pro
PLANNER_PROVIDER=custom
PLANNER_MODEL=deepseek-v4-pro
DEV_PROVIDER=custom
DEV_MODEL=deepseek-v4-pro
QA_PROVIDER=custom
QA_MODEL=deepseek-v4-flash
REVIEW_PROVIDER=custom
REVIEW_MODEL=deepseek-v4-pro
```

You can also change providers and models from the CodeVerdict terminal:

```text
/settings
/models
/model dev custom deepseek-v4-pro
```

Verify your local environment at any time:

```bash
codeverdict doctor
```

## Basic usage

Start by indexing a repository. CodeVerdict works on its own working copy, so the
source repository is not modified directly.

```text
/kb add https://github.com/pallets/click
```

Then describe the task in plain language and run the workflow:

```text
Add a --dry-run flag to the CLI runner
/tickets
/approve all
/run
/review
```

Common commands:

| Command | Description |
| --- | --- |
| `/doctor` | Check dependencies, model configuration, and optional tools. |
| `/kb add <repo-url>` | Clone and index a repository. |
| `/tickets` | View tasks generated from the request. |
| `/approve all` | Approve pending tasks before code changes are attempted. |
| `/run` | Run the Planner → Dev → QA → Review workflow. |
| `/review` | View the final verdict, findings, and diff. |
| `/models` | View model assignments for all stages. |
| `/settings` | Configure providers, API keys, delivery options, and limits. |
| `/costs` | View available token and cost information. |
| `/quit` | Exit the terminal. |

The same operations are available as commands for automation:

```bash
codeverdict doctor --json
codeverdict ingest https://github.com/pallets/click --wait
codeverdict scope "Add a --dry-run flag to the CLI runner"
codeverdict draft --json
codeverdict approve all
codeverdict run
```

## How it works

```text
Index repository
  → Retrieve relevant code and symbols
  → Clarify and create engineering tasks
  → Human approval
  → Plan implementation
  → Generate changes on an isolated branch
  → Run QA and review
  → Produce a delivery verdict
```

The knowledge base can use `codebase-memory-mcp` for code-graph analysis and
local embeddings with Qdrant for semantic retrieval. If optional tools are not
installed, CodeVerdict degrades to simpler symbol and keyword search rather than
failing immediately.

The review stage supports specialized independent judges, such as correctness,
reliability, security, and architecture. A foreperson consolidates their findings
into one verdict. This design is inspired by the
[SE-Jury paper](https://arxiv.org/abs/2505.20854); paper results are not presented
as benchmark results for this repository.

## Testing and development

Install development dependencies and run the checks:

```bash
pip install -e ".[dev]"
pytest
ruff check backend/app tests
```

The test suite covers the CLI, model backends, knowledge and retrieval pipeline,
jury logic, delivery gates, runtime settings, and other core workflow behavior.

Additional architecture and evaluation notes are available in:

- [Architecture](docs/architecture.md)
- [Knowledge base](docs/knowledge-base.md)
- [Configuration](docs/configuration.md)
- [Retrieval ablation](benchmarks/retrieval-ablation.md)
- [CodeVerdict and Claude Code task comparison](benchmarks/kb-vs-claude-code.md)

## License

MIT License. See [LICENSE](LICENSE).
