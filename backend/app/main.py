"""The local tools endpoint — not a web app.

CodeVerdict is a terminal program. What survives here is the one thing that has to
be reachable over a socket rather than a function call: the repository-index
tools the Dev and jury agents query.

Those agents run as separate processes (a headless coding CLI, sometimes a
container), so they cannot call into this process directly — and the index
cannot simply be opened twice, because the vector store is single-writer. So the
shim installed into the working copy posts here, and this process, which already
owns the graph and the store, answers. Authentication is a per-run token minted
in memory; the shim never names its own repo or working directory, the token
does.

There is no UI, no session cookie and no login: nothing here is meant for a
person with a browser.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlmodel import Session

from .core import CoreError
from .core import repos as core_repos
from .database import engine, init_db
from .routers import kb_tools
from .services import judges, runtime_settings
from .services.auth import ensure_bootstrap_admin


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    with Session(engine) as db:
        ensure_bootstrap_admin(db)       # the row a delivered PR is attributed to
        runtime_settings.apply_overrides(db)
        judges.ensure_seeded(db)
        # Nothing survives the process that ran it, so an `indexing` row here is
        # a corpse, not live work. Same call the terminal client makes at boot.
        core_repos.reconcile_interrupted(db)
    yield


app = FastAPI(title="CodeVerdict tools", version="0.1.0", lifespan=lifespan)


@app.exception_handler(CoreError)
def _core_error(request: Request, exc: CoreError) -> JSONResponse:
    """The use-case layer refuses work with a CoreError; its status codes are
    already HTTP-shaped, so this is the whole translation."""
    return JSONResponse(status_code=exc.status, content={"detail": exc.message})


# Bound to loopback by the launcher and gated per request by a run token.
app.include_router(kb_tools.router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}
