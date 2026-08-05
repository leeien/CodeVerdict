"""Repository-index tools for the Dev agent's `.codeverdict/kb` shim.

The shim runs as a separate process, but the code graph and the embedded vector
store live in THIS one (embedded Qdrant is single-writer — a second process gets
a lock error and silently degrades to keyword-only search). So the shim asks the
server, which runs the query with the full hybrid index and returns text.

Auth is the token minted by ``tools.install()`` when the shim was written for a
specific run: it is held in this process's memory, never persisted, dies with a
restart, and resolves to the ONE (repo, working copy) that run may query — the
caller cannot name its own. Read-only by construction: every tool is a lookup
over a repo the server already indexed.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..services.knowledge import tools

router = APIRouter(prefix="/kb", tags=["dev-tools"])


class ToolCall(BaseModel):
    token: str
    tool: str
    arg: str = ""


@router.post("/tool")
def run_tool(body: ToolCall) -> dict:
    bound = tools.resolve_token(body.token)
    if bound is None:
        raise HTTPException(403, "unknown or expired tool token")
    repo, cwd = bound
    return {"result": tools.call(repo, cwd, body.tool, body.arg)}
