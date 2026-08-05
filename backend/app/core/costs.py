"""Cost + token breakdown: totals -> per scope -> per ticket -> per agent.

Extracted verbatim from ``routers/costs.py`` so the terminal client reports the
same numbers the web UI does.
"""

from __future__ import annotations

from sqlmodel import Session, select

from ..models import AgentRun, ScopeSession, Task

AGENTS = ["pm", "dev", "qa", "review", "pr"]


def breakdown(db: Session, repo_id: int | None = None) -> dict:
    tq = select(Task)
    if repo_id is not None:
        tq = tq.where(Task.repo_id == repo_id)
    tasks = db.exec(tq).all()
    task_ids = {t.id for t in tasks}

    # Aggregate runs per task and per (task, agent), tracking in/out separately.
    def _blank():
        return {"cost": 0.0, "tokens_in": 0, "tokens_out": 0, "runs": 0, "unknown": False}

    per_task: dict[int, dict] = {}
    for r in db.exec(select(AgentRun)).all():
        if r.task_id not in task_ids:
            continue
        tin, tout = r.tokens_input or 0, r.tokens_output or 0
        pt = per_task.setdefault(
            r.task_id, {"cost": 0.0, "tokens_in": 0, "tokens_out": 0, "by_agent": {}, "unknown": False})
        pt["cost"] += r.cost_usd or 0.0
        pt["tokens_in"] += tin
        pt["tokens_out"] += tout
        # A backend that reports tokens but not a dollar cost (e.g. Codex, Cursor)
        # flags usage_unknown — carry it so the UI shows "n/a", not a fake $0.00.
        pt["unknown"] = pt["unknown"] or bool(r.usage_unknown)
        ba = pt["by_agent"].setdefault(r.agent_type, _blank())
        ba["cost"] += r.cost_usd or 0.0
        ba["tokens_in"] += tin
        ba["tokens_out"] += tout
        ba["runs"] += 1
        ba["unknown"] = ba["unknown"] or bool(r.usage_unknown)

    sess_by_id = {s.id: s for s in db.exec(select(ScopeSession)).all()}
    scopes: dict = {}
    unscoped = {"session_id": None, "title": "No scope", "cost": 0.0, "tokens_in": 0, "tokens_out": 0, "tickets": []}

    def _fmt_agent(v):
        return {"cost": round(v["cost"], 4), "tokens_in": v["tokens_in"], "tokens_out": v["tokens_out"],
                "tokens": v["tokens_in"] + v["tokens_out"], "cost_unknown": v["unknown"]}

    for t in tasks:
        pt = per_task.get(
            t.id, {"cost": 0.0, "tokens_in": 0, "tokens_out": 0, "by_agent": {}, "unknown": False})
        ticket = {
            "id": t.id, "key": t.key, "title": t.title, "status": t.status,
            "cost": round(pt["cost"], 4), "tokens_in": pt["tokens_in"], "tokens_out": pt["tokens_out"],
            "tokens": pt["tokens_in"] + pt["tokens_out"], "cost_unknown": pt["unknown"],
            "by_agent": {a: _fmt_agent(pt["by_agent"][a]) for a in AGENTS if a in pt["by_agent"]},
        }
        if t.session_id and t.session_id in sess_by_id:
            s = sess_by_id[t.session_id]
            bucket = scopes.setdefault(t.session_id, {"session_id": s.id, "title": s.title or f"Scope #{s.id}",
                                                      "cost": 0.0, "tokens_in": 0, "tokens_out": 0, "tickets": []})
        else:
            bucket = unscoped
        bucket["tickets"].append(ticket)
        bucket["cost"] += pt["cost"]
        bucket["tokens_in"] += pt["tokens_in"]
        bucket["tokens_out"] += pt["tokens_out"]

    # PM scoping is billed BEFORE any ticket exists, so it has no AgentRun to be
    # found under — it accumulates on the session itself. Leaving it out made
    # `/costs` disagree with the web overview (which does count it) and, worse,
    # hid real spend: a clarify loop that asks three questions costs six figures
    # of tokens and a scope that never got as far as tickets showed up nowhere at
    # all. The clarify loop is the cheapest place to catch a wrong requirement,
    # but only if what it costs is visible.
    repo_sessions = {t.session_id for t in tasks if t.session_id}
    for s in sess_by_id.values():
        pm_in, pm_out = s.pm_tokens_input or 0, s.pm_tokens_output or 0
        pm_cost = s.pm_cost_usd or 0.0
        if not (pm_in or pm_out or pm_cost):
            continue
        # With a repo filter, only sessions that produced tickets for this repo
        # can be attributed to it — a session carries no repo of its own.
        if repo_id is not None and s.id not in repo_sessions:
            continue
        bucket = scopes.setdefault(s.id, {"session_id": s.id, "title": s.title or f"Scope #{s.id}",
                                          "cost": 0.0, "tokens_in": 0, "tokens_out": 0, "tickets": []})
        bucket["scoping"] = {"cost": round(pm_cost, 4), "tokens_in": pm_in, "tokens_out": pm_out,
                             "tokens": pm_in + pm_out}
        bucket["cost"] += pm_cost
        bucket["tokens_in"] += pm_in
        bucket["tokens_out"] += pm_out

    scope_list = sorted(scopes.values(), key=lambda s: s["cost"], reverse=True)
    if unscoped["tickets"]:
        scope_list.append(unscoped)
    for s in scope_list:
        s["cost"] = round(s["cost"], 4)
        s["tokens"] = s["tokens_in"] + s["tokens_out"]

    return {
        "totals": {"cost": round(sum(s["cost"] for s in scope_list), 4),
                   "tokens_in": sum(s["tokens_in"] for s in scope_list),
                   "tokens_out": sum(s["tokens_out"] for s in scope_list),
                   "tokens": sum(s["tokens"] for s in scope_list),
                   "tickets": sum(len(s["tickets"]) for s in scope_list),
                   "cost_unknown": any(t.get("cost_unknown") for s in scope_list for t in s["tickets"])},
        "agents": AGENTS,
        "scopes": scope_list,
    }
