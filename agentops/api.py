"""HTTP front end: POST /chat with a thread id for conversation memory."""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .graph import GuardedAgent

app = FastAPI(title="agent-guardrails-observability")
agent = GuardedAgent()


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str = Field(default="default", max_length=64)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat")
def chat(body: ChatIn) -> dict:
    out = agent.run(body.message, body.thread_id)
    return {
        "answer": out.get("answer", ""),
        "blocked": bool(out.get("blocked")),
        "guard_events": out.get("guard_events", []),
        "tools": [r["tool"] for r in out.get("tool_results", [])],
    }
