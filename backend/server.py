"""
server.py
---------
FastAPI backend — exposes both assistants via a REST API.

Run:
    cd backend
    uvicorn server:app --reload --port 8000

Or from repo root:
    uvicorn backend.server:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .oss_assistant import OSSAssistant
from .frontier_assistant import FrontierAssistant
from .evaluator import evaluate


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

oss      = OSSAssistant()
frontier = FrontierAssistant()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 DualMind API starting …")
    yield
    print("🛑 DualMind API shutting down")


app = FastAPI(
    title="DualMind API",
    description="OSS vs Frontier AI Evaluation Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    use_llm_judge: bool = False


class ChatResponse(BaseModel):
    oss_text:           str
    frontier_text:      str
    oss_latency_ms:     int
    frontier_latency_ms: int
    oss_score:          dict
    frontier_score:     dict
    oss_model:          str
    frontier_model:     str


class SingleChatRequest(BaseModel):
    message: str


class ResetRequest(BaseModel):
    model: str   # "oss" | "frontier" | "both"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status":                "ok",
        "oss_model":             oss.model,
        "frontier_model":        frontier.model,
        "oss_history_turns":     len(oss.history) // 2,
        "frontier_history_turns": len(frontier.history) // 2,
    }


@app.post("/chat/both", response_model=ChatResponse)
async def chat_both(req: ChatRequest):
    """Send a message to both models in parallel."""
    loop = asyncio.get_event_loop()
    oss_resp, frontier_resp = await asyncio.gather(
        loop.run_in_executor(None, oss.chat,      req.message),
        loop.run_in_executor(None, frontier.chat,  req.message),
    )

    oss_score      = evaluate(req.message, oss_resp.text,      use_llm_judge=req.use_llm_judge)
    frontier_score = evaluate(req.message, frontier_resp.text, use_llm_judge=req.use_llm_judge)

    return ChatResponse(
        oss_text=oss_resp.text,
        frontier_text=frontier_resp.text,
        oss_latency_ms=oss_resp.latency_ms,
        frontier_latency_ms=frontier_resp.latency_ms,
        oss_score={
            "safety":       oss_score.safety,
            "hallucination": oss_score.hallucination,
            "bias":         oss_score.bias,
            "method":       oss_score.method,
        },
        frontier_score={
            "safety":       frontier_score.safety,
            "hallucination": frontier_score.hallucination,
            "bias":         frontier_score.bias,
            "method":       frontier_score.method,
        },
        oss_model=oss_resp.model,
        frontier_model=frontier_resp.model,
    )


@app.post("/chat/oss")
def chat_oss(req: SingleChatRequest):
    resp  = oss.chat(req.message)
    score = evaluate(req.message, resp.text)
    return {
        "text":       resp.text,
        "latency_ms": resp.latency_ms,
        "model":      resp.model,
        "score": {
            "safety":       score.safety,
            "hallucination": score.hallucination,
            "bias":         score.bias,
        },
    }


@app.post("/chat/frontier")
def chat_frontier(req: SingleChatRequest):
    resp  = frontier.chat(req.message)
    score = evaluate(req.message, resp.text)
    return {
        "text":       resp.text,
        "latency_ms": resp.latency_ms,
        "model":      resp.model,
        "cost_usd":   resp.cost_usd,
        "score": {
            "safety":       score.safety,
            "hallucination": score.hallucination,
            "bias":         score.bias,
        },
    }


@app.post("/reset")
def reset(req: ResetRequest):
    if req.model in ("oss", "both"):
        oss.reset()
    if req.model in ("frontier", "both"):
        frontier.reset()
    return {"status": "ok", "cleared": req.model}


@app.get("/history/{model}")
def get_history(model: str):
    if model == "oss":
        return {"history": oss.get_history()}
    if model == "frontier":
        return {"history": frontier.get_history()}
    raise HTTPException(status_code=400, detail="model must be 'oss' or 'frontier'")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
