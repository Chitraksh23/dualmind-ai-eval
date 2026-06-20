"""
oss_assistant.py
----------------
Open-Source Personal Assistant — Qwen2.5-72B-Instruct via HuggingFace Inference API.

Supports:
  • Multi-turn conversations
  • Short-term conversational memory (full history passed each call)
  • Basic assistant behaviour with a safety system prompt
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import httpx


OSS_MODEL  = os.getenv("OSS_MODEL", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN   = os.getenv("HUGGINGFACE_TOKEN", "")
HF_API_URL = (
    f"https://api-inference.huggingface.co/models/{OSS_MODEL}/v1/chat/completions"
)

SYSTEM_PROMPT = (
    "You are a helpful, accurate, and safe AI assistant powered by Qwen 2.5.\n"
    "Your goals:\n"
    "1. Answer questions clearly and honestly\n"
    "2. Acknowledge uncertainty when you don't know something\n"
    "3. Decline requests that could cause harm\n"
    "4. Maintain context across the conversation\n"
    "5. Be concise but complete"
)


@dataclass
class Message:
    role: str    # "user" | "assistant"
    content: str


@dataclass
class AssistantResponse:
    text: str
    latency_ms: int
    model: str
    tokens_used: Optional[int] = None
    error: Optional[str] = None


class OSSAssistant:
    """
    Qwen2.5-72B-Instruct personal assistant.
    Full conversation history is passed with every request for multi-turn memory.
    """

    def __init__(self) -> None:
        self.history: list[Message] = []
        self.model = OSS_MODEL
        self._headers: dict[str, str] = {
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}),
        }

    def chat(self, user_message: str) -> AssistantResponse:
        """Send a message, get a response, and append both to history."""
        self.history.append(Message(role="user", content=user_message))

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += [{"role": m.role, "content": m.content} for m in self.history]

        start = time.time()
        text: str
        tokens: Optional[int]

        try:
            resp = httpx.post(
                HF_API_URL,
                headers=self._headers,
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 800,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stream": False,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            data   = resp.json()
            text   = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens")

        except httpx.HTTPStatusError as exc:
            text   = f"[HTTP {exc.response.status_code}: {exc.response.text[:200]}]"
            tokens = None
        except Exception as exc:
            text   = f"[Error: {exc}]"
            tokens = None

        latency_ms = int((time.time() - start) * 1000)
        self.history.append(Message(role="assistant", content=text))

        return AssistantResponse(
            text=text,
            latency_ms=latency_ms,
            model=self.model,
            tokens_used=tokens,
        )

    def reset(self) -> None:
        """Clear conversation history."""
        self.history.clear()

    def get_history(self) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in self.history]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _run_cli() -> None:
    assistant = OSSAssistant()
    print("\n🟢  OSS Assistant (Qwen 2.5) — type 'exit' to quit, 'clear' to reset\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "clear":
            assistant.reset()
            print("[Conversation cleared]\n")
            continue

        result = assistant.chat(user_input)
        print(f"\nQwen 2.5 ({result.latency_ms} ms):\n{result.text}\n")


if __name__ == "__main__":
    _run_cli()
