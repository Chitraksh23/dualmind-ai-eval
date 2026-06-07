"""
frontier_assistant.py
---------------------
Frontier Model Personal Assistant — Claude Sonnet 4 via Anthropic API.

Supports:
  • Multi-turn conversations
  • Short-term conversational memory (full history passed each call)
  • Basic assistant behaviour; safety alignment is built into the model
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import anthropic


FRONTIER_MODEL = os.getenv("FRONTIER_MODEL", "claude-sonnet-4-20250514")

SYSTEM_PROMPT = (
    "You are a helpful, accurate, and safe AI assistant.\n"
    "Your goals:\n"
    "1. Answer questions clearly and honestly\n"
    "2. Acknowledge uncertainty when you don't know something\n"
    "3. Decline requests that could cause harm\n"
    "4. Maintain context across the conversation\n"
    "5. Be concise but complete"
)


@dataclass
class AssistantResponse:
    text: str
    latency_ms: int
    model: str
    input_tokens:  Optional[int] = None
    output_tokens: Optional[int] = None
    error: Optional[str] = None

    @property
    def cost_usd(self) -> float:
        """Approximate cost (Claude Sonnet 4 pricing)."""
        return round(
            (self.input_tokens  or 0) * 3e-6 +   # $3  / 1M input tokens
            (self.output_tokens or 0) * 15e-6,    # $15 / 1M output tokens
            6,
        )


class FrontierAssistant:
    """
    Claude Sonnet 4 personal assistant.
    Full conversation history is passed with every request for multi-turn memory.
    """

    def __init__(self) -> None:
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.history: list[dict] = []   # {"role": str, "content": str}
        self.model = FRONTIER_MODEL

    def chat(self, user_message: str) -> AssistantResponse:
        """Send a message, get a response, and append both to history."""
        self.history.append({"role": "user", "content": user_message})

        start = time.time()
        text: str
        input_tokens:  Optional[int]
        output_tokens: Optional[int]
        error: Optional[str]

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=800,
                system=SYSTEM_PROMPT,
                messages=self.history,
            )
            text          = response.content[0].text
            input_tokens  = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            error         = None

        except anthropic.APIStatusError as exc:
            text          = f"[API Error {exc.status_code}: {exc.message}]"
            input_tokens  = None
            output_tokens = None
            error         = str(exc)

        except Exception as exc:
            text          = f"[Error: {exc}]"
            input_tokens  = None
            output_tokens = None
            error         = str(exc)

        latency_ms = int((time.time() - start) * 1000)
        self.history.append({"role": "assistant", "content": text})

        return AssistantResponse(
            text=text,
            latency_ms=latency_ms,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error=error,
        )

    def reset(self) -> None:
        """Clear conversation history."""
        self.history.clear()

    def get_history(self) -> list[dict]:
        return list(self.history)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _run_cli() -> None:
    assistant = FrontierAssistant()
    print("\n🟣  Frontier Assistant (Claude Sonnet 4) — type 'exit' to quit, 'clear' to reset\n")

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
        print(f"\nClaude ({result.latency_ms} ms, ~${result.cost_usd}):\n{result.text}\n")


if __name__ == "__main__":
    _run_cli()
