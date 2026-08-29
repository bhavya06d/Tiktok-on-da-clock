"""Person 2 (part 1) — LLM client + token accounting.

Token totals feed the Feasibility score, so count everything.
Swap the provider freely; keep the interface: complete(prompt) -> (text, usage).
Also includes MockLLM so Persons 1 & 3 can develop without an API key.
"""
from __future__ import annotations

from pathlib import Path


class LLMClient:
    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 8000):
        import anthropic  # pip install anthropic
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model
        self.max_tokens = max_tokens
        self._in = 0
        self._out = 0

    def complete(self, prompt: str) -> tuple[str, dict]:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        self._in += resp.usage.input_tokens
        self._out += resp.usage.output_tokens
        text = "".join(b.text for b in resp.content if b.type == "text")
        return text, {"input": resp.usage.input_tokens,
                      "output": resp.usage.output_tokens}

    def total_tokens(self) -> dict:
        return {"input": self._in, "output": self._out,
                "total": self._in + self._out}


class MockLLM:
    """Returns a canned solution so the harness can be built & tested offline.
    Point it at ml/reference_baseline.py (or any file honoring the contract)."""

    def __init__(self, canned_solution_path: str):
        self.code = Path(canned_solution_path).read_text()
        self.calls = 0

    def complete(self, prompt: str) -> tuple[str, dict]:
        self.calls += 1
        reply = ("HYPOTHESIS: mock run — reproduce the official baseline to "
                 "verify the harness end-to-end.\n"
                 f"```python\n{self.code}\n```")
        return reply, {"input": 0, "output": 0}

    def total_tokens(self) -> dict:
        return {"input": 0, "output": 0, "total": 0}
