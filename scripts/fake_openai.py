"""Deterministic OpenAI-compatible chat server for CPU smoke tests of the job pipeline.

Generator prompts get a cited answer built from the first document; judge prompts get
well-formed verdicts. Run: python scripts/fake_openai.py --port 8000
"""

from __future__ import annotations

import argparse
import re

import uvicorn
from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health() -> dict[str, str]:
    return {"ok": "1"}


@app.post("/v1/chat/completions")
def chat(body: dict) -> dict:
    msgs = body["messages"]
    system = msgs[0]["content"] if msgs[0]["role"] == "system" else ""
    user = msgs[-1]["content"]
    if "Statements:" in user:
        n = len(re.findall(r"(?m)^\d+: ", user.split("Statements:")[-1]))
        text = "\n".join(f"{i}: {'SUPPORTED' if i % 2 else 'UNSUPPORTED'}" for i in range(1, n + 1))
    elif "Reference answer:" in user:
        text = "CORRECT"
    else:
        m = re.search(r"\[1\] Title: (.*)\n(.*)", user)
        first = m.group(2).split(". ")[0].rstrip(".") if m else "Unknown"
        text = f"{first} [1][2]. This sentence has no support [3]."
        if "<answer>" in system:
            text = f"<reasoning>doc 1</reasoning><answer>{text}</answer>"
    return {
        "model": body["model"],
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": len(user) // 4, "completion_tokens": len(text) // 4},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
