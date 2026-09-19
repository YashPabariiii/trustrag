"""Local OpenAI-compatible stand-in for Groq, for CI and for offline demos.

CI has no GROQ_API_KEY, and an integration test that depends on a paid
third-party API is a test that fails for reasons that are not your fault. This
serves the same OpenAI-compatible surface the real Groq SDK talks to, so every
line of TrustRAG's own code runs unchanged; only the model is fake.

Point the app at it with `GROQ_BASE_URL` (decision.md D-51, D-70):
  * POST /openai/v1/chat/completions, JSON and SSE
  * answers ONLY from the context it is handed, so citations are real
  * doubles as a RAGAS judge: it recognises each RAGAS prompt by the shape of
    its `input:` block and returns conforming output, deciding entailment by
    lexical/numeric overlap.

It measures rather than fabricates, which is what makes the acceptance criteria
testable. It is not a language model.
"""

import json
import re
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()

STOPWORDS = {
    "the", "a", "an", "of", "for", "to", "in", "on", "and", "or", "is", "was",
    "were", "are", "be", "what", "which", "how", "did", "does", "do", "at",
    "by", "with", "that", "this", "it", "its", "as", "from", "than", "over",
}
WORD_RE = re.compile(r"[a-z0-9][a-z0-9.%$,-]*")
NUMBER_RE = re.compile(r"\d[\d,.]*")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def tokens(text: str) -> set[str]:
    return {w.strip(".,") for w in WORD_RE.findall(text.lower())} - STOPWORDS


def numbers(text: str) -> set[str]:
    return {n.rstrip(".,") for n in NUMBER_RE.findall(text)}


def overlap(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / len(ta) if ta else 0.0


# --- RAG generation -------------------------------------------------------

CONTEXT_BLOCK_RE = re.compile(
    r"\[(\d+)\] Source: (.+?) \| Page: (\d+)\n(.*?)(?=\n\n---\n\n|\n\nQuestion:)",
    re.DOTALL,
)


def answer_from_context(prompt: str) -> str:
    question_match = re.search(r"\nQuestion: (.+?)\n", prompt)
    question = question_match.group(1) if question_match else ""

    best = None
    for _, filename, page, text in CONTEXT_BLOCK_RE.findall(prompt):
        for sentence in SENTENCE_RE.split(text.replace("\n", " ")):
            if len(sentence.strip()) < 20:
                continue
            score = overlap(question, sentence)
            if best is None or score > best[0]:
                best = (score, sentence.strip(), filename.strip(), int(page))

    if best is None or best[0] == 0.0:
        return "I could not find this in the provided documents."
    _, sentence, filename, page = best
    return f"{sentence} [Doc: {filename}, Page: {page}]"


# --- RAGAS judging --------------------------------------------------------

INPUT_RE = re.compile(r"\ninput: (\{.*\})\nOutput:", re.DOTALL)
RELEVANCE_RE = re.compile(r"### Question: (.*?)\n\n### Context: (.*?)\n\nDo not try", re.DOTALL)


def judge(prompt: str) -> str | None:
    # ContextRelevance is a plain-text prompt that wants "0", "1" or "2".
    relevance = RELEVANCE_RE.search(prompt)
    if relevance and "Relevance score" in prompt:
        question, context = relevance.group(1), relevance.group(2)
        ratio = overlap(question, context)
        return "2" if ratio >= 0.5 else "1" if ratio > 0.0 else "0"

    match = INPUT_RE.search(prompt)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None

    # Faithfulness step 1: break the answer into standalone statements.
    if "answer" in data and "question" in data:
        answer = re.sub(r"\[Doc:[^\]]*\]", "", data["answer"])
        statements = [s.strip() for s in SENTENCE_RE.split(answer) if len(s.strip()) > 5]
        return json.dumps({"statements": statements or [answer.strip()]})

    # Faithfulness step 2: is each statement entailed by the context?
    if "context" in data and "statements" in data:
        context = data["context"]
        context_numbers = numbers(context)
        judged = []
        for statement in data["statements"]:
            words_ok = overlap(statement, context) >= 0.6
            statement_numbers = numbers(statement)
            numbers_ok = statement_numbers.issubset(context_numbers)
            verdict = 1 if (words_ok and numbers_ok) else 0
            judged.append(
                {
                    "statement": statement,
                    "reason": (
                        "supported by the context"
                        if verdict
                        else "terms or figures absent from the context"
                    ),
                    "verdict": verdict,
                }
            )
        return json.dumps({"statements": judged})

    # ResponseRelevancy: reconstruct the question the answer replies to.
    if set(data) == {"response"}:
        response = re.sub(r"\[Doc:[^\]]*\]", "", data["response"]).strip()
        noncommittal = int("could not find" in response.lower())
        return json.dumps(
            {"question": f"What does the context say about {response[:80]}?",
             "noncommittal": noncommittal}
        )

    return None


def reply_for(messages: list[dict]) -> str:
    prompt = "\n".join(m.get("content") or "" for m in messages)
    judged = judge(prompt)
    if judged is not None:
        return judged
    return answer_from_context(prompt)


# --- OpenAI-compatible surface -------------------------------------------


def envelope(model: str, content: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 128,
            "completion_tokens": max(len(content.split()), 1),
            "total_tokens": 128 + max(len(content.split()), 1),
        },
    }


@app.post("/openai/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "stub-model")
    content = reply_for(body.get("messages") or [])

    if not body.get("stream"):
        return envelope(model, content)

    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    words = content.split(" ")

    def frames():
        for word in words:
            payload = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"content": word + " "}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(payload)}\n\n"
        final = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            # groq-python reads usage off the last chunk under x_groq (D-45).
            "x_groq": {
                "id": chunk_id,
                "usage": {
                    "prompt_tokens": 128,
                    "completion_tokens": len(words),
                    "total_tokens": 128 + len(words),
                },
            },
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(frames(), media_type="text/event-stream")


@app.get("/openai/v1/models")
async def models():
    return {"object": "list", "data": [{"id": "stub-model", "object": "model"}]}
