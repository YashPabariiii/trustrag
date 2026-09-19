<div align="center">

# TrustRAG

**Production RAG platform with built-in evaluation — every answer scored for
faithfulness, relevance, and hallucination using RAGAs.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Redis](https://img.shields.io/badge/Redis-7-DC382D?logo=redis&logoColor=white)](https://redis.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-vector%20store-FF6B6B)](https://www.trychroma.com/)
[![Celery](https://img.shields.io/badge/Celery-workers-37814A?logo=celery&logoColor=white)](https://docs.celeryq.dev/)
[![RAGAs](https://img.shields.io/badge/RAGAs-evaluation-6E4AFF)](https://docs.ragas.io/)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-F55036)](https://groq.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Prometheus](https://img.shields.io/badge/Prometheus-metrics-E6522C?logo=prometheus&logoColor=white)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-4%20dashboards-F46800?logo=grafana&logoColor=white)](https://grafana.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-manifests-326CE5?logo=kubernetes&logoColor=white)](https://kubernetes.io/)

</div>

---

## The pitch

Every RAG project returns answers. **TrustRAG returns answers you can trust.**
Built-in RAGAs evaluation runs asynchronously on every query — faithfulness (is
it supported by the context?), context relevance (did retrieval find the right
chunks?), answer relevance (does it address the question?), and a hallucination
score. Low-scoring answers are flagged with specific improvement suggestions,
and the flag names the component at fault: a low context-relevance score is a
retrieval defect, not a prompt you should go and rewrite. The Retrieval Lab
lets you A/B test retrieval configurations against real eval metrics — not
vibes.

This is what separates a notebook RAG demo from a production RAG system. The
teams that publish seriously on retrieval — Anthropic and Cohere among them —
publish evaluation methodology alongside it, because the failure mode of a
retrieval system is statistical, silent, and only visible in aggregate. One bad
answer is an anecdote; a faithfulness average sliding from 0.91 to 0.78 over a
fortnight is a defect. TrustRAG makes evaluation
a first-class citizen rather than an afterthought: scores are attributed to the
retrieval configuration that produced them, rolled up into a KB health report,
turned into a named config change, and re-checked by a golden-pair regression
suite after the change lands.

---

## Features

| | |
|---|---|
| ✅ **Hybrid retrieval** | semantic + BM25, merged with Reciprocal Rank Fusion (`k=60`) |
| ✅ **Cross-encoder reranking** | `ms-marco-MiniLM-L-6-v2` over the fused candidates |
| ✅ **Groq Llama 3.3 70B** | grounded prompt, parsed citations, SSE streaming |
| ✅ **RAGAs on every query** | faithfulness · context relevance · answer relevance · hallucination |
| ✅ **Async evaluation** | the answer never waits for its score |
| ✅ **Live eval scores in the chat UI** | four bars and a badge, under the answer |
| ✅ **KB health analytics** | trends, score distribution, worst queries, weak-spot clusters |
| ✅ **Retrieval Lab** | A/B two configs, scored by real RAGAs metrics, winner past a 0.05 margin |
| ✅ **Golden-pair test suites** | regression testing for RAG, with pass rates |
| ✅ **Improvement engine** | recommendations that name the exact field to change, with a working Apply |
| ✅ **Query caching** | Redis, keyed by retrieval config so A/B stays honest |
| ✅ **Multi-tenant + JWT auth** | every read scoped by tenant at the query, not in the app |
| ✅ **Free / Pro tiers** | KB, document and monthly-query ceilings, plan-aware rate limits |
| ✅ **Prometheus + Grafana** | 29 metric families, 4 provisioned dashboards |
| ✅ **Kubernetes manifests** | 3-replica API with an HPA, 2 workers, StatefulSets for state |
| ✅ **CI** | lint, image build, and a real `docker compose` integration run |

---

## Architecture

```
                    ┌──────────────┐            ┌──────────────┐
   browser  ───────▶│  Streamlit   │            │   Grafana    │◀── browser
                    │  :8501       │            │   :3000      │
                    └──────┬───────┘            └──────┬───────┘
                           │ HTTP (JWT)                 │ PromQL
                           ▼                            ▼
                    ┌──────────────────────┐     ┌──────────────┐
   curl / SDK ─────▶│      FastAPI         │◀────│  Prometheus  │
                    │      :8000           │     │   :9090      │
                    │                      │     └──────┬───────┘
                    │  auth · tiers · rate │            │ scrape
                    │  limit · correlation │            │
                    └───┬──────┬───────┬───┘            │
              retrieve  │      │       │ enqueue        │
                        │      │       │                │
        ┌───────────────┘      │       └──────────┐     │
        ▼                      ▼                  ▼     │
 ┌─────────────┐        ┌─────────────┐    ┌────────────┴──┐
 │  ChromaDB   │        │ PostgreSQL  │    │     Redis     │
 │  vectors    │        │ 8 tables    │    │ broker·cache  │
 │  :8001      │        │ :5432       │    │ :6379         │
 └─────────────┘        └─────────────┘    └───────┬───────┘
        ▲                      ▲                   │ consume
        │                      │                   ▼
        │                      │            ┌──────────────┐
        └──────────────────────┴────────────│    Celery    │──▶ :9100/metrics
                 index · evaluate           │    worker    │
                                            └──────┬───────┘
                                                   │
                                                   ▼
                                            ┌──────────────┐
                                            │ Groq  ·  LLM │
                                            │ generate +   │
                                            │ RAGAs judge  │
                                            └──────────────┘

  one question, end to end
  ────────────────────────
  ask ─▶ cache key (tenant+kb+question+CONFIG) ─▶ hit? replay
                                                └▶ miss: retrieve (semantic│bm25│hybrid+RRF)
                                                        ─▶ cross-encoder rerank
                                                        ─▶ Groq generate (or stream)
                                                        ─▶ parse citations
                                                        ─▶ persist query (eval_status=pending)
                                                        ─▶ RETURN THE ANSWER          ◀── user waits here
                                                        ─▶ enqueue evaluation
  worker ─▶ rehydrate FULL chunk text ─▶ RAGAs × 3 ─▶ hallucination = 1 − faithfulness
         ─▶ weighted overall ─▶ flags + suggestions ─▶ evaluations row ─▶ config averages
```

Two processes export metrics. Query, cache and tier counters live in the API;
indexing, evaluation and suite counters live in the worker. Prometheus scrapes
both — a counter belongs to the process that incremented it.

---

## The RAGAs metrics

| Metric | What it measures | Good score | What a bad score means |
|---|---|---|---|
| **Faithfulness** | The share of claims in the answer that are supported by the retrieved chunks | **≥ 0.80** | The model is asserting things the context does not say. Tighten `rerank_top_n`, or shorten the answer |
| **Context relevance** | Whether retrieval surfaced chunks capable of answering the question at all | **≥ 0.70** | A **retrieval** defect. Raise `top_k` or switch to hybrid. Do not touch the prompt |
| **Answer relevance** | Whether the answer addresses the question that was asked | **≥ 0.75** | Grounded but off-target — usually a vague question, or a prompt that permits summarising instead of answering |
| **Hallucination** | `1 − faithfulness`. Derived, never judged separately, so the two can never disagree | **≤ 0.30** | At least one unsupported claim. This is the one that costs you a customer |
| **Overall RAG score** | `0.40·F + 0.35·CR + 0.25·AR`, re-normalised over whichever metrics returned a number | **≥ 0.80** | RAGAs can return NaN for one metric; dividing by the full weight would deflate a good answer into a bad one |

Faithfulness is weighted highest on purpose: an answer that invents a number is
worse than one that is merely off-topic.

### Score interpretation

| Band | Badge | Reading |
|---|---|---|
| **≥ 0.80** | 🟢 **HIGH** | Ship it. Promote this configuration |
| **0.60 – 0.80** | 🟡 **MEDIUM** | Works, but something is leaking. Check which metric is dragging |
| **< 0.60** | 🔴 **LOW** | Do not put this in front of a customer. The flags say why |

The same three bands drive every badge, bar, gauge and Grafana threshold in the
product — a 0.79 never reads green on one screen and amber on another.

### Flags and what they trigger

| Condition | Flag | Suggestion the engine gives you |
|---|---|---|
| faithfulness < 0.70 | Answer contains unsupported claims | Enable reranking, or reduce `max_tokens` |
| context relevance < 0.60 | Retrieved chunks may not be relevant | Increase `top_k`, or switch to hybrid retrieval |
| answer relevance < 0.70 | Answer may not address the question | The question may be too vague — refine the prompt |
| hallucination > 0.30 | Potential hallucination detected | Shorter answers stay closer to the context |

---

## The Retrieval Lab

Tuning a RAG pipeline is normally done by changing `top_k`, re-reading three
answers, and deciding they feel better. The Retrieval Lab replaces that with a
measurement.

```
Config A                          Config B
semantic · top_k=3                hybrid · top_k=8 · rerank top 3
   │                                 │
   └──────── same question ──────────┘
                 │
      both answered, both judged by RAGAs
                 │
   ┌─────────────┴──────────────┐
   │ faithfulness   1.00  1.00  │
   │ context rel    0.75  1.00  │
   │ answer rel     0.42  0.61  │
   │ OVERALL        0.75  0.93  │
   └─────────────┬──────────────┘
                 ▼
     🏆 Config B wins (+0.175 overall)
```

Three rules make the result mean something:

1. **A winner needs a 0.05 margin.** Below that it reports *no meaningful
   difference* rather than dressing up judge noise as a result.
2. **The cache key includes the retrieval config.** Config B can never be
   served Config A's cached answer.
3. **Scores are attributed to the config that produced them.** Every config
   carries its own running averages, so "did that change help?" is a lookup,
   not an argument.

A batch run puts up to 20 questions through both configurations and reports
per-question wins alongside the averaged score. When you promote the winner, the
promotion is logged with the previous config id and its score — the audit trail
for *why did this change?*

**Golden-pair test suites** are the other half: a set of questions with known
good answers, re-run after every configuration change. A step down in pass rate
right after a config activation is a regression, and the Grafana *RAG Health by
KB* dashboard plots those two series on the same screen.

---

## Quick start

```bash
git clone <this repo> && cd trustrag
cp .env.example .env

# Everything: Postgres, Redis, ChromaDB, API, worker, dashboard,
# Prometheus, Grafana.
docker compose up --build -d
docker compose exec api alembic upgrade head

# No Groq key? Add the CI overlay — it swaps in a local stub that speaks the
# same OpenAI-compatible protocol, so every line of TrustRAG runs unchanged.
docker compose -f docker-compose.yml -f docker-compose.ci.yml up --build -d
```

| | |
|---|---|
| Dashboard | <http://localhost:8501> |
| API docs | <http://localhost:8000/docs> |
| Grafana | <http://localhost:3000> — anonymous viewer, TrustRAG folder |
| Prometheus | <http://localhost:9090> |
| Metrics | <http://localhost:8000/metrics> · <http://localhost:9100/metrics> |

Then the guided tour:

```bash
python demo/run_demo.py
```

It registers a tenant, indexes two synthetic policy documents, asks five
questions — four answerable and one deliberately vague — prints the evaluation
table, the KB health report, an A/B comparison and a golden-pair pass rate.

> **Running without Docker?** `pending.md` § 3 has the native setup that every
> verification run in `test_commands.md` actually used.

---

## Sample evaluation output

Real output from `python demo/run_demo.py`, unedited:

```
3. Upload and index the sample documents
──────────────────────────────────────────────────────────────────────
  ai_safety_report.txt         → indexed 16 chunks, 3 pages
  financial_regulations.txt    → indexed 14 chunks, 3 pages

  Indexed 2 docs, 30 chunks

5. Evaluation scores
──────────────────────────────────────────────────────────────────────
  Evaluation runs asynchronously — the answers above were returned
  before any of this existed.

  Question                            Faithful  Context Rel  Answer Rel  Overall   Flags
  ─────────────────────────────────────────────────────────────────────────────────────
  What are the main topics covered?     0.00       0.00        0.00       0.00     Answer contains unsupported claims;
                                                                                   Retrieved chunks may not be relevant;
                                                                                   Answer may not address the question;
                                                                                   Potential hallucination detected
                                        💡 Consider increasing top_k or switching to hybrid retrieval
  What compliance requirements apply?   1.00       1.00        0.62       0.91     Answer may not address the question
  Summarize the key findings.           1.00       1.00        0.39       0.85     Answer may not address the question
  What actions are recommended?         1.00       1.00        0.37       0.84     Answer may not address the question
  Tell me everything.                   0.00       0.00        0.00       0.00     Answer contains unsupported claims;
                                                                                   Retrieved chunks may not be relevant;
                                                                                   Answer may not address the question;
                                                                                   Potential hallucination detected

6. Knowledge base health
──────────────────────────────────────────────────────────────────────
  health score  43.9 / 100   ·  5 evaluations
  avg scores    faithfulness 0.60  context 0.60  answer 0.28  overall 0.52
  hallucination rate 40%

  Recommendations
  · increase_top_k        top_k: 5 → 8
    Low retrieval relevance (0.60) — the generator is being handed chunks
    that do not answer the question.
  · tighten_rerank_top_n  rerank_top_n: 3 → 2
  · reduce_generation_tokens  max_tokens: 1024 → 512
  · refine_prompt         system_prompt: None → None

8. Golden-pair regression suite
──────────────────────────────────────────────────────────────────────
  auto-generated from answers scoring above 0.85: 2 pairs
  2/2 passed (100%)  ·  average overall score  0.87
```

Read that honestly: **"Tell me everything" is meant to score 0** — a question
with no answerable shape should be flagged, and a platform that scores it well
is not measuring anything. *"What are the main topics covered?"* also scores 0,
but that one is an artefact of the offline stub judge used when no
`GROQ_API_KEY` is present; a real model answers it. `pending.md` § 1.2 explains
exactly which numbers the stub distorts and which it does not.

That is the point of the whole product: it will tell you when it is doing badly.

---

## Grafana

Four dashboards, provisioned from `grafana/dashboards/*.json` — the JSON in git
is the source of truth, and UI edits are overwritten on reload.

| Dashboard | Panels | What it answers |
|---|---|---|
| **RAG Quality Overview** | 8 | Are the answers any good? The three metric trends on one axis, a per-KB hallucination gauge, the overall-score distribution as a heatmap, and flagged-answer volume split by flag type |
| **Query Performance** | 13 | Queries/min total and per KB, p50/p95/p99 for every pipeline stage, cache hit rate, stream-vs-sync split, eval completion rate, tier-limit rejections |
| **RAG Health by KB** | 7 | Per-KB average scores as a heatmap, hallucination rate, config-activation timeline, and test-suite pass rate — the last two on the same screen so a regression is visible |
| **System Health** | 12 | Celery queue depth, Chroma collections and vectors, database pool, Redis memory, documents/hour, and a chunks-per-document distribution |

```
TrustRAG — RAG Quality Overview                                  last 6h ⟳ 30s
┌──────────────────────┬──────────────────────┬──────────────────────┐
│ Average faithfulness │ Avg context relevance│ Average answer relev.│
│ 1.0 ┤    ╭──────     │ 1.0 ┤ ╭─╮   ╭────    │ 1.0 ┤                │
│ 0.8 ┤╌╌╌╱╌╌╌╌╌╌╌ 🟢  │ 0.8 ┤╌╯╌╰───╯╌╌╌ 🟢  │ 0.8 ┤╌╌╌╌╌╌╌╌╌╌╌     │
│ 0.6 ┤╱             │ 0.6 ┤              │ 0.6 ┤    ╭───╮  🟡 │
│ 0.0 └──────────────  │ 0.0 └──────────────  │ 0.0 └────╯───╰──    │
├──────────────────────┴───────┬──────────────┴──────────────────────┤
│ Hallucination rate per KB    │ Overall RAG score distribution      │
│      ╭───────╮               │ 1.0 ┤░░░░▓▓▓████████████            │
│     ╱  0.05   ╲   🟢         │ 0.8 ┤░░▓▓████████▓▓░░░░             │
│    │   ●       │             │ 0.6 ┤░░░░▓▓░░░░░░                   │
│     ╲_________╱              │ 0.4 ┤░░░░                           │
│      0        1              │ 0.2 ┤░░                             │
├──────────────────────────────┴─────────────────────────────────────┤
│ Flagged answers per minute, by flag                                │
│  ▁▁▂▂▃▃▂▂▁▁  low_context      ▁▁▁▂▂▁▁  low_faithfulness            │
│  ▁▂▃▄▃▂▁▁▁▁  low_answer_rel   ▁▁▁▁▁▁▁  hallucination               │
└────────────────────────────────────────────────────────────────────┘
```

All 44 PromQL expressions across the four dashboards were executed against a
live Prometheus scraping the real API and worker: **44 returned data, 0
errored** (`test_commands.md` § S6.3). `ci/check_dashboards.py` additionally
asserts that every metric a panel queries is one the application actually
defines, so a renamed metric fails CI rather than silently emptying a panel.

---

## API reference

| Method | Path | Notes |
|---|---|---|
| `POST` | `/v1/auth/register` | → `{tenant_id, api_key}`. The key is shown once; only its SHA-256 hash is stored |
| `POST` | `/v1/auth/token` | api_key → `{access_token, expires_in}` |
| `GET` | `/v1/auth/me` | current tenant + usage counters |
| `POST` `GET` | `/v1/knowledge-bases` | create (Chroma collection + default config) / list |
| `GET` `DELETE` | `/v1/knowledge-bases/{id}` | detail with live counts / drop the collection then cascade |
| `POST` `GET` | `/v1/knowledge-bases/{id}/documents` | multipart upload ≤ 50 MB → `202` + Celery task / paginated list |
| `GET` `DELETE` | `/v1/documents/{id}` | metadata + live `chunk_count` — poll this for indexing progress |
| **`POST`** | **`/v1/chat/{kb_id}`** | **ask.** `stream:true` → SSE. Returns `eval_status: "pending"` and never waits |
| `GET` | `/v1/chat/{kb_id}/history` | filters: `cached`, `eval_status`, `date_from`, `date_to` |
| `GET` | `/v1/queries/{id}` | question, answer, citations, context, config snapshot, evaluation |
| **`GET`** | **`/v1/queries/{id}/evaluation`** | **the scores.** Or `{status: "pending"}` — poll it |
| `GET` | `/v1/knowledge-bases/{id}/evaluations` | eval history; `min_score`, `max_score`, `low_score_only`, dates |
| **`GET`** | **`/v1/knowledge-bases/{id}/health`** | **health score, trend, worst queries, weak spots, field-level recommendations** |
| `GET` | `/v1/knowledge-bases/{id}/eval-stats` | aggregates + daily score history |
| `POST` `GET` | `/v1/knowledge-bases/{id}/configs` | create (always inactive) / list with running averages |
| `PATCH` `POST` | `…/configs/{id}/activate` · `…/promote` | exactly one active config per KB, enforced by a partial unique index |
| **`POST`** | **`/v1/knowledge-bases/{id}/ab-test`** | **one question, two configs, both scored, winner past a 0.05 margin** |
| `POST` | `…/ab-test/batch` | up to 20 questions, averaged, with per-question win counts |
| `POST` `GET` | `/v1/knowledge-bases/{id}/test-suites` | create / list golden-pair suites |
| `POST` | `…/test-suites/generate` | build a suite from past answers scoring > 0.85 |
| `POST` `GET` | `/v1/test-suites/{id}/run` · `/results` | `202` + Celery run / pass rate, averages, per-pair detail |
| `GET` | `/health/live` · `/health/ready` | no I/O, always answers · `{db, redis, chroma}`, `503` if any is down |
| `GET` | `/metrics` | Prometheus, no auth |

Errors are always `{"error": {"code", "message", "detail"}, "correlation_id"}`.
Every request carries an `X-Correlation-ID`, echoed back and bound into every
log line for that request.

### Limits

| | Free | Pro |
|---|---|---|
| Knowledge bases | 2 | ∞ |
| Documents per KB | 10 | ∞ |
| Queries per month | 50 | ∞ |
| Requests | 20/min | 200/min |

---

## Repository layout

```
app/                       FastAPI service
├── core/                  auth · logging · middleware · exceptions · metrics
├── models/                8 SQLAlchemy tables
├── processing/            parser · chunker · embedder · indexer
├── rag/                   retriever · reranker · llm · cache · pipeline
├── evaluation/            ragas_evaluator · ab_tester · test_suite_runner
│                          · improvement_engine
├── workers/               celery_app · tasks
├── routes/                auth · health · knowledge_bases · documents · chat
│                          · evaluations · retrieval_configs · test_suites
└── migrations/            alembic

dashboard/                 Streamlit product UI — HTTP only, no DB
├── api/client.py          httpx wrapper, error envelope, SSE consumer
├── auth/session.py        login/register, JWT + expiry, plan badge
├── components/            score_badge · eval_scores · citation_card
│                          · stream_handler
└── pages/                 knowledge_bases · chat · eval_analytics
                           · retrieval_lab · test_suites

grafana/                   4 provisioned dashboards + datasource
monitoring/                prometheus.yml (scrapes API and worker)
k8s/                       namespace · configmap · secret template
                           api (deployment/hpa/service/ingress) · worker
                           · postgres · chromadb · redis
ci/                        integration test · metrics check · dashboard check
                           · the offline Groq/RAGAs stub
demo/                      run_demo.py + two synthetic policy documents
```

## Documentation

| File | What's in it |
|---|---|
| **`pending.md`** | **Everything written but not proven, and everything knowingly missing. Read this before trusting the repo** |
| `decision.md` | Every non-obvious choice, with the reason and the cost if it's wrong |
| `flow.md` | End-to-end code flow, function by function |
| `test_commands.md` | The e2e runs and their real output, plus the acceptance matrix |

---

## Author

**Yash Pabari** 
