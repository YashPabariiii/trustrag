"""Dashboard settings. One place that knows the API address and the score bands."""

import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

# Generation and evaluation are LLM-bound, so these are deliberately generous.
REQUEST_TIMEOUT = 60.0
STREAM_TIMEOUT = 300.0
LONG_TIMEOUT = 1800.0  # A/B batch: 20 questions x 2 configs x (generate + judge)

# Score bands. Every badge, bar and metric card reads these, so a 0.79 never
# renders green on one page and amber on another.
SCORE_HIGH = 0.80
SCORE_MEDIUM = 0.60

GREEN = "#1a9850"
AMBER = "#e8a33d"
RED = "#d73027"
GREY = "#9aa0a6"

DOC_STATUS_BADGE = {
    "queued": "\U0001f7e1 queued",
    "processing": "\U0001f535 processing",
    "indexed": "\U0001f7e2 indexed",
    "failed": "\U0001f534 failed",
}

EVAL_POLL_SECONDS = 3
EVAL_POLL_MAX = 20  # 60 s, then stop and let the user refresh

SUITE_POLL_SECONDS = 5
SUITE_POLL_MAX = 120  # 10 min; a 20-pair suite is estimated at 4

DOC_POLL_SECONDS = 2
DOC_POLL_MAX = 60  # 2 min per file

HISTORY_SIZE = 15
MAX_BATCH_QUESTIONS = 20
MAX_GOLDEN_PAIRS = 20
TREND_DAYS = 30

# Config fields a KB health recommendation can actually write. The API also
# recommends `max_tokens` and `system_prompt`, which are not columns on
# retrieval_configs — those get no Apply button.
APPLICABLE_REC_FIELDS = {
    "top_k",
    "retrieval_type",
    "rerank_enabled",
    "rerank_top_n",
    "chunk_size",
    "chunk_overlap",
}
