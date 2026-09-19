"""SSE consumer adapted to `st.write_stream`.

`st.write_stream` wants a generator of strings and returns the concatenation.
The stream also carries one final non-token frame (query_id, citations,
latencies) that the caller needs, so it is captured into a dict the caller owns
rather than yielded.
"""

from collections.abc import Iterator
from typing import Any

from api import client


def token_stream(
    kb_id: str, question: str, retrieval_config_id: str | None, sink: dict[str, Any]
) -> Iterator[str]:
    """Yield tokens; write the final frame into `sink["final"]`, errors into `sink["error"]`."""
    for kind, value in client.stream_chat(kb_id, question, retrieval_config_id):
        if kind == "token":
            yield value
        elif kind == "final":
            sink["final"] = value
        else:
            sink["error"] = value
