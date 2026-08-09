"""Server-Sent Events streaming helper — Skill 31 / 32.

Wraps any token iterator (sync or async) into a `text/event-stream`
StreamingResponse compatible with the frontend's EventSource/fetch consumer.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Iterable, Union

from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

TokenIterator = Union[Iterable[str], AsyncIterator[str]]

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def format_sse_event(data: dict) -> str:
    """Format a single SSE `data:` line from a JSON-serializable dict."""
    return f"data: {json.dumps(data)}\n\n"


async def _to_sse(token_iter: TokenIterator) -> AsyncIterator[str]:
    try:
        if hasattr(token_iter, "__aiter__"):
            async for token in token_iter:  # type: ignore[union-attr]
                yield format_sse_event({"token": token})
        else:
            for token in token_iter:  # type: ignore[union-attr]
                yield format_sse_event({"token": token})
    except Exception as e:
        logger.error(f"SSE stream failed mid-stream: {e}")
        yield format_sse_event({"error": str(e)})
    finally:
        yield "data: [DONE]\n\n"


def sse_stream(token_iter: TokenIterator) -> StreamingResponse:
    """Wrap a token iterator (sync or async) as a StreamingResponse."""
    return StreamingResponse(_to_sse(token_iter), media_type="text/event-stream", headers=SSE_HEADERS)
