from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest

from connectors.tei_reranker import (
    TEIReranker,
    _extract_score_from_response,
    _extract_text,
)


def test_endpoint_normalization():
    r1 = TEIReranker(endpoint="http://localhost:8080")
    assert r1.endpoint == "http://localhost:8080/rerank"

    r2 = TEIReranker(endpoint="http://localhost:8080/")
    assert r2.endpoint == "http://localhost:8080/rerank"

    r3 = TEIReranker(endpoint="http://localhost:8080/rerank")
    assert r3.endpoint == "http://localhost:8080/rerank"


def test_header_generation():
    reranker = TEIReranker(
        endpoint="http://tei:8080/rerank",
        api_key="secret-token",
        headers={"X-Custom": "test-val"},
    )
    headers = reranker._get_headers()
    assert headers["Authorization"] == "Bearer secret-token"
    assert headers["Content-Type"] == "application/json"
    assert headers["X-Custom"] == "test-val"


def test_extract_text():
    assert _extract_text(None) == ""
    assert _extract_text("simple text") == "simple text"
    assert _extract_text({"text": "dict text"}) == "dict text"

    class MockPwJson:
        def __init__(self, value):
            self.value = value

    assert _extract_text(MockPwJson("json text")) == "json text"
    assert _extract_text(MockPwJson({"text": "nested json text"})) == "nested json text"


def test_extract_score_from_response():
    # TEI / Infinity standard format
    assert _extract_score_from_response([{"index": 0, "score": 0.875}]) == 0.875

    # Jina / Cohere format
    assert _extract_score_from_response({"results": [{"index": 0, "relevance_score": 0.912}]}) == 0.912

    # Float list format
    assert _extract_score_from_response([0.75]) == 0.75

    # Malformed response
    with pytest.raises(ValueError):
        _extract_score_from_response({"unexpected": 123})


@pytest.mark.asyncio
async def test_wrapped_single_scoring():
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [{"index": 0, "score": 0.945}]
    mock_resp.raise_for_status.return_value = None
    mock_client.post.return_value = mock_resp

    reranker = TEIReranker(
        endpoint="http://localhost:8080/rerank",
        client=mock_client,
    )

    score = await reranker.__wrapped__(
        doc="Pathway is a reactive streaming data processing framework.",
        query="What is Pathway?",
    )

    assert score == pytest.approx(0.945)
    mock_client.post.assert_awaited_once()
    call_args = mock_client.post.await_args
    assert call_args.args[0] == "http://localhost:8080/rerank"
    assert call_args.kwargs["json"]["query"] == "What is Pathway?"
    assert call_args.kwargs["json"]["texts"] == ["Pathway is a reactive streaming data processing framework."]


@pytest.mark.asyncio
async def test_empty_input_returns_zero():
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    reranker = TEIReranker(client=mock_client)

    score_empty_doc = await reranker.__wrapped__(doc="", query="What is Pathway?")
    assert score_empty_doc == 0.0

    score_empty_query = await reranker.__wrapped__(doc="Doc text", query="   ")
    assert score_empty_query == 0.0

    mock_client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_rerank_batch():
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [
        {"index": 0, "score": 0.25},
        {"index": 1, "score": 0.88},
        {"index": 2, "score": 0.42},
    ]
    mock_resp.raise_for_status.return_value = None
    mock_client.post.return_value = mock_resp

    reranker = TEIReranker(client=mock_client)

    results = await reranker.rerank_batch(
        query="Test query",
        texts=["doc 0", "doc 1", "doc 2"],
    )

    # Verify results are sorted descending by score
    assert len(results) == 3
    assert results[0] == {"index": 1, "score": 0.88}
    assert results[1] == {"index": 2, "score": 0.42}
    assert results[2] == {"index": 0, "score": 0.25}


@pytest.mark.asyncio
async def test_rag_question_answerer_compatibility():
    """Verify that TEIReranker's __wrapped__ is a coroutine function for RAG question answerer."""
    reranker = TEIReranker()
    score_fn = getattr(reranker, "__wrapped__", reranker)
    assert inspect.iscoroutinefunction(score_fn)
