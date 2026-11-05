from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any

import httpx
import pathway as pw

try:
    from pathway.internals import udfs
except (ImportError, AttributeError):
    try:
        from pathway import udfs  # type: ignore[no-redef]
    except (ImportError, AttributeError):
        udfs = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

UDFBase = pw.UDF if hasattr(pw, "UDF") else object


def _extract_text(data: Any) -> str:
    """Extract string text from a string, dict, or pw.Json wrapper."""
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    if hasattr(data, "value"):
        return _extract_text(data.value)
    if isinstance(data, dict):
        if "text" in data:
            return _extract_text(data["text"])
        return str(data)
    return str(data)


def _extract_score_from_response(data: Any) -> float:
    """
    Parse relevance score from various reranker response formats:
      - TEI / Infinity: [{"index": 0, "score": 0.85}]
      - Cohere / Jina:  {"results": [{"index": 0, "relevance_score": 0.85}]}
      - Simple list:    [0.85] or [{"score": 0.85}]
    """
    if isinstance(data, list) and len(data) > 0:
        first = data[0]
        if isinstance(first, dict):
            if "score" in first:
                return float(first["score"])
            if "relevance_score" in first:
                return float(first["relevance_score"])
        elif isinstance(first, (int, float)):
            return float(first)

    if isinstance(data, dict):
        if "results" in data and isinstance(data["results"], list) and len(data["results"]) > 0:
            first = data["results"][0]
            if isinstance(first, dict):
                if "relevance_score" in first:
                    return float(first["relevance_score"])
                if "score" in first:
                    return float(first["score"])
        if "score" in data:
            return float(data["score"])

    raise ValueError(f"Unable to parse reranker score from server response: {data!r}")


class TEIReranker(UDFBase):
    """
    Asynchronous external Cross-Encoder Reranker connector.

    Designed for Hugging Face Text Embeddings Inference (TEI), Infinity, or
    compatible reranker services serving models like:
      - Alibaba-NLP/gte-multilingual-reranker-base
      - BAAI/bge-reranker-large
      - cross-encoder/ms-marco-MiniLM-L-6-v2

    Sends asynchronous requests to the `/rerank` HTTP endpoint, keeping heavy
    neural network inference out of the Pathway streaming process.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:8080/rerank",
        api_key: str | None = None,
        truncate: bool = True,
        raw_scores: bool = False,
        capacity: int | None = 20,
        timeout: float = 30.0,
        retry_strategy: Any | None = None,
        cache_strategy: Any | None = None,
        headers: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        # Normalize endpoint URL to ensure trailing /rerank if user passed base URL
        endpoint_clean = endpoint.rstrip("/")
        if not endpoint_clean.endswith("/rerank"):
            endpoint_clean = f"{endpoint_clean}/rerank"
        self.endpoint = endpoint_clean

        self.api_key = api_key
        self.truncate = truncate
        self.raw_scores = raw_scores
        self.timeout = timeout
        self.custom_headers = dict(headers or {})
        self._injected_client = client
        self._client: httpx.AsyncClient | None = client

        # Initialize Pathway UDF with async executor if available
        if hasattr(pw, "UDF"):
            executor = None
            retry = retry_strategy
            if retry is None and hasattr(pw, "udfs") and hasattr(pw.udfs, "ExponentialBackoffRetryStrategy"):
                retry = pw.udfs.ExponentialBackoffRetryStrategy()

            if hasattr(pw, "udfs") and hasattr(pw.udfs, "async_executor"):
                try:
                    executor = pw.udfs.async_executor(capacity=capacity, retry_strategy=retry)
                except TypeError:
                    executor = pw.udfs.async_executor(capacity=capacity)
            elif udfs is not None and hasattr(udfs, "async_executor"):
                try:
                    executor = udfs.async_executor(capacity=capacity, retry_strategy=retry)
                except TypeError:
                    executor = udfs.async_executor(capacity=capacity)

            super().__init__(
                executor=executor,
                cache_strategy=cache_strategy,
                return_type=float,
            )
        else:
            super().__init__()

    def _get_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.custom_headers)
        return headers

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._injected_client is not None:
            return self._injected_client
        if self._client is None or getattr(self._client, "is_closed", False):
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        """Close the internal HTTP client."""
        if self._client is not None and not self._client.is_closed and self._client is not self._injected_client:
            await self._client.aclose()
            self._client = None

    async def __wrapped__(self, doc: Any, query: Any, **kwargs: Any) -> float:
        """
        Pointwise scoring invoked by Pathway's UDF execution or async streaming answerer.

        Args:
            doc: Candidate document chunk or text.
            query: User query or search prompt.

        Returns:
            float: Relevance score computed by the external cross-encoder.
        """
        doc_text = _extract_text(doc)
        query_text = _extract_text(query)

        if not doc_text.strip() or not query_text.strip():
            return 0.0

        client = self._ensure_client()
        headers = self._get_headers()

        payload = {
            "query": query_text,
            "texts": [doc_text],
            "truncate": kwargs.get("truncate", self.truncate),
            "raw_scores": kwargs.get("raw_scores", self.raw_scores),
        }

        try:
            resp = await client.post(self.endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return _extract_score_from_response(data)
        except httpx.HTTPStatusError as e:
            logger.error("TEI reranker HTTP %d error from %s: %s", e.response.status_code, self.endpoint, e.response.text)
            raise
        except Exception as e:
            logger.error("TEI reranker request failed for endpoint %s: %s", self.endpoint, e)
            raise

    async def rerank_batch(
        self,
        query: str,
        texts: list[str],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Batch rerank multiple candidate texts in a single HTTP request to TEI.

        Returns a list of dictionaries with {"index": int, "score": float}, sorted
        descending by score.
        """
        if not texts:
            return []

        client = self._ensure_client()
        headers = self._get_headers()

        payload = {
            "query": str(query),
            "texts": [str(t) for t in texts],
            "truncate": kwargs.get("truncate", self.truncate),
            "raw_scores": kwargs.get("raw_scores", self.raw_scores),
        }

        resp = await client.post(self.endpoint, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        results: list[dict[str, Any]] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "index" in item and "score" in item:
                    results.append({"index": int(item["index"]), "score": float(item["score"])})
        elif isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
            for item in data["results"]:
                idx = item.get("index", 0)
                score = item.get("relevance_score", item.get("score", 0.0))
                results.append({"index": int(idx), "score": float(score)})

        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    def __call__(self, doc: Any, query: Any, **kwargs: Any) -> Any:
        """
        Apply UDF on Pathway table ColumnExpressions, or run synchronously if invoked directly
        on raw Python values.
        """
        if hasattr(pw, "UDF") and isinstance(self, pw.UDF):
            # Check if arguments are Pathway ColumnExpressions
            is_pw_expr = hasattr(doc, "_is_column_expression") or hasattr(query, "_is_column_expression")
            if is_pw_expr or not isinstance(doc, (str, dict)) or not isinstance(query, (str, dict)):
                return super().__call__(doc, query, **kwargs)

        # Fallback for direct non-streaming invocation on plain values
        score_coro = self.__wrapped__(doc, query, **kwargs)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # If already inside an async loop, schedule a task or return coroutine
            return score_coro
        return asyncio.run(score_coro)
