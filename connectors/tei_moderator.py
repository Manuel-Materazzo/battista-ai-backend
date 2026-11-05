from __future__ import annotations

import asyncio
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

DEFAULT_TOXIC_LABELS = {
    "toxic",
    "toxicity",
    "toxic_comment",
    "hate",
    "hatespeech",
    "hate_speech",
    "offensive",
    "abuse",
    "abusive",
    "threat",
    "insult",
    "label_1",
    "1",
}

DEFAULT_NEUTRAL_LABELS = {
    "neutral",
    "non-toxic",
    "nontoxic",
    "not_toxic",
    "normal",
    "clean",
    "nothate",
    "not_hate",
    "neither",
    "label_0",
    "0",
}


def _extract_text(data: Any) -> str:
    """Extract string text from a string, dict, or pw.Json wrapper."""
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    if hasattr(data, "value"):
        return _extract_text(data.value)
    if isinstance(data, dict):
        if "prompt" in data:
            return _extract_text(data["prompt"])
        if "text" in data:
            return _extract_text(data["text"])
        return str(data)
    return str(data)


def parse_toxicity_response(
    data: Any,
    toxic_labels: set[str] | None = None,
    threshold: float = 0.5,
) -> bool:
    """
    Evaluate toxicity from TEI classification predictions.

    TEI /predict returns:
      - [{"label": "toxic", "score": 0.95}, {"label": "neutral", "score": 0.05}]
      - or nested list for batch: [[{"label": "...", "score": ...}]]
    """
    if data is None:
        return False
    if isinstance(data, bool):
        return data

    predictions = data
    # Handle single element batch: [[{"label": ...}]] -> [{"label": ...}]
    if isinstance(predictions, list) and len(predictions) > 0 and isinstance(predictions[0], list):
        predictions = predictions[0]

    # Handle dict with results / predictions list
    if isinstance(predictions, dict):
        if "predictions" in predictions and isinstance(predictions["predictions"], list):
            predictions = predictions["predictions"]
        elif "results" in predictions and isinstance(predictions["results"], list):
            predictions = predictions["results"]
        elif "label" in predictions and "score" in predictions:
            predictions = [predictions]
        elif "toxic" in predictions:
            return bool(predictions["toxic"])

    if not isinstance(predictions, list) or len(predictions) == 0:
        return False

    parsed_preds: list[tuple[str, float]] = []
    for item in predictions:
        if isinstance(item, dict) and "label" in item and "score" in item:
            lbl = str(item["label"]).strip().lower()
            try:
                score = float(item["score"])
                parsed_preds.append((lbl, score))
            except (ValueError, TypeError):
                continue

    if not parsed_preds:
        return False

    targets = toxic_labels if toxic_labels is not None else DEFAULT_TOXIC_LABELS

    # Check if any recognized toxic label meets or exceeds threshold
    for label, score in parsed_preds:
        if label in targets and score >= threshold:
            return True

    # If explicit toxic_labels was provided and none exceeded threshold, return False
    if toxic_labels is not None:
        return False

    # Auto-detection fallback: Check top prediction (highest confidence score)
    top_label, top_score = max(parsed_preds, key=lambda x: x[1])
    if top_label in targets and top_score >= threshold:
        return True
    if top_label in DEFAULT_NEUTRAL_LABELS:
        return False

    return False


class TEIModerator(UDFBase):
    """
    Asynchronous external sequence classification moderator connector.

    Connects to Hugging Face Text Embeddings Inference (TEI), Infinity, or
    compatible sequence classification servers serving toxicity / moderation models:
      - textdetox/xlmr-large-toxicity-classifier-v2
      - unitary/multilingual-toxic-xlm-roberta
      - facebook/roberta-hate-speech-dynabench-r4-target

    Sends asynchronous requests to the `/predict` HTTP endpoint, offloading heavy
    PyTorch/Transformers execution from the local Python backend.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:8080/predict",
        toxic_label: str | list[str] | None = None,
        threshold: float = 0.5,
        truncate: bool = True,
        raw_scores: bool = False,
        capacity: int | None = 20,
        timeout: float = 10.0,
        api_key: str | None = None,
        retry_strategy: Any | None = None,
        cache_strategy: Any | None = None,
        headers: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        # Normalize endpoint URL to ensure trailing /predict if user passed base URL
        endpoint_clean = endpoint.rstrip("/")
        if not endpoint_clean.endswith("/predict"):
            endpoint_clean = f"{endpoint_clean}/predict"
        self.endpoint = endpoint_clean

        if isinstance(toxic_label, str):
            self.toxic_labels: set[str] | None = {toxic_label.strip().lower()}
        elif isinstance(toxic_label, (list, tuple, set)):
            self.toxic_labels = {str(lbl).strip().lower() for lbl in toxic_label}
        else:
            self.toxic_labels = None

        self.threshold = threshold
        self.truncate = truncate
        self.raw_scores = raw_scores
        self.timeout = timeout
        self.api_key = api_key
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
                return_type=bool,
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

    async def __wrapped__(self, prompt: Any, **kwargs: Any) -> bool:
        """
        Evaluate toxicity of a prompt via external TEI /predict endpoint.

        Args:
            prompt: Text to classify (str, dict, or pw.Json).

        Returns:
            bool: True if toxic, False if neutral/safe.
        """
        prompt_text = _extract_text(prompt)
        if not prompt_text.strip():
            return False

        client = self._ensure_client()
        headers = self._get_headers()

        payload = {
            "inputs": prompt_text,
            "truncate": kwargs.get("truncate", self.truncate),
            "raw_scores": kwargs.get("raw_scores", self.raw_scores),
        }

        try:
            resp = await client.post(self.endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return parse_toxicity_response(
                data,
                toxic_labels=self.toxic_labels,
                threshold=kwargs.get("threshold", self.threshold),
            )
        except httpx.HTTPStatusError as e:
            logger.error("TEI moderator HTTP %d error from %s: %s", e.response.status_code, self.endpoint, e.response.text)
            raise
        except Exception as e:
            logger.error("TEI moderator request failed for endpoint %s: %s", self.endpoint, e)
            raise

    async def predict_raw(self, prompt: str, **kwargs: Any) -> list[dict[str, Any]]:
        """
        Send a classification request to TEI and return the raw list of label predictions.
        """
        prompt_text = _extract_text(prompt)
        if not prompt_text.strip():
            return []

        client = self._ensure_client()
        headers = self._get_headers()
        payload = {
            "inputs": prompt_text,
            "truncate": kwargs.get("truncate", self.truncate),
            "raw_scores": kwargs.get("raw_scores", self.raw_scores),
        }
        resp = await client.post(self.endpoint, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            return data[0]
        return data if isinstance(data, list) else [data]

    def __call__(self, prompt: Any, **kwargs: Any) -> Any:
        """
        Apply UDF on Pathway table ColumnExpressions, or run synchronously if invoked directly
        on raw Python values.
        """
        if hasattr(pw, "UDF") and isinstance(self, pw.UDF):
            is_pw_expr = hasattr(prompt, "_is_column_expression")
            if is_pw_expr or not isinstance(prompt, (str, dict)):
                return super().__call__(prompt, **kwargs)

        coro = self.__wrapped__(prompt, **kwargs)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            return coro
        return asyncio.run(coro)
