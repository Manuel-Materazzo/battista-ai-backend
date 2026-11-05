from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import httpx
import pytest

from connectors.tei_moderator import (
    TEIModerator,
    parse_toxicity_response,
    _extract_text,
)
from processors.chat_moderator import ChatModerator


def test_endpoint_normalization():
    m1 = TEIModerator(endpoint="http://localhost:8080")
    assert m1.endpoint == "http://localhost:8080/predict"

    m2 = TEIModerator(endpoint="http://localhost:8080/")
    assert m2.endpoint == "http://localhost:8080/predict"

    m3 = TEIModerator(endpoint="http://localhost:8080/predict")
    assert m3.endpoint == "http://localhost:8080/predict"


def test_extract_text():
    assert _extract_text(None) == ""
    assert _extract_text("safe prompt") == "safe prompt"
    assert _extract_text({"prompt": "dict prompt"}) == "dict prompt"
    assert _extract_text({"text": "dict text"}) == "dict text"

    class MockPwJson:
        def __init__(self, value):
            self.value = value

    assert _extract_text(MockPwJson("json prompt")) == "json prompt"
    assert _extract_text(MockPwJson({"prompt": "nested json"})) == "nested json"


def test_parse_toxicity_response():
    # 1. Standard toxic label
    toxic_data = [{"label": "toxic", "score": 0.95}, {"label": "neutral", "score": 0.05}]
    assert parse_toxicity_response(toxic_data) is True

    # 2. Standard neutral label
    neutral_data = [{"label": "neutral", "score": 0.88}, {"label": "toxic", "score": 0.12}]
    assert parse_toxicity_response(neutral_data) is False

    # 3. Model with LABEL_1 (toxic) / LABEL_0 (neutral)
    label1_data = [{"label": "LABEL_1", "score": 0.91}, {"label": "LABEL_0", "score": 0.09}]
    assert parse_toxicity_response(label1_data) is True

    label0_data = [{"label": "LABEL_0", "score": 0.94}, {"label": "LABEL_1", "score": 0.06}]
    assert parse_toxicity_response(label0_data) is False

    # 4. Explicit toxic label with threshold
    custom_data = [{"label": "offensive", "score": 0.65}, {"label": "clean", "score": 0.35}]
    assert parse_toxicity_response(custom_data, toxic_labels={"offensive"}, threshold=0.7) is False
    assert parse_toxicity_response(custom_data, toxic_labels={"offensive"}, threshold=0.6) is True

    # 5. Empty or unexpected data
    assert parse_toxicity_response([]) is False
    assert parse_toxicity_response(None) is False


@pytest.mark.asyncio
async def test_wrapped_moderator_call_toxic():
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [
        {"label": "toxic", "score": 0.98},
        {"label": "neutral", "score": 0.02},
    ]
    mock_resp.raise_for_status.return_value = None
    mock_client.post.return_value = mock_resp

    moderator = TEIModerator(
        endpoint="http://localhost:8080/predict",
        client=mock_client,
    )

    is_toxic = await moderator.__wrapped__("You are terrible!")
    assert is_toxic is True

    mock_client.post.assert_awaited_once()
    call_args = mock_client.post.await_args
    assert call_args.args[0] == "http://localhost:8080/predict"
    assert call_args.kwargs["json"]["inputs"] == "You are terrible!"


@pytest.mark.asyncio
async def test_wrapped_moderator_call_safe():
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [
        {"label": "neutral", "score": 0.99},
        {"label": "toxic", "score": 0.01},
    ]
    mock_resp.raise_for_status.return_value = None
    mock_client.post.return_value = mock_resp

    moderator = TEIModerator(
        endpoint="http://localhost:8080/predict",
        client=mock_client,
    )

    is_toxic = await moderator.__wrapped__("Hello, how can I craft a sword?")
    assert is_toxic is False


@pytest.mark.asyncio
async def test_empty_prompt_is_not_toxic():
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    moderator = TEIModerator(client=mock_client)

    assert await moderator.__wrapped__("") is False
    assert await moderator.__wrapped__("   ") is False
    mock_client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_predict_raw():
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    expected = [
        {"label": "neutral", "score": 0.85},
        {"label": "toxic", "score": 0.15},
    ]
    mock_resp.json.return_value = expected
    mock_resp.raise_for_status.return_value = None
    mock_client.post.return_value = mock_resp

    moderator = TEIModerator(client=mock_client)
    raw = await moderator.predict_raw("Sample prompt")
    assert raw == expected


def test_chat_moderator_compatibility():
    moderator = TEIModerator(endpoint="http://tei-moderator:8080/predict")
    chat_mod = ChatModerator(moderator=moderator)
    assert chat_mod.moderator is moderator
