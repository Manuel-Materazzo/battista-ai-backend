import base64
import json
import io
import wave
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from connectors.gemini_live_chat import (
    GeminiLiveChat,
    _extract_prompt_and_system,
    _prepare_messages,
)


def test_gemini_live_chat_init():
    chat = GeminiLiveChat(
        model="gemini-3.8-live",
        api_key="test-key",
        voice_name="Charon",
        audio_format="wav",
        return_audio=False,
    )
    assert chat.model == "gemini-3.8-live"
    assert chat.voice_name == "Charon"
    assert chat.audio_format == "wav"
    assert chat.return_audio is False
    assert chat.supports_audio is True
    assert chat._accepts_call_arg("return_audio") is True
    assert chat._accepts_call_arg("model") is True
    assert chat._accepts_call_arg("unknown_arg") is False


def test_prepare_messages():
    # String input
    msgs = _prepare_messages("Hello world")
    assert msgs == [{"role": "user", "content": "Hello world"}]

    # List of dicts
    sample = [{"role": "user", "content": "Query"}]
    assert _prepare_messages(sample) == sample


def test_extract_prompt_and_system():
    msgs = [
        {"role": "system", "content": "You are a Minecraft helper."},
        {"role": "user", "content": "How do I make a sword?"},
    ]
    prompt, system = _extract_prompt_and_system(msgs)
    assert prompt == "How do I make a sword?"
    assert system == "You are a Minecraft helper."


def test_pcm_to_wav_conversion():
    # 1 second of 24kHz 16-bit mono silence (48,000 bytes)
    sample_rate = 24000
    pcm_bytes = b"\x00\x00" * sample_rate

    wav_bytes = GeminiLiveChat._pcm_to_wav(pcm_bytes, sample_rate=sample_rate)

    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[8:12] == b"WAVE"

    # Verify with wave module
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == sample_rate
        assert wf.getnframes() == sample_rate


@pytest.mark.asyncio
async def test_wrapped_returns_text_when_return_audio_false():
    chat = GeminiLiveChat(api_key="fake-key", return_audio=False)

    with patch.object(
        chat,
        "_call_live_websocket",
        new=AsyncMock(return_value=("This is the text response.", None, 24000)),
    ):
        result = await chat.__wrapped__("Hello", return_audio=False)
        assert result == "This is the text response."


@pytest.mark.asyncio
async def test_wrapped_returns_audio_dict_when_return_audio_true():
    chat = GeminiLiveChat(api_key="fake-key", return_audio=False)

    fake_wav_b64 = "UklGRiQAAABXQVZF"
    with patch.object(
        chat,
        "_call_live_websocket",
        new=AsyncMock(return_value=("Spoken answer", fake_wav_b64, 24000)),
    ):
        result = await chat.__wrapped__("Hello", return_audio=True)
        assert isinstance(result, dict)
        assert result["response"] == "Spoken answer"
        assert result["text"] == "Spoken answer"
        assert result["audio"] == fake_wav_b64
        assert result["audio_mime_type"] == "audio/wav"


@pytest.mark.asyncio
async def test_websocket_parsing_protocol():
    chat = GeminiLiveChat(api_key="fake-key")

    # Simulate raw 16-bit PCM chunk
    raw_pcm = b"\x10\x00" * 100
    pcm_b64 = base64.b64encode(raw_pcm).decode("ascii")

    server_messages = [
        json.dumps({"setupComplete": {}}),
        json.dumps({
            "serverContent": {
                "outputAudioTranscription": {"text": "Hello "},
                "modelTurn": {
                    "parts": [
                        {"text": "world!"},
                        {
                            "inlineData": {
                                "mimeType": "audio/pcm;rate=24000",
                                "data": pcm_b64,
                            }
                        },
                    ]
                },
                "turnComplete": True,
            }
        }),
    ]

    mock_ws = AsyncMock()
    mock_ws.send = AsyncMock()
    mock_ws.recv = AsyncMock(side_effect=server_messages)

    class MockConnect:
        async def __aenter__(self):
            return mock_ws

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch("websockets.connect", return_value=MockConnect()):
        transcript, audio_b64, rate = await chat._call_live_websocket(
            prompt="Hi",
            model="gemini-3.8-live",
            api_key="fake",
            api_base="generativelanguage.googleapis.com",
            voice_name="Puck",
            return_audio=True,
            audio_format="wav",
            system_instruction=None,
            temperature=None,
        )

        assert transcript == "Hello world!"
        assert audio_b64 is not None
        assert rate == 24000

        # Decode generated wav and verify
        wav_data = base64.b64decode(audio_b64)
        assert wav_data[:4] == b"RIFF"


@pytest.mark.asyncio
async def test_rest_transport():
    chat = GeminiLiveChat(api_key="fake-key", transport="rest", return_audio=True)

    fake_pcm = b"\x20\x00" * 50
    pcm_b64 = base64.b64encode(fake_pcm).decode("ascii")

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "REST answer"},
                        {
                            "inlineData": {
                                "mimeType": "audio/pcm;rate=24000",
                                "data": pcm_b64,
                            }
                        },
                    ]
                }
            }
        ]
    }
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp

    class MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return mock_client

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch("httpx.AsyncClient", MockAsyncClient):
        result = await chat.__wrapped__("REST query", return_audio=True)
        assert isinstance(result, dict)
        assert result["response"] == "REST answer"
        assert result["audio"] != ""
        assert result["audio_mime_type"] == "audio/wav"

