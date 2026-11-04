from __future__ import annotations

import base64
import io
import json
import os
import wave
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from connectors.openai_audio_chat import (
    OpenAIAudioChat,
    OpenAILiveChat,
    _extract_prompt_and_system,
    _format_openai_messages,
    _prepare_messages,
)


def test_openai_audio_chat_init():
    chat = OpenAIAudioChat(
        model="gpt-4o-audio-preview",
        api_key="test-key",
        voice="echo",
        audio_format="wav",
        return_audio=False,
    )
    assert chat.model == "gpt-4o-audio-preview"
    assert chat.voice == "echo"
    assert chat.voice_name == "echo"
    assert chat.audio_format == "wav"
    assert chat.return_audio is False
    assert chat.supports_audio is True
    assert chat._accepts_call_arg("return_audio") is True
    assert chat._accepts_call_arg("model") is True
    assert chat._accepts_call_arg("voice") is True
    assert chat._accepts_call_arg("voice_name") is True
    assert chat._accepts_call_arg("unknown_arg") is False


def test_voice_name_alias():
    chat = OpenAIAudioChat(
        api_key="test-key",
        voice_name="shimmer",
    )
    assert chat.voice == "shimmer"
    assert chat.voice_name == "shimmer"


def test_openai_live_chat_alias():
    assert OpenAILiveChat is OpenAIAudioChat


def test_prepare_messages():
    # String input
    msgs = _prepare_messages("Hello world")
    assert msgs == [{"role": "user", "content": "Hello world"}]

    # Single dict
    single = {"role": "user", "content": "Query"}
    assert _prepare_messages(single) == [single]

    # List of dicts
    sample = [{"role": "user", "content": "Query"}]
    assert _prepare_messages(sample) == sample


def test_format_openai_messages():
    msgs = [
        {"role": "user", "content": [{"text": "Hello"}, {"text": "world"}]}
    ]
    formatted = _format_openai_messages(msgs, system_instruction="Be brief.")
    assert len(formatted) == 2
    assert formatted[0] == {"role": "system", "content": "Be brief."}
    assert formatted[1] == {"role": "user", "content": "Hello world"}


def test_extract_prompt_and_system():
    msgs = [
        {"role": "system", "content": "You are a Minecraft assistant."},
        {"role": "user", "content": "How do I craft a pickaxe?"},
    ]
    prompt, system = _extract_prompt_and_system(msgs)
    assert prompt == "How do I craft a pickaxe?"
    assert system == "You are a Minecraft assistant."


def test_pcm_to_wav_conversion():
    sample_rate = 24000
    pcm_bytes = b"\x00\x00" * sample_rate

    wav_bytes = OpenAIAudioChat._pcm_to_wav(pcm_bytes, sample_rate=sample_rate)

    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[8:12] == b"WAVE"

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == sample_rate
        assert wf.getnframes() == sample_rate


def test_api_key_resolution():
    # OpenAI.com endpoint requires key
    chat = OpenAIAudioChat(api_key=None, api_base="https://api.openai.com/v1")
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            chat._resolve_api_key()

    # Local endpoint allows local fallback key
    local_chat = OpenAIAudioChat(
        api_key=None, api_base="http://localhost:1234/v1"
    )
    assert local_chat._resolve_api_key() == "local-no-key"

    # Environment variable resolution
    with patch.dict(os.environ, {"OPENAI_API_KEY": "env-secret-key"}):
        assert chat._resolve_api_key() == "env-secret-key"


@pytest.mark.asyncio
async def test_wrapped_returns_text_when_return_audio_false():
    chat = OpenAIAudioChat(api_key="fake-key", return_audio=False)

    with patch.object(
        chat,
        "_call_rest_non_streaming",
        new=AsyncMock(return_value=("Standard text answer.", None, 24000)),
    ):
        result = await chat.__wrapped__("Hello", return_audio=False)
        assert result == "Standard text answer."


@pytest.mark.asyncio
async def test_wrapped_returns_audio_dict_when_return_audio_true():
    chat = OpenAIAudioChat(api_key="fake-key", return_audio=False)

    fake_wav_b64 = "UklGRiQAAABXQVZF"

    with patch.object(
        chat,
        "_call_rest_non_streaming",
        new=AsyncMock(
            return_value=("Spoken answer.", fake_wav_b64, 24000)
        ),
    ):
        result = await chat.__wrapped__("Hello with audio", return_audio=True)
        assert isinstance(result, dict)
        assert result["response"] == "Spoken answer."
        assert result["text"] == "Spoken answer."
        assert result["audio"] == fake_wav_b64
        assert result["audio_mime_type"] == "audio/wav"


@pytest.mark.asyncio
async def test_call_rest_non_streaming_success():
    chat = OpenAIAudioChat(api_key="test-key", audio_format="wav")

    fake_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Crafting recipe for torch.",
                    "audio": {
                        "id": "audio_123",
                        "transcript": "Crafting recipe for torch.",
                        "data": base64.b64encode(b"RIFF....WAVEdata").decode("ascii"),
                    },
                }
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = fake_response
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)):
        transcript, audio_b64, _ = await chat._call_rest_non_streaming(
            messages_formatted=[{"role": "user", "content": "torch"}],
            model="gpt-4o-audio-preview",
            api_key="test-key",
            api_base="https://api.openai.com/v1",
            voice="alloy",
            return_audio=True,
            audio_format="wav",
            temperature=0.2,
            tts_model=None,
        )

        assert transcript == "Crafting recipe for torch."
        assert audio_b64 is not None


@pytest.mark.asyncio
async def test_call_rest_streaming():
    chat = OpenAIAudioChat(api_key="test-key", audio_format="wav")

    sse_lines = [
        b'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n',
        b'data: {"choices":[{"delta":{"audio":{"transcript":"world!","data":"AAAA"}}}]}\n\n',
        b'data: [DONE]\n\n',
    ]

    class MockStreamContext:
        async def __aenter__(self):
            mock_stream_resp = MagicMock()
            mock_stream_resp.status_code = 200
            mock_stream_resp.raise_for_status = MagicMock()

            async def mock_aiter_lines():
                for line in sse_lines:
                    yield line.decode("utf-8")

            mock_stream_resp.aiter_lines = mock_aiter_lines
            return mock_stream_resp

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    received_chunks = []

    def on_chunk(chunk):
        received_chunks.append(chunk)

    with patch("httpx.AsyncClient.stream", return_value=MockStreamContext()):
        transcript, audio_b64, _ = await chat._call_rest_streaming(
            messages_formatted=[{"role": "user", "content": "Hi"}],
            model="gpt-4o-audio-preview",
            api_key="test-key",
            api_base="https://api.openai.com/v1",
            voice="alloy",
            return_audio=True,
            audio_format="wav",
            temperature=0.2,
            tts_model=None,
            chunk_callback=on_chunk,
        )

        assert transcript == "Hello world!"
        assert any(c.get("type") == "text" for c in received_chunks)
        assert any(c.get("type") == "audio" for c in received_chunks)
        assert audio_b64 is not None


@pytest.mark.asyncio
async def test_tts_speech_synthesis():
    chat = OpenAIAudioChat(api_key="test-key", tts_model="tts-1", audio_format="wav")

    pcm_samples = b"\x00\x00" * 2400
    mock_tts_resp = MagicMock()
    mock_tts_resp.content = pcm_samples
    mock_tts_resp.is_success = True
    mock_tts_resp.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_tts_resp)):
        b64 = await chat._synthesize_speech(
            text="Testing speech synthesis",
            api_key="test-key",
            api_base="https://api.openai.com/v1",
            voice="alloy",
            audio_format="wav",
            tts_model="tts-1",
        )
        assert b64 is not None
        decoded = base64.b64decode(b64)
        assert decoded[:4] == b"RIFF"


@pytest.mark.asyncio
async def test_stream_generator():
    chat = OpenAIAudioChat(api_key="test-key", return_audio=True)

    fake_wav_b64 = "UklGRiQAAABXQVZF"

    async def mock_wrapped(messages, return_audio=False, **kwargs):
        callback = kwargs.get("chunk_callback")
        if callback:
            await callback({"type": "text", "delta": "Streamed text"})
            if return_audio:
                await callback(
                    {
                        "type": "audio",
                        "data": "AAAA",
                        "pcm_bytes": b"\x00\x00",
                        "mime_type": "audio/wav",
                    }
                )
        return {
            "response": "Streamed text",
            "text": "Streamed text",
            "audio": fake_wav_b64 if return_audio else None,
            "audio_mime_type": "audio/wav" if return_audio else None,
        }

    with patch.object(chat, "__wrapped__", side_effect=mock_wrapped):
        events = []
        async for event in chat.stream("Hello streaming", return_audio=True):
            events.append(event)

        event_types = [e["type"] for e in events]
        assert "text" in event_types
        assert "audio" in event_types
        assert "done" in event_types

        done_event = next(e for e in events if e["type"] == "done")
        assert done_event["response"] == "Streamed text"
        assert done_event["audio"] == fake_wav_b64


@pytest.mark.asyncio
async def test_realtime_websocket():
    chat = OpenAIAudioChat(
        api_key="test-key", transport="websocket", return_audio=True
    )

    fake_events = [
        json.dumps({"type": "response.text.delta", "delta": "Hello from "}),
        json.dumps({"type": "response.audio_transcript.delta", "delta": "realtime!"}),
        json.dumps(
            {
                "type": "response.audio.delta",
                "delta": base64.b64encode(b"\x00\x00" * 100).decode("ascii"),
            }
        ),
        json.dumps({"type": "response.done"}),
    ]

    mock_ws = AsyncMock()
    mock_ws.recv = AsyncMock(side_effect=fake_events)
    mock_ws.send = AsyncMock()

    class MockConnectContext:
        async def __aenter__(self):
            return mock_ws

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch("websockets.connect", return_value=MockConnectContext()):
        transcript, audio_b64, _ = await chat._call_realtime_websocket(
            prompt="Hello",
            system_instruction="Be brief",
            model="gpt-4o-realtime-preview",
            api_key="test-key",
            api_base="https://api.openai.com/v1",
            voice="alloy",
            return_audio=True,
            audio_format="wav",
            temperature=0.2,
        )

        assert transcript == "Hello from realtime!"
        assert audio_b64 is not None
        decoded = base64.b64decode(audio_b64)
        assert decoded[:4] == b"RIFF"


@pytest.mark.asyncio
async def test_realtime_websocket_fallback_to_rest():
    chat = OpenAIAudioChat(
        api_key="test-key", transport="websocket", return_audio=False
    )

    with patch.object(
        chat,
        "_call_realtime_websocket",
        side_effect=ConnectionError("WebSocket connection refused"),
    ), patch.object(
        chat,
        "_call_rest_non_streaming",
        new=AsyncMock(return_value=("Fallback response", None, 24000)),
    ):
        result = await chat.__wrapped__("Hello", return_audio=False)
        assert result == "Fallback response"
