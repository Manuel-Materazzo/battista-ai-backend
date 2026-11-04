from __future__ import annotations

import asyncio
import base64
import copy
import inspect
import io
import json
import logging
import os
import wave
from collections.abc import AsyncGenerator, Callable
from typing import Any, Iterable, Literal

import pathway as pw

try:
    from pathway.internals import udfs
except (ImportError, AttributeError):
    try:
        from pathway import udfs  # type: ignore[no-redef]
    except (ImportError, AttributeError):
        udfs = None  # type: ignore[assignment]

try:
    from pathway.xpacks.llm.llms import BaseChat
except (ImportError, AttributeError):
    # Fallback if BaseChat is not directly exportable
    class BaseChat(pw.UDF if hasattr(pw, "UDF") else object):  # type: ignore[no-redef]
        kwargs: dict[str, Any]

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if hasattr(pw, "UDF"):
                super().__init__(*args, **kwargs)
            self.kwargs = {}

        @property
        def model(self) -> str | None:
            return self.kwargs.get("model")


logger = logging.getLogger(__name__)


def _prepare_messages(messages: Any) -> list[dict]:
    """Normalize input messages into a list of role/content dictionaries."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]

    has_pw_json = hasattr(pw, "Json")
    if has_pw_json and isinstance(messages, pw.Json):
        messages_as_list: Iterable[dict | Any] = messages.as_list()
    elif isinstance(messages, (list, tuple)):
        messages_as_list = messages
    elif isinstance(messages, dict):
        return [messages]
    else:
        # Fallback if messages is another object or expression
        try:
            messages_as_list = list(messages)
        except TypeError:
            return [{"role": "user", "content": str(messages)}]

    messages_decoded: list[dict] = [
        message.as_dict() if has_pw_json and isinstance(message, pw.Json) else message
        for message in messages_as_list
    ]
    return messages_decoded


def _format_openai_messages(
    messages_decoded: list[dict],
    system_instruction: str | None = None,
) -> list[dict[str, str]]:
    """Format normalized messages into standard OpenAI role/content message list."""
    formatted: list[dict[str, str]] = []
    has_system = any(msg.get("role") == "system" for msg in messages_decoded)

    if system_instruction and not has_system:
        formatted.append({"role": "system", "content": system_instruction})

    for msg in messages_decoded:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = [
                b.get("text", "") for b in content if isinstance(b, dict) and "text" in b
            ]
            content = " ".join(text_parts) if text_parts else str(content)
        formatted.append({"role": role, "content": str(content)})

    return formatted


def _extract_prompt_and_system(
    messages_decoded: list[dict],
) -> tuple[str, str | None]:
    """Extract prompt text and optional system instruction from messages."""
    system_parts: list[str] = []
    user_parts: list[str] = []

    for msg in messages_decoded:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_blocks = [
                b.get("text", "") for b in content if isinstance(b, dict) and "text" in b
            ]
            content = " ".join(text_blocks)

        if role == "system":
            system_parts.append(str(content))
        else:
            user_parts.append(str(content))

    system_instruction = "\n".join(system_parts) if system_parts else None
    prompt = "\n".join(user_parts)
    return prompt, system_instruction


class OpenAIAudioChat(BaseChat):
    """
    Pathway Chat connector for OpenAI and OpenAI-compatible audio models.

    Supports native audio generation via standard OpenAI Chat Completions API
    (e.g. ``gpt-4o-audio-preview``, ``gpt-4o-mini-audio-preview``) using the
    ``audio`` modality and voice selection, as well as optional Realtime
    WebSocket streaming and text-to-speech (TTS) fallback.

    When ``return_audio`` is True, returns synthesized speech in WAV (or raw PCM/MP3)
    along with response text and transcripts.
    """

    def __init__(
        self,
        model: str = "gpt-4o-audio-preview",
        api_key: str | None = None,
        api_base: str = "https://api.openai.com/v1",
        voice: str = "alloy",
        voice_name: str | None = None,
        audio_format: Literal["wav", "mp3", "flac", "opus", "pcm16", "pcm"] = "wav",
        return_audio: bool = False,
        transport: Literal["rest", "websocket"] = "rest",
        system_instruction: str | None = None,
        temperature: float | None = None,
        timeout: float = 60.0,
        tts_model: str | None = None,
        capacity: int | None = None,
        retry_strategy: Any | None = None,
        cache_strategy: Any | None = None,
        *,
        async_mode: Literal["batch_async", "fully_async"] = "batch_async",
        **kwargs: Any,
    ) -> None:
        if hasattr(pw, "udfs"):
            retry = retry_strategy or getattr(
                pw.udfs, "ExponentialBackoffRetryStrategy", lambda: None
            )()
            if async_mode == "fully_async":
                executor = pw.udfs.fully_async_executor(
                    capacity=capacity, retry_strategy=retry
                )
            else:
                executor = pw.udfs.async_executor(
                    capacity=capacity, retry_strategy=retry
                )
            super().__init__(
                executor=executor, cache_strategy=cache_strategy, return_type=Any
            )
        else:
            super().__init__()

        effective_voice = voice_name if voice_name is not None else voice

        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.voice = effective_voice
        self.voice_name = effective_voice
        self.audio_format = audio_format
        self.return_audio = return_audio
        self.transport = transport
        self.system_instruction = system_instruction
        self.temperature = temperature
        self.timeout = timeout
        self.tts_model = tts_model
        self.supports_audio = True

        self.kwargs: dict[str, Any] = {
            "model": model,
            "api_key": api_key,
            "api_base": self.api_base,
            "voice": effective_voice,
            "voice_name": effective_voice,
            "audio_format": audio_format,
            "return_audio": return_audio,
            "transport": transport,
            "tts_model": tts_model,
            **kwargs,
        }

    @property
    def model(self) -> str | None:
        return self.kwargs.get("model", "gpt-4o-audio-preview")

    def _accepts_call_arg(self, arg_name: str) -> bool:
        """Indicate supported query-time keyword arguments."""
        return arg_name in {
            "model",
            "api_key",
            "api_base",
            "return_audio",
            "voice",
            "voice_name",
            "temperature",
            "audio_format",
            "system_instruction",
            "tts_model",
            "transport",
            "verbose",
        }

    @staticmethod
    def _pcm_to_wav(
        pcm_data: bytes,
        sample_rate: int = 24000,
        channels: int = 1,
        sample_width: int = 2,
    ) -> bytes:
        """Convert raw 16-bit linear PCM audio into a standard RIFF WAV byte stream."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)
        return buffer.getvalue()

    def _resolve_api_key(self, key_override: str | None = None) -> str:
        """Resolve API key from override, instance, or environment variables."""
        api_key = key_override or self.api_key
        if not api_key or api_key == "fake":
            env_key = os.environ.get("OPENAI_API_KEY")
            if env_key:
                api_key = env_key

        is_local_endpoint = any(
            local in self.api_base
            for local in (
                "localhost",
                "127.0.0.1",
                "host.docker.internal",
                "0.0.0.0",
            )
        )
        if not api_key:
            if is_local_endpoint:
                api_key = "local-no-key"
            else:
                raise ValueError(
                    "OpenAI API key is required. Set the OPENAI_API_KEY environment variable "
                    "or provide a valid api_key to OpenAIAudioChat."
                )

        return api_key

    def _get_mime_type(self, audio_format: str) -> str:
        """Get standard MIME type for audio format."""
        if audio_format == "wav":
            return "audio/wav"
        if audio_format in ("pcm", "pcm16"):
            return "audio/pcm;rate=24000"
        if audio_format == "mp3":
            return "audio/mpeg"
        if audio_format == "opus":
            return "audio/opus"
        if audio_format == "flac":
            return "audio/flac"
        return f"audio/{audio_format}"

    async def _call_rest_non_streaming(
        self,
        messages_formatted: list[dict[str, str]],
        model: str,
        api_key: str,
        api_base: str,
        voice: str,
        return_audio: bool,
        audio_format: str,
        temperature: float | None,
        tts_model: str | None,
    ) -> tuple[str, str | None, int]:
        """Call standard OpenAI Chat Completions REST API non-streaming."""
        import httpx

        url = f"{api_base.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        openai_audio_format = "pcm16" if audio_format == "pcm" else audio_format
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages_formatted,
        }
        if temperature is not None:
            payload["temperature"] = temperature

        if return_audio:
            payload["modalities"] = ["text", "audio"]
            payload["audio"] = {
                "voice": voice,
                "format": openai_audio_format,
            }

        sample_rate = 24000

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)

            # If request failed with 400 when requesting audio, handle fallback
            if resp.status_code == 400 and return_audio and "modalities" in payload:
                logger.warning(
                    "OpenAI audio modality rejected (status 400). Retrying text-only chat."
                )
                fallback_payload = copy.deepcopy(payload)
                fallback_payload.pop("modalities", None)
                fallback_payload.pop("audio", None)
                resp = await client.post(url, headers=headers, json=fallback_payload)

            resp.raise_for_status()
            data = resp.json()

        choices = data.get("choices", [])
        if not choices:
            return "", None, sample_rate

        message = choices[0].get("message", {})
        content = message.get("content") or ""
        audio_obj = message.get("audio") or {}

        audio_b64 = audio_obj.get("data")
        transcript = audio_obj.get("transcript") or content

        if return_audio and not audio_b64 and (transcript or content) and tts_model:
            # TTS fallback via /v1/audio/speech
            logger.info("Generating synthesized audio using TTS model: %s", tts_model)
            audio_b64 = await self._synthesize_speech(
                text=transcript or content,
                api_key=api_key,
                api_base=api_base,
                voice=voice,
                audio_format=audio_format,
                tts_model=tts_model,
            )

        if return_audio and audio_b64:
            raw_bytes = base64.b64decode(audio_b64)
            if audio_format == "wav" and not raw_bytes.startswith(b"RIFF"):
                wav_bytes = self._pcm_to_wav(raw_bytes, sample_rate=sample_rate)
                audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

        return transcript, audio_b64, sample_rate

    async def _synthesize_speech(
        self,
        text: str,
        api_key: str,
        api_base: str,
        voice: str,
        audio_format: str,
        tts_model: str,
    ) -> str | None:
        """Synthesize speech using OpenAI /v1/audio/speech endpoint."""
        import httpx

        url = f"{api_base.rstrip('/')}/audio/speech"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        format_param = "pcm" if audio_format in ("pcm", "pcm16") else audio_format
        payload = {
            "model": tts_model,
            "input": text,
            "voice": voice,
            "response_format": format_param,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                audio_bytes = resp.content
                if audio_format == "wav" and not audio_bytes.startswith(b"RIFF"):
                    audio_bytes = self._pcm_to_wav(audio_bytes, sample_rate=24000)
                return base64.b64encode(audio_bytes).decode("ascii")
        except Exception as e:
            logger.warning("TTS audio speech synthesis failed: %s", e)
            return None

    async def _call_rest_streaming(
        self,
        messages_formatted: list[dict[str, str]],
        model: str,
        api_key: str,
        api_base: str,
        voice: str,
        return_audio: bool,
        audio_format: str,
        temperature: float | None,
        tts_model: str | None,
        chunk_callback: Callable[[dict[str, Any]], Any] | None = None,
    ) -> tuple[str, str | None, int]:
        """Call standard OpenAI Chat Completions REST API with SSE streaming."""
        import httpx

        url = f"{api_base.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        openai_audio_format = "pcm16" if audio_format == "pcm" else audio_format
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages_formatted,
            "stream": True,
        }
        if temperature is not None:
            payload["temperature"] = temperature

        if return_audio:
            payload["modalities"] = ["text", "audio"]
            payload["audio"] = {
                "voice": voice,
                "format": openai_audio_format,
            }

        sample_rate = 24000
        transcript_parts: list[str] = []
        audio_pcm_chunks: list[bytes] = []
        mime_type = self._get_mime_type(audio_format)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as resp:
                if resp.status_code == 400 and return_audio:
                    # Fallback to non-streaming or text-only path
                    return await self._call_rest_non_streaming(
                        messages_formatted=messages_formatted,
                        model=model,
                        api_key=api_key,
                        api_base=api_base,
                        voice=voice,
                        return_audio=return_audio,
                        audio_format=audio_format,
                        temperature=temperature,
                        tts_model=tts_model,
                    )

                resp.raise_for_status()
                async for raw_line in resp.aiter_lines():
                    line = raw_line.strip()
                    if not line or not line.startswith("data:"):
                        continue

                    data_str = line[len("data:") :].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})

                    # Text delta
                    text_delta = delta.get("content")
                    if text_delta:
                        transcript_parts.append(text_delta)
                        if chunk_callback:
                            res = chunk_callback({"type": "text", "delta": text_delta})
                            if inspect.isawaitable(res):
                                await res

                    # Audio delta
                    audio_delta = delta.get("audio")
                    if isinstance(audio_delta, dict):
                        transcript_delta = audio_delta.get("transcript")
                        if transcript_delta:
                            transcript_parts.append(transcript_delta)
                            if chunk_callback:
                                res = chunk_callback(
                                    {"type": "text", "delta": transcript_delta}
                                )
                                if inspect.isawaitable(res):
                                    await res

                        audio_chunk_b64 = audio_delta.get("data")
                        if audio_chunk_b64 and return_audio:
                            chunk_pcm = base64.b64decode(audio_chunk_b64)
                            audio_pcm_chunks.append(chunk_pcm)
                            if chunk_callback:
                                res = chunk_callback(
                                    {
                                        "type": "audio",
                                        "data": audio_chunk_b64,
                                        "pcm_bytes": chunk_pcm,
                                        "mime_type": mime_type,
                                    }
                                )
                                if inspect.isawaitable(res):
                                    await res

        transcript = "".join(transcript_parts).strip()
        audio_b64: str | None = None

        if return_audio and audio_pcm_chunks:
            all_pcm = b"".join(audio_pcm_chunks)
            if audio_format == "wav":
                wav_bytes = self._pcm_to_wav(all_pcm, sample_rate=sample_rate)
                audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
            else:
                audio_b64 = base64.b64encode(all_pcm).decode("ascii")
        elif return_audio and transcript and tts_model:
            audio_b64 = await self._synthesize_speech(
                text=transcript,
                api_key=api_key,
                api_base=api_base,
                voice=voice,
                audio_format=audio_format,
                tts_model=tts_model,
            )
            if audio_b64 and chunk_callback:
                res = chunk_callback(
                    {
                        "type": "audio",
                        "data": audio_b64,
                        "mime_type": mime_type,
                    }
                )
                if inspect.isawaitable(res):
                    await res

        return transcript, audio_b64, sample_rate

    async def _call_realtime_websocket(
        self,
        prompt: str,
        system_instruction: str | None,
        model: str,
        api_key: str,
        api_base: str,
        voice: str,
        return_audio: bool,
        audio_format: str,
        temperature: float | None,
        chunk_callback: Callable[[dict[str, Any]], Any] | None = None,
    ) -> tuple[str, str | None, int]:
        """Bidirectional WebSocket streaming to OpenAI Realtime API (/v1/realtime)."""
        import websockets

        ws_base = api_base.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = f"{ws_base}/realtime?model={model}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        openai_audio_format = "pcm16" if audio_format in ("wav", "pcm") else audio_format
        session_config: dict[str, Any] = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"] if return_audio else ["text"],
                "voice": voice,
                "output_audio_format": openai_audio_format,
            },
        }
        if system_instruction:
            session_config["session"]["instructions"] = system_instruction
        if temperature is not None:
            session_config["session"]["temperature"] = temperature

        transcript_parts: list[str] = []
        audio_pcm_chunks: list[bytes] = []
        sample_rate = 24000
        mime_type = self._get_mime_type(audio_format)

        async with websockets.connect(
            ws_url,
            additional_headers=headers,
            open_timeout=self.timeout,
            close_timeout=self.timeout,
        ) as ws:
            # 1. Update session settings
            await ws.send(json.dumps(session_config))

            # 2. Append user input message
            user_msg = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        }
                    ],
                },
            }
            await ws.send(json.dumps(user_msg))

            # 3. Request model response
            await ws.send(json.dumps({"type": "response.create"}))

            # 4. Stream response events
            while True:
                try:
                    raw_event = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
                except asyncio.TimeoutError:
                    logger.warning("OpenAI Realtime WebSocket stream timed out waiting for server event.")
                    break

                event = json.loads(raw_event)
                event_type = event.get("type", "")

                if event_type in (
                    "response.audio_transcript.delta",
                    "response.text.delta",
                ):
                    delta_text = event.get("delta", "")
                    if delta_text:
                        transcript_parts.append(delta_text)
                        if chunk_callback:
                            res = chunk_callback({"type": "text", "delta": delta_text})
                            if inspect.isawaitable(res):
                                await res

                elif event_type == "response.audio.delta":
                    delta_b64 = event.get("delta", "")
                    if delta_b64 and return_audio:
                        pcm_bytes = base64.b64decode(delta_b64)
                        audio_pcm_chunks.append(pcm_bytes)
                        if chunk_callback:
                            res = chunk_callback(
                                {
                                    "type": "audio",
                                    "data": delta_b64,
                                    "pcm_bytes": pcm_bytes,
                                    "mime_type": mime_type,
                                }
                            )
                            if inspect.isawaitable(res):
                                await res

                elif event_type == "response.done":
                    break

                elif event_type == "error":
                    err_msg = event.get("error", {}).get("message", "Realtime stream error")
                    raise RuntimeError(f"OpenAI Realtime API error: {err_msg}")

        transcript = "".join(transcript_parts).strip()
        audio_b64: str | None = None
        if return_audio and audio_pcm_chunks:
            all_pcm = b"".join(audio_pcm_chunks)
            if audio_format == "wav":
                wav_bytes = self._pcm_to_wav(all_pcm, sample_rate=sample_rate)
                audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
            else:
                audio_b64 = base64.b64encode(all_pcm).decode("ascii")

        return transcript, audio_b64, sample_rate

    async def __wrapped__(
        self,
        messages: list[dict] | pw.Json | str,
        return_audio: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute chat completions with OpenAI-compatible audio models."""
        combined_kwargs = {**self.kwargs, **kwargs}
        model = combined_kwargs.get("model", self.model or "gpt-4o-audio-preview")
        voice = (
            combined_kwargs.get("voice")
            or combined_kwargs.get("voice_name")
            or self.voice
        )
        audio_format = combined_kwargs.get("audio_format", self.audio_format)
        transport = combined_kwargs.get("transport", self.transport)
        system_instruction = combined_kwargs.get(
            "system_instruction", self.system_instruction
        )
        temperature = combined_kwargs.get("temperature", self.temperature)
        tts_model = combined_kwargs.get("tts_model", self.tts_model)
        api_base = combined_kwargs.get("api_base", self.api_base).rstrip("/")

        if return_audio is None:
            return_audio = combined_kwargs.get("return_audio", self.return_audio)

        api_key = self._resolve_api_key(combined_kwargs.get("api_key"))

        messages_decoded = _prepare_messages(messages)
        messages_formatted = _format_openai_messages(
            messages_decoded, system_instruction=system_instruction
        )
        prompt, extracted_system = _extract_prompt_and_system(messages_decoded)
        effective_system = system_instruction or extracted_system

        logger.info(
            "OpenAIAudioChat request: model=%s transport=%s return_audio=%s voice=%s",
            model,
            transport,
            return_audio,
            voice,
        )

        chunk_callback = combined_kwargs.get("chunk_callback")

        if transport == "websocket":
            try:
                transcript, audio_b64, _ = await self._call_realtime_websocket(
                    prompt=prompt,
                    system_instruction=effective_system,
                    model=model,
                    api_key=api_key,
                    api_base=api_base,
                    voice=voice,
                    return_audio=return_audio,
                    audio_format=audio_format,
                    temperature=temperature,
                    chunk_callback=chunk_callback,
                )
            except Exception as ws_err:
                logger.warning(
                    "OpenAI Realtime WebSocket failed (%s); falling back to REST generation.",
                    ws_err,
                )
                if chunk_callback:
                    transcript, audio_b64, _ = await self._call_rest_streaming(
                        messages_formatted=messages_formatted,
                        model=model,
                        api_key=api_key,
                        api_base=api_base,
                        voice=voice,
                        return_audio=return_audio,
                        audio_format=audio_format,
                        temperature=temperature,
                        tts_model=tts_model,
                        chunk_callback=chunk_callback,
                    )
                else:
                    transcript, audio_b64, _ = await self._call_rest_non_streaming(
                        messages_formatted=messages_formatted,
                        model=model,
                        api_key=api_key,
                        api_base=api_base,
                        voice=voice,
                        return_audio=return_audio,
                        audio_format=audio_format,
                        temperature=temperature,
                        tts_model=tts_model,
                    )
        elif chunk_callback:
            transcript, audio_b64, _ = await self._call_rest_streaming(
                messages_formatted=messages_formatted,
                model=model,
                api_key=api_key,
                api_base=api_base,
                voice=voice,
                return_audio=return_audio,
                audio_format=audio_format,
                temperature=temperature,
                tts_model=tts_model,
                chunk_callback=chunk_callback,
            )
        else:
            transcript, audio_b64, _ = await self._call_rest_non_streaming(
                messages_formatted=messages_formatted,
                model=model,
                api_key=api_key,
                api_base=api_base,
                voice=voice,
                return_audio=return_audio,
                audio_format=audio_format,
                temperature=temperature,
                tts_model=tts_model,
            )

        if return_audio:
            mime_type = self._get_mime_type(audio_format)
            return {
                "response": transcript,
                "text": transcript,
                "audio": audio_b64 or "",
                "audio_mime_type": mime_type,
            }

        return transcript

    async def stream(
        self,
        messages: list[dict] | pw.Json | str,
        return_audio: bool | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Asynchronously stream incremental response chunks in real time.

        Yields dictionaries with event types:
          - ``{"type": "text", "delta": str}``: Incremental text or transcript tokens.
          - ``{"type": "audio", "data": str, "pcm_bytes": bytes, "mime_type": str}``: Audio frames.
          - ``{"type": "done", "response": str, "audio": str | None, "audio_mime_type": str | None}``: Turn complete.
        """
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def _on_chunk(chunk: dict[str, Any]) -> None:
            await queue.put(chunk)

        producer_kwargs = {**kwargs, "chunk_callback": _on_chunk}

        async def _producer() -> None:
            try:
                res = await self.__wrapped__(
                    messages,
                    return_audio=return_audio,
                    **producer_kwargs,
                )
                if isinstance(res, dict):
                    await queue.put(
                        {
                            "type": "done",
                            "response": res.get("response", ""),
                            "audio": res.get("audio"),
                            "audio_mime_type": res.get("audio_mime_type"),
                        }
                    )
                else:
                    await queue.put({"type": "done", "response": str(res)})
            except Exception as e:
                await queue.put({"type": "error", "error": str(e)})
            finally:
                await queue.put(None)

        producer_task = asyncio.create_task(_producer())

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if item.get("type") == "error":
                    raise RuntimeError(item.get("error", "Stream error"))
                yield item
        finally:
            if not producer_task.done():
                producer_task.cancel()


# Alias for naming consistency with GeminiLiveChat
OpenAILiveChat = OpenAIAudioChat
