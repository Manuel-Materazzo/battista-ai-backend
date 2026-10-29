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
    else:
        messages_as_list = messages

    messages_decoded: list[dict] = [
        message.as_dict() if has_pw_json and isinstance(message, pw.Json) else message
        for message in messages_as_list
    ]
    return messages_decoded


def _extract_prompt_and_system(
    messages_decoded: list[dict],
) -> tuple[str, str | None]:
    """Extract the main user prompt text and optional system instruction."""
    system_parts: list[str] = []
    user_parts: list[str] = []

    for msg in messages_decoded:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Extract text blocks if content is structured
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


class GeminiLiveChat(BaseChat):
    """
    Pathway Chat connector for Google AI Studio Gemini Multimodal Live models.

    Supports native audio generation using models such as ``gemini-3.8-live``,
    streaming bidirectional WebSockets (via ``BidiGenerateContent``) and REST fallback.

    When ``return_audio`` is True, generates synthesized speech in WAV (or raw PCM)
    along with text transcripts.
    """

    def __init__(
        self,
        model: str = "gemini-3.8-live",
        api_key: str | None = None,
        api_base: str = "generativelanguage.googleapis.com",
        voice_name: str = "Puck",
        audio_format: Literal["wav", "pcm"] = "wav",
        return_audio: bool = False,
        transport: Literal["websocket", "rest"] = "websocket",
        system_instruction: str | None = None,
        temperature: float | None = None,
        timeout: float = 60.0,
        capacity: int | None = None,
        retry_strategy: Any | None = None,
        cache_strategy: Any | None = None,
        *,
        async_mode: Literal["batch_async", "fully_async"] = "batch_async",
        **kwargs: Any,
    ) -> None:
        if hasattr(pw, "udfs"):
            retry = retry_strategy or getattr(pw.udfs, "ExponentialBackoffRetryStrategy", lambda: None)()
            if async_mode == "fully_async":
                executor = pw.udfs.fully_async_executor(capacity=capacity, retry_strategy=retry)
            else:
                executor = pw.udfs.async_executor(capacity=capacity, retry_strategy=retry)
            super().__init__(executor=executor, cache_strategy=cache_strategy, return_type=Any)
        else:
            super().__init__()

        self.api_key = api_key
        self.api_base = api_base
        self.voice_name = voice_name
        self.audio_format = audio_format
        self.return_audio = return_audio
        self.transport = transport
        self.system_instruction = system_instruction
        self.temperature = temperature
        self.timeout = timeout
        self.supports_audio = True

        self.kwargs: dict[str, Any] = {
            "model": model,
            "api_key": api_key,
            "voice_name": voice_name,
            "audio_format": audio_format,
            "return_audio": return_audio,
            "transport": transport,
            **kwargs,
        }

    @property
    def model(self) -> str | None:
        return self.kwargs.get("model", "gemini-3.8-live")

    def _accepts_call_arg(self, arg_name: str) -> bool:
        """Indicate supported query-time keyword arguments."""
        return arg_name in {
            "model",
            "api_key",
            "return_audio",
            "voice_name",
            "temperature",
            "audio_format",
            "system_instruction",
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

    async def _call_live_websocket(
        self,
        prompt: str,
        model: str,
        api_key: str,
        api_base: str,
        voice_name: str,
        return_audio: bool,
        audio_format: str,
        system_instruction: str | None,
        temperature: float | None,
        chunk_callback: Callable[[dict[str, Any]], Any] | None = None,
    ) -> tuple[str, str | None, int]:
        """Bidirectional WebSocket streaming to Gemini Live API (BidiGenerateContent)."""
        import websockets

        clean_model = model if model.startswith("models/") else f"models/{model}"
        ws_url = (
            f"wss://{api_base}/ws/google.ai.generativelanguage.v1beta."
            f"GenerativeService.BidiGenerateContent?key={api_key}"
        )

        gen_config: dict[str, Any] = {
            "responseModalities": ["AUDIO"] if return_audio else ["TEXT"],
        }
        if return_audio and voice_name:
            gen_config["speechConfig"] = {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": voice_name,
                    }
                }
            }
        if temperature is not None:
            gen_config["temperature"] = temperature

        setup_msg: dict[str, Any] = {
            "setup": {
                "model": clean_model,
                "generationConfig": gen_config,
            }
        }
        if return_audio:
            setup_msg["setup"]["outputAudioTranscription"] = {}
        if system_instruction:
            setup_msg["setup"]["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        transcript_parts: list[str] = []
        pcm_chunks: list[bytes] = []
        sample_rate = 24000

        async with websockets.connect(
            ws_url,
            ssl=True,
            open_timeout=self.timeout,
            close_timeout=self.timeout,
        ) as ws:
            # 1. Send initialization setup frame
            await ws.send(json.dumps(setup_msg))

            # 2. Send client input turn
            client_turn = {
                "clientContent": {
                    "turns": [
                        {
                            "role": "user",
                            "parts": [{"text": prompt}],
                        }
                    ],
                    "turnComplete": True,
                }
            }
            await ws.send(json.dumps(client_turn))

            # 3. Stream server response
            while True:
                try:
                    raw_msg = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
                except asyncio.TimeoutError:
                    logger.warning("Gemini Live WebSocket stream timed out waiting for server content.")
                    break

                msg = json.loads(raw_msg)
                if "serverContent" in msg:
                    sc = msg["serverContent"]

                    # Extract transcript from outputAudioTranscription
                    text_delta = ""
                    if "outputTranscription" in sc and "text" in sc["outputTranscription"]:
                        text_delta = sc["outputTranscription"]["text"]
                    elif "outputAudioTranscription" in sc and "text" in sc["outputAudioTranscription"]:
                        text_delta = sc["outputAudioTranscription"]["text"]

                    if text_delta:
                        transcript_parts.append(text_delta)
                        if chunk_callback:
                            res = chunk_callback({"type": "text", "delta": text_delta})
                            if inspect.isawaitable(res):
                                await res

                    # Extract text or audio from modelTurn parts
                    if "modelTurn" in sc:
                        for part in sc["modelTurn"].get("parts", []):
                            if "text" in part:
                                part_text = part["text"]
                                transcript_parts.append(part_text)
                                if chunk_callback:
                                    res = chunk_callback({"type": "text", "delta": part_text})
                                    if inspect.isawaitable(res):
                                        await res
                            if "inlineData" in part:
                                inline = part["inlineData"]
                                mime = inline.get("mimeType", "")
                                if "rate=" in mime:
                                    try:
                                        sample_rate = int(mime.split("rate=")[1].split(";")[0])
                                    except (ValueError, IndexError):
                                        pass
                                chunk_b64 = inline.get("data", "")
                                if chunk_b64:
                                    chunk_pcm = base64.b64decode(chunk_b64)
                                    pcm_chunks.append(chunk_pcm)
                                    if chunk_callback:
                                        res = chunk_callback({
                                            "type": "audio",
                                            "data": chunk_b64,
                                            "pcm_bytes": chunk_pcm,
                                            "mime_type": mime or f"audio/pcm;rate={sample_rate}",
                                        })
                                        if inspect.isawaitable(res):
                                            await res

                    if sc.get("turnComplete"):
                        break

        transcript = "".join(transcript_parts).strip()
        if not transcript and return_audio and pcm_chunks:
            transcript = "[Audio generated]"

        audio_b64: str | None = None
        if return_audio and pcm_chunks:
            pcm_bytes = b"".join(pcm_chunks)
            if audio_format == "wav":
                wav_bytes = self._pcm_to_wav(pcm_bytes, sample_rate=sample_rate)
                audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
            else:
                audio_b64 = base64.b64encode(pcm_bytes).decode("ascii")

        return transcript, audio_b64, sample_rate

    async def _call_rest(
        self,
        prompt: str,
        model: str,
        api_key: str,
        api_base: str,
        voice_name: str,
        return_audio: bool,
        audio_format: str,
        system_instruction: str | None,
        temperature: float | None,
    ) -> tuple[str, str | None, int]:
        """REST API fallback via generateContent."""
        import httpx

        clean_model = model.removeprefix("models/")
        url = f"https://{api_base}/v1beta/models/{clean_model}:generateContent?key={api_key}"

        gen_config: dict[str, Any] = {
            "responseModalities": ["AUDIO", "TEXT"] if return_audio else ["TEXT"],
        }
        if return_audio and voice_name:
            gen_config["speechConfig"] = {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": voice_name,
                    }
                }
            }
        if temperature is not None:
            gen_config["temperature"] = temperature

        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": gen_config,
        }
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        candidates = data.get("candidates", [])
        if not candidates:
            return "", None, 24000

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        transcript_parts: list[str] = []
        pcm_chunks: list[bytes] = []
        sample_rate = 24000

        for part in parts:
            if "text" in part:
                transcript_parts.append(part["text"])
            if "inlineData" in part:
                inline = part["inlineData"]
                mime = inline.get("mimeType", "")
                if "rate=" in mime:
                    try:
                        sample_rate = int(mime.split("rate=")[1].split(";")[0])
                    except (ValueError, IndexError):
                        pass
                chunk_b64 = inline.get("data", "")
                if chunk_b64:
                    pcm_chunks.append(base64.b64decode(chunk_b64))

        transcript = "".join(transcript_parts).strip()
        audio_b64: str | None = None
        if return_audio and pcm_chunks:
            pcm_bytes = b"".join(pcm_chunks)
            if audio_format == "wav":
                wav_bytes = self._pcm_to_wav(pcm_bytes, sample_rate=sample_rate)
                audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
            else:
                audio_b64 = base64.b64encode(pcm_bytes).decode("ascii")

        return transcript, audio_b64, sample_rate

    async def __wrapped__(
        self,
        messages: list[dict] | pw.Json | str,
        return_audio: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute the inference request with Google AI Studio Live / Multimodal models."""
        combined_kwargs = {**self.kwargs, **kwargs}
        model = combined_kwargs.get("model", self.model or "gemini-3.8-live")
        voice_name = combined_kwargs.get("voice_name", self.voice_name)
        audio_format = combined_kwargs.get("audio_format", self.audio_format)
        transport = combined_kwargs.get("transport", self.transport)
        system_instruction = combined_kwargs.get("system_instruction", self.system_instruction)
        temperature = combined_kwargs.get("temperature", self.temperature)

        if return_audio is None:
            return_audio = combined_kwargs.get("return_audio", self.return_audio)

        api_key = combined_kwargs.get("api_key") or self.api_key
        if not api_key or api_key == "fake":
            api_key = (
                os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
                or api_key
            )

        if not api_key or api_key == "fake":
            raise ValueError(
                "Gemini API key is required. Set GEMINI_API_KEY or GOOGLE_API_KEY "
                "environment variable, or provide a valid api_key to GeminiLiveChat."
            )

        messages_decoded = _prepare_messages(messages)
        prompt, extracted_system = _extract_prompt_and_system(messages_decoded)
        effective_system = system_instruction or extracted_system

        logger.info(
            "GeminiLiveChat request: model=%s transport=%s return_audio=%s voice=%s",
            model,
            transport,
            return_audio,
            voice_name,
        )

        chunk_callback = combined_kwargs.get("chunk_callback")

        if transport == "rest":
            transcript, audio_b64, _ = await self._call_rest(
                prompt=prompt,
                model=model,
                api_key=api_key,
                api_base=self.api_base,
                voice_name=voice_name,
                return_audio=return_audio,
                audio_format=audio_format,
                system_instruction=effective_system,
                temperature=temperature,
            )
            if chunk_callback:
                if transcript:
                    res = chunk_callback({"type": "text", "delta": transcript})
                    if inspect.isawaitable(res):
                        await res
                if audio_b64:
                    res = chunk_callback({
                        "type": "audio",
                        "data": audio_b64,
                        "mime_type": f"audio/{audio_format}",
                    })
                    if inspect.isawaitable(res):
                        await res
        else:
            try:
                transcript, audio_b64, _ = await self._call_live_websocket(
                    prompt=prompt,
                    model=model,
                    api_key=api_key,
                    api_base=self.api_base,
                    voice_name=voice_name,
                    return_audio=return_audio,
                    audio_format=audio_format,
                    system_instruction=effective_system,
                    temperature=temperature,
                    chunk_callback=chunk_callback,
                )
            except Exception as ws_err:
                logger.warning(
                    "Gemini Live WebSocket failed (%s); falling back to REST generation.",
                    ws_err,
                )
                transcript, audio_b64, _ = await self._call_rest(
                    prompt=prompt,
                    model=model,
                    api_key=api_key,
                    api_base=self.api_base,
                    voice_name=voice_name,
                    return_audio=return_audio,
                    audio_format=audio_format,
                    system_instruction=effective_system,
                    temperature=temperature,
                )
                if chunk_callback:
                    if transcript:
                        res = chunk_callback({"type": "text", "delta": transcript})
                        if inspect.isawaitable(res):
                            await res
                    if audio_b64:
                        res = chunk_callback({
                            "type": "audio",
                            "data": audio_b64,
                            "mime_type": f"audio/{audio_format}",
                        })
                        if inspect.isawaitable(res):
                            await res

        if return_audio:
            mime_type = "audio/wav" if audio_format == "wav" else "audio/pcm"
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
        Asynchronously stream incremental response chunks from Gemini in real time.

        Yields dictionaries with event types:
          - ``{"type": "text", "delta": str}``: Incremental text or transcript tokens.
          - ``{"type": "audio", "data": str, "pcm_bytes": bytes, "mime_type": str}``: Raw PCM audio frames.
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
                    await queue.put({
                        "type": "done",
                        "response": res.get("response", ""),
                        "audio": res.get("audio"),
                        "audio_mime_type": res.get("audio_mime_type"),
                    })
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

