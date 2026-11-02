import asyncio
import inspect
import json
import logging
from typing import Any, Callable

try:
    from aiohttp import web
except ImportError:
    web = None
import pathway as pw

if not hasattr(pw, "Schema"):
    class Schema:  # type: ignore[no-redef]
        pass
    pw.Schema = Schema  # type: ignore[attr-defined]

if not hasattr(pw, "column_definition"):
    def _col_def(**kwargs):
        class ColDef:
            default_value = kwargs.get("default_value")
        return ColDef()
    pw.column_definition = _col_def  # type: ignore[attr-defined]
try:
    from pathway.xpacks.llm.question_answering import BaseQuestionAnswerer
    from pathway.xpacks.llm.servers import BaseRestServer
except (ImportError, AttributeError):
    class BaseQuestionAnswerer:  # type: ignore[no-redef]
        pass

    class BaseRestServer:  # type: ignore[no-redef]
        def __init__(self, host: str, port: int, **kwargs):
            class WebServer:
                def __init__(self):
                    self._app = web.Application() if web else None
            self.webserver = WebServer()

        def serve(self, route: str, schema: Any, handler: Any, **kwargs):
            pass

from processors.chat_moderator import ChatModerator

logger = logging.getLogger(__name__)


class QAScopedRestServer(BaseRestServer):
    """
    Creates a REST Server for answering queries to a given instance of ``BaseQuestionAnswerer``.
    It exposes endpoints:
    - ``/v1/retrieve``: document retrieval
    - ``/v2/answer``: RAG question answering with optional audio generation and SSE streaming
    - ``/v2/list_documents``: list documents and indexing status
    - ``/v1/moderate``: chat moderation (optional)

    Args:
        host: host on which server will run
        port: port on which server will run
        rag_question_answerer: instance of ``BaseQuestionAnswerer`` which is used
            to answer queries received in the endpoints.
        chat_moderator: optional instance of ``ChatModerator``
        rest_kwargs: optional kwargs to be passed to ``pw.io.http.rest_connector``
    """

    def __init__(
        self,
        host: str,
        port: int,
        rag_question_answerer: BaseQuestionAnswerer,
        chat_moderator: ChatModerator | None = None,
        **rest_kwargs,
    ):
        super().__init__(host, port, **rest_kwargs)
        self.host = host
        self.port = port
        self.rag_question_answerer = rag_question_answerer
        self.chat_moderator = chat_moderator
        self.schemas: dict[str, Any] = {}

        class CustomAnswerQuerySchema(pw.Schema):
            prompt: str
            filters: str | None = pw.column_definition(default_value=None)
            model: str | None = pw.column_definition(
                default_value=rag_question_answerer.llm.model
            )
            return_context_docs: bool = pw.column_definition(default_value=False)
            return_prompt: bool = pw.column_definition(default_value=False)
            return_audio: bool = pw.column_definition(default_value=False)
            stream: bool = pw.column_definition(default_value=False)

        logger.info("Initializing QAScopedRestServer")

        # Intercept /v2/answer at the handler level so that request.clone() in
        # Pathway's _wrap_handler_with_logger runs before the body is read.
        if hasattr(self.webserver, "_register_endpoint"):
            orig_register_endpoint = self.webserver._register_endpoint

            def custom_register_endpoint(route, handler, *args, **kwargs):
                if route == "/v2/answer":
                    handler = self._create_streaming_handler(handler)

                return orig_register_endpoint(route, handler, *args, **kwargs)

            self.webserver._register_endpoint = custom_register_endpoint

        self.serve(
            "/v1/retrieve",
            rag_question_answerer.RetrieveQuerySchema,
            rag_question_answerer.retrieve,
            **rest_kwargs,
        )
        self.serve(
            "/v2/answer",
            CustomAnswerQuerySchema,
            rag_question_answerer.answer_query,
            **rest_kwargs,
        )
        self.serve(
            "/v2/list_documents",
            rag_question_answerer.InputsQuerySchema,
            rag_question_answerer.list_documents,
            **rest_kwargs,
        )

        if chat_moderator:
            self.serve(
                "/v1/moderate",
                chat_moderator.ModerationSchema,
                chat_moderator.moderate,
                **rest_kwargs,
            )

    def _create_streaming_handler(
        self, orig_handler: Callable[[Any], Any]
    ) -> Callable[[Any], Any]:
        """Wrap the standard REST handler to intercept requests with stream=True."""
        async def streaming_aware_handler(request: web.Request) -> Any:
            try:
                payload = await request.json()
            except Exception:
                payload = {}

            is_stream = str(
                payload.get("stream", request.query.get("stream", ""))
            ).lower() in ("true", "1")

            if is_stream:
                return await self._handle_stream_answer(request, payload)

            return await orig_handler(request)

        return streaming_aware_handler

    async def _retrieve_context_docs(
        self, prompt: str, filters: str | None = None
    ) -> list[dict]:
        """Fetch candidate context documents from the running /v1/retrieve endpoint."""
        search_topk = getattr(self.rag_question_answerer, "search_topk", 6)

        # Allow custom retrieval helper if implemented by the question answerer
        if hasattr(self.rag_question_answerer, "retrieve_docs"):
            try:
                res = self.rag_question_answerer.retrieve_docs(
                    prompt, k=search_topk, filters=filters
                )
                if inspect.isawaitable(res):
                    return await res
                return res
            except Exception as e:
                logger.warning("Custom retrieve_docs helper failed: %s", e)

        retrieve_host = "127.0.0.1" if self.host in ("0.0.0.0", "") else self.host
        url = f"http://{retrieve_host}:{self.port}/v1/retrieve"

        req_payload: dict[str, Any] = {
            "query": prompt,
            "k": search_topk,
        }
        if filters is not None:
            req_payload["metadata_filter"] = filters

        try:
            import aiohttp

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10.0)
            ) as session:
                async with session.post(url, json=req_payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if isinstance(data, list):
                            return data
                    else:
                        logger.warning(
                            "Failed to retrieve context documents from %s: HTTP %s",
                            url,
                            resp.status,
                        )
        except Exception as err:
            logger.warning("Could not query %s for streaming context: %s", url, err)

        return []

    async def _handle_stream_answer(
        self, request: web.Request, payload: dict
    ) -> web.StreamResponse:
        """Handle streaming RAG requests over Server-Sent Events (SSE)."""
        prompt = payload.get("prompt")
        if not prompt:
            return web.json_response(
                {"error": "Missing required field 'prompt'"}, status=400
            )

        return_audio = bool(payload.get("return_audio", False))
        return_context_docs = bool(payload.get("return_context_docs", False))
        return_prompt = bool(payload.get("return_prompt", False))
        filters = payload.get("filters")
        model = payload.get("model")

        # 1. Retrieve candidate context documents
        docs = await self._retrieve_context_docs(prompt, filters)

        # 2. Setup Server-Sent Events (SSE) StreamResponse
        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)

        async def send_event(event_type: str, data: dict):
            msg = f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            await response.write(msg.encode("utf-8"))

        full_response_text = ""
        full_audio_b64: str | None = None
        audio_mime_type = "audio/wav" if return_audio else None

        try:
            if hasattr(self.rag_question_answerer, "stream_query"):
                async for chunk in self.rag_question_answerer.stream_query(
                    prompt=prompt,
                    docs=docs,
                    return_audio=return_audio,
                    model=model,
                ):
                    chunk_type = chunk.get("type")
                    if chunk_type == "text":
                        delta = chunk.get("delta", "")
                        full_response_text += delta
                        await send_event("text", {"delta": delta})
                    elif chunk_type == "audio":
                        if return_audio:
                            await send_event(
                                "audio",
                                {
                                    "data": chunk.get("data", ""),
                                    "mime_type": chunk.get(
                                        "mime_type", "audio/pcm;rate=24000"
                                    ),
                                },
                            )
                    elif chunk_type == "done":
                        done_resp = chunk.get("response", "")
                        if done_resp:
                            full_response_text = done_resp
                        if chunk.get("audio"):
                            full_audio_b64 = chunk.get("audio")
                        if chunk.get("audio_mime_type"):
                            audio_mime_type = chunk.get("audio_mime_type")
            else:
                # Fallback if question answerer does not implement stream_query
                formatted_prompt = prompt
                if hasattr(self.rag_question_answerer, "format_rag_prompt"):
                    formatted_prompt = self.rag_question_answerer.format_rag_prompt(
                        docs, prompt
                    )

                llm = getattr(self.rag_question_answerer, "llm", None)
                if llm and hasattr(llm, "stream"):
                    async for chunk in llm.stream(
                        formatted_prompt, return_audio=return_audio
                    ):
                        chunk_type = chunk.get("type")
                        if chunk_type == "text":
                            delta = chunk.get("delta", "")
                            full_response_text += delta
                            await send_event("text", {"delta": delta})
                        elif chunk_type == "audio" and return_audio:
                            await send_event(
                                "audio",
                                {
                                    "data": chunk.get("data", ""),
                                    "mime_type": chunk.get(
                                        "mime_type", "audio/pcm;rate=24000"
                                    ),
                                },
                            )
                        elif chunk_type == "done":
                            if chunk.get("response"):
                                full_response_text = chunk["response"]
                            if chunk.get("audio"):
                                full_audio_b64 = chunk["audio"]
                elif llm:
                    call_fn = getattr(llm, "__wrapped__", llm)
                    if inspect.iscoroutinefunction(call_fn):
                        res = await call_fn(
                            formatted_prompt, return_audio=return_audio
                        )
                    else:
                        res = await asyncio.to_thread(
                            call_fn, formatted_prompt, return_audio=return_audio
                        )
                    if isinstance(res, dict):
                        full_response_text = res.get("response") or res.get("text", "")
                        full_audio_b64 = res.get("audio")
                    else:
                        full_response_text = str(res) if res is not None else ""

                    if full_response_text:
                        await send_event("text", {"delta": full_response_text})

            # Send terminal 'done' event with final payload
            done_payload: dict[str, Any] = {
                "response": full_response_text,
            }
            if return_audio:
                done_payload["audio"] = full_audio_b64 or ""
                done_payload["audio_mime_type"] = audio_mime_type or "audio/wav"
            if return_context_docs:
                done_payload["context_docs"] = docs
            if return_prompt:
                formatted_p = prompt
                if hasattr(self.rag_question_answerer, "format_rag_prompt"):
                    formatted_p = self.rag_question_answerer.format_rag_prompt(
                        docs, prompt
                    )
                done_payload["prompt"] = formatted_p

            await send_event("done", done_payload)

        except (ConnectionResetError, asyncio.CancelledError):
            logger.info("Client disconnected during stream for prompt: %.30s...", prompt)
        except Exception as e:
            logger.exception("Error while streaming response: %s", e)
            try:
                await send_event("error", {"error": str(e)})
            except Exception:
                pass
        finally:
            try:
                await response.write_eof()
            except Exception:
                pass

        return response

    def serve(
        self,
        route: str,
        schema: type[pw.Schema],
        handler: Callable[[pw.Table], pw.Table],
        **additional_endpoint_kwargs,
    ):
        self.schemas[route] = schema
        if hasattr(pw, "io") and hasattr(pw.io, "http") and hasattr(pw.io.http, "rest_connector"):
            queries, writer = pw.io.http.rest_connector(
                webserver=self.webserver,
                route=route,
                schema=schema,
                autocommit_duration_ms=50,
                delete_completed_queries=False,
                **additional_endpoint_kwargs,
            )
            writer(handler(queries))
