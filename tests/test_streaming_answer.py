import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop, make_mocked_request

from processors.reranked_rag_question_answerer import RerankedRAGQuestionAnswerer
from server.qa_scoped_rest_server import QAScopedRestServer


class FakeLLM:
    def __init__(self, model="test-model"):
        self.model = model

    async def stream(self, messages, return_audio=False, **kwargs):
        yield {"type": "text", "delta": "Hello "}
        if return_audio:
            yield {
                "type": "audio",
                "data": "AAAA",
                "pcm_bytes": b"\x00\x00",
                "mime_type": "audio/pcm;rate=24000",
            }
        yield {"type": "text", "delta": "world!"}
        yield {
            "type": "done",
            "response": "Hello world!",
            "audio": "RIFF_WAV_BASE64" if return_audio else None,
            "audio_mime_type": "audio/wav" if return_audio else None,
        }


class FakeIndexer:
    def __init__(self):
        self.RetrieveQuerySchema = MagicMock()
        self.StatisticsQuerySchema = MagicMock()
        self.InputsQuerySchema = MagicMock()

    def retrieve_query(self, queries):
        return queries


class FakeReranker:
    def __wrapped__(self, doc_text, query):
        if "relevant" in doc_text:
            return 0.95
        return 0.10


def test_docs_to_context_and_prompt_formatting():
    llm = FakeLLM()
    indexer = FakeIndexer()
    prompt_template = "Context:\n{context}\n\nQuestion: {query}"
    answerer = RerankedRAGQuestionAnswerer(
        llm=llm,
        indexer=indexer,
        prompt_template=prompt_template,
    )

    docs = [
        {"text": "Towny is a territory plugin.", "metadata": {"path": "/rules/towny.md"}},
        {"text": "Use /plot claim to claim.", "metadata": {"path": "/rules/commands.md"}},
    ]
    context_str = answerer.docs_to_context(docs)
    assert "/rules/towny.md" in context_str
    assert "Towny is a territory plugin." in context_str

    formatted_prompt = answerer.format_rag_prompt(docs, "How do I claim?")
    assert "Context:" in formatted_prompt
    assert "Towny is a territory plugin." in formatted_prompt
    assert "Question: How do I claim?" in formatted_prompt


@pytest.mark.asyncio
async def test_stream_query_text_only():
    llm = FakeLLM()
    indexer = FakeIndexer()
    answerer = RerankedRAGQuestionAnswerer(
        llm=llm,
        indexer=indexer,
        prompt_template="Context: {context}\nQuery: {query}",
    )

    chunks = []
    async for chunk in answerer.stream_query(
        prompt="Hi",
        docs=[{"text": "Doc text"}],
        return_audio=False,
    ):
        chunks.append(chunk)

    text_deltas = [c["delta"] for c in chunks if c.get("type") == "text"]
    assert "".join(text_deltas) == "Hello world!"

    done_chunks = [c for c in chunks if c.get("type") == "done"]
    assert len(done_chunks) == 1
    assert done_chunks[0]["response"] == "Hello world!"
    assert done_chunks[0]["audio"] is None


@pytest.mark.asyncio
async def test_stream_query_with_audio():
    llm = FakeLLM()
    indexer = FakeIndexer()
    answerer = RerankedRAGQuestionAnswerer(
        llm=llm,
        indexer=indexer,
        prompt_template="Context: {context}\nQuery: {query}",
    )

    chunks = []
    async for chunk in answerer.stream_query(
        prompt="Hi",
        docs=[],
        return_audio=True,
    ):
        chunks.append(chunk)

    audio_chunks = [c for c in chunks if c.get("type") == "audio"]
    assert len(audio_chunks) == 1
    assert audio_chunks[0]["data"] == "AAAA"

    done_chunk = [c for c in chunks if c.get("type") == "done"][0]
    assert done_chunk["audio"] == "RIFF_WAV_BASE64"
    assert done_chunk["audio_mime_type"] == "audio/wav"


@pytest.mark.asyncio
async def test_stream_query_with_reranker():
    llm = FakeLLM()
    indexer = FakeIndexer()
    reranker = FakeReranker()
    answerer = RerankedRAGQuestionAnswerer(
        llm=llm,
        indexer=indexer,
        reranker=reranker,
        rerank_topk=1,
        prompt_template="Context: {context}\nQuery: {query}",
    )

    docs = [
        {"text": "irrelevant text"},
        {"text": "relevant text for towny"},
    ]

    chunks = []
    async for chunk in answerer.stream_query(
        prompt="How do I claim?",
        docs=docs,
        return_audio=False,
    ):
        chunks.append(chunk)

    done_chunk = [c for c in chunks if c.get("type") == "done"][0]
    assert done_chunk["response"] == "Hello world!"


def make_mock_request_with_body(method, path, body_dict, writer=None):
    from aiohttp.streams import StreamReader
    loop = asyncio.get_event_loop()
    proto = MagicMock()
    proto.loop = loop
    proto._reading_paused = False
    stream = StreamReader(proto, 65536, loop=loop)
    stream.feed_data(json.dumps(body_dict).encode("utf-8"))
    stream.feed_eof()
    kwargs = {"payload": stream, "loop": loop, "headers": {"Content-Type": "application/json"}}
    if writer:
        kwargs["writer"] = writer
    return make_mocked_request(method, path, **kwargs)


@pytest.mark.asyncio
async def test_qa_scoped_rest_server_schema_has_stream():
    llm = FakeLLM()
    indexer = FakeIndexer()
    answerer = RerankedRAGQuestionAnswerer(llm=llm, indexer=indexer)

    server = QAScopedRestServer(
        host="127.0.0.1",
        port=9999,
        rag_question_answerer=answerer,
    )

    schema = server.schemas["/v2/answer"]
    assert hasattr(schema, "stream")
    stream_col = getattr(schema, "stream")
    default_val = getattr(stream_col, "default_value", False)
    assert default_val is False
    assert hasattr(schema, "return_audio")


@pytest.mark.asyncio
async def test_qa_scoped_rest_server_stream_text():
    llm = FakeLLM()
    indexer = FakeIndexer()
    answerer = RerankedRAGQuestionAnswerer(llm=llm, indexer=indexer)

    server = QAScopedRestServer(
        host="127.0.0.1",
        port=9999,
        rag_question_answerer=answerer,
    )

    # Mock _retrieve_context_docs to return a doc
    server._retrieve_context_docs = AsyncMock(return_value=[
        {"text": "Mock document", "metadata": {"path": "doc.md"}}
    ])

    # Simulate Pathway's _wrap_handler_with_logger
    async def pathway_wrapped_handler(req):
        headers = req.headers.copy()
        headers["X-Pathway-Session"] = "test-session"
        cloned_req = req.clone(headers=headers)
        return await streaming_handler(cloned_req)

    # Dummy fallback handler that should only be reached if stream=False
    async def dummy_handler(request):
        assert "X-Pathway-Session" in request.headers
        return web.json_response({"response": "non_streamed_batch"})

    streaming_handler = server._create_streaming_handler(dummy_handler)

    # 1. Test stream=True
    mock_writer = AsyncMock()
    mock_writer.drain = AsyncMock()
    mock_writer.write = AsyncMock()
    mock_writer.write_headers = AsyncMock()
    mock_writer.write_eof = AsyncMock()
    mock_writer.send_headers = MagicMock()
    mock_writer.enable_chunking = MagicMock()
    mock_writer.enable_compression = MagicMock()

    request_stream = make_mock_request_with_body(
        "POST",
        "/v2/answer",
        {
            "prompt": "Test streaming",
            "stream": True,
            "return_audio": False,
            "return_context_docs": True,
        },
        writer=mock_writer,
    )

    response = await pathway_wrapped_handler(request_stream)
    assert isinstance(response, web.StreamResponse)
    assert response.headers["Content-Type"] == "text/event-stream"

    # 2. Test stream=False (fallback to standard handler without breaking request.clone)
    request_batch = make_mock_request_with_body(
        "POST",
        "/v2/answer",
        {
            "prompt": "Test batch",
            "stream": False,
        },
    )

    response_batch = await pathway_wrapped_handler(request_batch)
    assert isinstance(response_batch, web.Response)
    body = json.loads(response_batch.text)
    assert body["response"] == "non_streamed_batch"


@pytest.mark.asyncio
async def test_qa_scoped_rest_server_stream_both_audio_and_text():
    llm = FakeLLM()
    indexer = FakeIndexer()
    answerer = RerankedRAGQuestionAnswerer(llm=llm, indexer=indexer)

    server = QAScopedRestServer(
        host="127.0.0.1",
        port=9999,
        rag_question_answerer=answerer,
    )

    server._retrieve_context_docs = AsyncMock(return_value=[
        {"text": "Towny plot claim guide", "metadata": {"path": "towny.md"}}
    ])

    mock_writer = AsyncMock()
    mock_writer.drain = AsyncMock()
    written_data = []
    mock_writer.write = AsyncMock(side_effect=lambda data: written_data.append(data))
    mock_writer.write_headers = AsyncMock()
    mock_writer.write_eof = AsyncMock()
    mock_writer.send_headers = MagicMock()
    mock_writer.enable_chunking = MagicMock()
    mock_writer.enable_compression = MagicMock()

    request = make_mock_request_with_body(
        "POST",
        "/v2/answer",
        {
            "prompt": "How to claim plot?",
            "stream": True,
            "return_audio": True,
            "return_context_docs": True,
        },
        writer=mock_writer,
    )

    async def dummy_handler(_req):
        return web.json_response({})

    streaming_handler = server._create_streaming_handler(dummy_handler)
    response = await streaming_handler(request)
    assert isinstance(response, web.StreamResponse)

    combined_output = b"".join(written_data).decode("utf-8")
    assert "event: text" in combined_output
    assert "event: audio" in combined_output
    assert "event: done" in combined_output
    assert "RIFF_WAV_BASE64" in combined_output


@pytest.mark.asyncio
async def test_qa_scoped_rest_server_missing_prompt():
    llm = FakeLLM()
    indexer = FakeIndexer()
    answerer = RerankedRAGQuestionAnswerer(llm=llm, indexer=indexer)

    server = QAScopedRestServer(
        host="127.0.0.1",
        port=9999,
        rag_question_answerer=answerer,
    )

    request = make_mock_request_with_body(
        "POST",
        "/v2/answer",
        {
            "stream": True,
        },
    )

    async def dummy_handler(_req):
        return web.json_response({})

    streaming_handler = server._create_streaming_handler(dummy_handler)
    response = await streaming_handler(request)
    assert response.status == 400
    body = json.loads(response.text)
    assert "Missing required field 'prompt'" in body["error"]

