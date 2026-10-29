import logging
from typing import Any

import pathway as pw
from pathway.xpacks.llm import llms, rerankers
from pathway.xpacks.llm.question_answering import (
    BaseRAGQuestionAnswerer,
    _limit_documents,
    _prepare_RAG_response,
)

logger = logging.getLogger(__name__)


class RerankedRAGQuestionAnswerer(BaseRAGQuestionAnswerer):
    """
    Enhanced RAG Question Answerer with reranking support and multimodal audio output.

    This implementation extends BaseRAGQuestionAnswerer to support multi-stage retrieval
    with reranking, as well as audio response presentation when requested.
    """

    def __init__(
        self,
        llm,
        indexer,
        reranker=None,
        rerank_topk=None,
        **kwargs,
    ):
        super().__init__(llm=llm, indexer=indexer, **kwargs)
        self.reranker = reranker
        self.rerank_topk = (
            rerank_topk if rerank_topk is not None else max(1, self.search_topk // 2)
        )

    def _apply_reranking(self, pw_ai_results: pw.Table) -> pw.Table:
        """Apply reranking to retrieved documents."""

        @pw.udf
        def add_score_to_doc(doc: pw.Json, score: float) -> dict:
            return {**doc.as_dict(), "reranker_score": score}

        # Flatten docs into rows
        pw_ai_results_exploded = pw_ai_results.flatten(
            pw_ai_results.docs, origin_id="query_id"
        )

        # Apply reranker
        pw_ai_results_scored = pw_ai_results_exploded.select(
            pw.this.prompt,
            pw.this.model,
            pw.this.filters,
            pw.this.return_context_docs,
            pw.this.return_prompt,
            pw.this.return_audio,
            pw.this.query_id,
            doc=pw.this.docs,
            reranker_score=self.reranker(pw.this.docs["text"], pw.this.prompt),
        )

        pw_ai_results_scored = pw_ai_results_scored.await_futures()

        # Add score and sort
        pw_ai_results_scored = pw_ai_results_scored.with_columns(
            sort_key=-pw.this.reranker_score,
            doc_with_score=add_score_to_doc(pw.this.doc, pw.this.reranker_score),
        )

        # Reassemble documents
        pw_ai_results_reranked = (
            pw_ai_results_scored.groupby(
                pw.this.query_id, sort_by=pw.this.sort_key
            )
            .reduce(
                query_id=pw.this.query_id,
                docs=pw.reducers.tuple(pw.this.doc_with_score),
            )
            .with_id(pw.this.query_id)
        )

        # Keep only top k
        pw_ai_results_reranked = pw_ai_results_reranked.with_columns(
            docs=_limit_documents(pw.this.docs, k=self.rerank_topk)
        )

        @pw.udf
        def coalesce_docs(reranked: Any, original: Any) -> tuple:
            if reranked is not None:
                return tuple(reranked)
            if original is not None:
                return tuple(original)
            return ()

        # Join back with original table to ensure queries with 0 retrieved documents are never dropped
        return pw_ai_results.join_left(
            pw_ai_results_reranked,
            id=pw_ai_results.id,
        ).select(
            pw.left.prompt,
            pw.left.model,
            pw.left.filters,
            pw.left.return_context_docs,
            pw.left.return_prompt,
            pw.left.return_audio,
            docs=coalesce_docs(pw.right.docs, pw.left.docs),
        )

    @pw.table_transformer
    def answer_query(self, pw_ai_queries: pw.Table) -> pw.Table:
        """Answer a question based on the available information with document reduction and optional audio output."""

        # Ensure return_audio column exists in query table
        has_return_audio = hasattr(pw_ai_queries, "return_audio") or (
            hasattr(pw_ai_queries, "schema")
            and hasattr(pw_ai_queries.schema, "__columns__")
            and "return_audio" in pw_ai_queries.schema.__columns__
        )
        if not has_return_audio:
            pw_ai_queries = pw_ai_queries.with_columns(
                return_audio=pw.cast(bool, False)
            )

        pw_ai_results = pw_ai_queries + self.indexer.retrieve_query(
            pw_ai_queries.select(
                metadata_filter=pw.this.filters,
                filepath_globpattern=pw.cast(str | None, None),
                query=pw.this.prompt,
                k=self.search_topk,
            )
        ).select(
            docs=pw.this.result,
        )

        if self.reranker is not None:
            pw_ai_results = self._apply_reranking(pw_ai_results)

        pw_ai_results += pw_ai_results.select(
            context=self.docs_to_context_transformer(pw.this.docs)
        )

        pw_ai_results += pw_ai_results.select(
            rag_prompt=self.prompt_udf(pw.this.context, pw.this.prompt)
        )

        accepts_audio = (
            hasattr(self.llm, "_accepts_call_arg")
            and self.llm._accepts_call_arg("return_audio")
        )

        if accepts_audio:
            pw_ai_results += pw_ai_results.select(
                response=self.llm(
                    llms.prompt_chat_single_qa(pw.this.rag_prompt),
                    model=pw.this.model,
                    return_audio=pw.this.return_audio,
                )
            )
        else:
            pw_ai_results += pw_ai_results.select(
                response=self.llm(
                    llms.prompt_chat_single_qa(pw.this.rag_prompt),
                    model=pw.this.model,
                )
            )

        pw_ai_results = pw_ai_results.await_futures()

        @pw.udf
        def custom_prepare_RAG_response(
            prompt: str,
            response: Any,
            docs: list[dict],
            return_context_docs: bool,
            return_prompt: bool,
            return_audio: bool = False,
        ) -> pw.Json:
            api_response: dict = {}

            if isinstance(response, dict):
                api_response["response"] = (
                    response.get("text") or response.get("response", "")
                )
                if return_audio or "audio" in response:
                    if "audio" in response:
                        api_response["audio"] = response["audio"]
                    if "audio_mime_type" in response:
                        api_response["audio_mime_type"] = response["audio_mime_type"]
            elif isinstance(response, pw.Json):
                resp_dict = response.as_dict()
                api_response["response"] = (
                    resp_dict.get("text") or resp_dict.get("response", "")
                )
                if return_audio or "audio" in resp_dict:
                    if "audio" in resp_dict:
                        api_response["audio"] = resp_dict["audio"]
                    if "audio_mime_type" in resp_dict:
                        api_response["audio_mime_type"] = resp_dict["audio_mime_type"]
            else:
                api_response["response"] = str(response) if response is not None else ""

            if return_context_docs:
                api_response["context_docs"] = docs
            if return_prompt:
                api_response["prompt"] = prompt

            return pw.Json(api_response)

        pw_ai_results += pw_ai_results.select(
            result=custom_prepare_RAG_response(
                pw.this.rag_prompt,
                pw.this.response,
                pw.this.docs,
                pw.this.return_context_docs,
                pw.this.return_prompt,
                pw.this.return_audio,
            )
        )

        return pw_ai_results
