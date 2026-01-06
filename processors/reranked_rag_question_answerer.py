import logging
import pathway as pw
from pathway.xpacks.llm import llms, rerankers
from pathway.xpacks.llm.question_answering import BaseRAGQuestionAnswerer, _prepare_RAG_response, _limit_documents

logger = logging.getLogger(__name__)


class RerankedRAGQuestionAnswerer(BaseRAGQuestionAnswerer):
    """
    Enhanced RAG Question Answerer with reranking support.

    This implementation uses a simpler approach that works within Pathway's constraints.
    """

    def __init__(
            self,
            llm,
            indexer,
            reranker=None,
            rerank_topk=None,
            **kwargs
    ):
        super().__init__(llm=llm, indexer=indexer, **kwargs)
        self.reranker = reranker
        self.rerank_topk = rerank_topk if rerank_topk is not None else max(1, self.search_topk // 2)

    def _apply_reranking(self, pw_ai_results: pw.Table) -> pw.Table:
        """Apply reranking to retrieved documents."""

        @pw.udf
        def add_score_to_doc(doc: pw.Json, score: float) -> dict:
            return {**doc.as_dict(), "reranker_score": score}

        # Flatten docs into rows
        pw_ai_results_exploded = pw_ai_results.flatten(pw_ai_results.docs, origin_id='query_id')

        # Apply reranker
        pw_ai_results_scored = pw_ai_results_exploded.select(
            pw.this.prompt,
            pw.this.model,
            pw.this.filters,
            pw.this.return_context_docs,
            pw.this.return_prompt,
            pw.this.query_id,
            doc=pw.this.docs,
            reranker_score=self.reranker(pw.this.docs["text"], pw.this.prompt)
        )

        pw_ai_results_scored = pw_ai_results_scored.await_futures()

        # Add score and sort
        pw_ai_results_scored = pw_ai_results_scored.with_columns(
            sort_key=-pw.this.reranker_score,
            doc_with_score=add_score_to_doc(pw.this.doc, pw.this.reranker_score)
        )

        # Reassemble documents
        pw_ai_results = pw_ai_results_scored.groupby(pw.this.query_id, sort_by=pw.this.sort_key).reduce(
            query_id=pw.this.query_id,
            prompt=pw.reducers.any(pw.this.prompt),
            model=pw.reducers.any(pw.this.model),
            filters=pw.reducers.any(pw.this.filters),
            return_context_docs=pw.reducers.any(pw.this.return_context_docs),
            return_prompt=pw.reducers.any(pw.this.return_prompt),
            docs=pw.reducers.tuple(pw.this.doc_with_score),
        ).with_id(pw.this.query_id)

        # Keep only top k
        return pw_ai_results.with_columns(
            docs=_limit_documents(pw.this.docs, k=self.rerank_topk)
        )

    @pw.table_transformer
    def answer_query(self, pw_ai_queries: pw.Table) -> pw.Table:
        """Answer a question based on the available information with document reduction."""

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

        pw_ai_results += pw_ai_results.select(
            response=self.llm(
                llms.prompt_chat_single_qa(pw.this.rag_prompt),
                model=pw.this.model,
            )
        )

        pw_ai_results = pw_ai_results.await_futures()

        @pw.udf
        def custom_prepare_RAG_response(
                prompt: str, response: str, docs: list[dict], return_context_docs: bool, return_prompt: bool
        ) -> pw.Json:
            api_response: dict = {"response": response}
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
                pw.this.return_prompt
            )
        )

        return pw_ai_results
