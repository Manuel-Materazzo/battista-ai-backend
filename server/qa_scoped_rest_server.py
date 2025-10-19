import logging
import pathway as pw
from typing import Callable
from pathway.xpacks.llm.question_answering import BaseQuestionAnswerer
from pathway.xpacks.llm.servers import BaseRestServer

logger = logging.getLogger(__name__)

class QAScopedRestServer(BaseRestServer):
    """
    Creates a REST Server for answering queries to a given instance of ``BaseQuestionAnswerer``.
    It exposes four endpoints:
    - ``/v1/retrieve`` which is answered using ``retrieve`` method,
    - ``/v2/answer`` which is answered using ``answer_query`` method,

    Args:
        host: host on which server will run
        port: port on which server will run
        rag_question_answerer: instance of ``BaseQuestionAnswerer`` which is used
            to answer queries received in the endpoints.
        rest_kwargs: optional kwargs to be passed to ``pw.io.http.rest_connector``
    """

    def __init__(
        self,
        host: str,
        port: int,
        rag_question_answerer: BaseQuestionAnswerer,
        **rest_kwargs,
    ):
        super().__init__(host, port, **rest_kwargs)

        logger.info("Inizializing QAScopedRestServer")

        self.serve(
            "/v1/retrieve",
            rag_question_answerer.RetrieveQuerySchema,
            rag_question_answerer.retrieve,
            **rest_kwargs,
        )
        self.serve(
            "/v2/answer",
            rag_question_answerer.AnswerQuerySchema,
            rag_question_answerer.answer_query,
            **rest_kwargs,
        )
        self.serve(
            "/v2/list_documents",
            rag_question_answerer.InputsQuerySchema,
            rag_question_answerer.list_documents,
            **rest_kwargs,
        )

    def serve(
        self,
        route: str,
        schema: type[pw.Schema],
        handler: Callable[[pw.Table], pw.Table],
        **additional_endpoint_kwargs,
    ):

        queries, writer = pw.io.http.rest_connector(
            webserver=self.webserver,
            route=route,
            schema=schema,
            autocommit_duration_ms=50,
            delete_completed_queries=False,
            **additional_endpoint_kwargs,
        )
        writer(handler(queries))
