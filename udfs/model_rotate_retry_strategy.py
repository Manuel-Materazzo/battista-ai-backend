import asyncio
import random
import logging
from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar
logger = logging.getLogger(__name__)

from pathway.internals.udfs.retries import AsyncRetryStrategy

T = TypeVar("T")
P = ParamSpec("P")


class ModelRotateRetryStrategy(AsyncRetryStrategy):
    """Retry strategy with exponential backoff with jitter and maximum retries."""

    _model_list: list[str]
    _initial_delay: float
    _backoff_factor: float
    _jitter: float

    def __init__(
            self,
            model_list: list[str] = [],
            initial_delay: int = 1_000,
            backoff_factor: float = 2,
            jitter_ms: int = 300,
    ) -> None:
        """
        Args:
            model_list: List of model names to rotate.
            initial_delay: First delay in milliseconds.
            backoff_factor: Factor by which the delay between retries increases exponentially. Set as float.
            jitter_ms: Maximum random jitter added to the delay between retries in milliseconds.
        Returns:
            None"""
        logger.info("Inizializing ModelRotateRetryStrategy")
        self._initial_delay = initial_delay / 1_000
        self._model_list = model_list
        self._backoff_factor = backoff_factor
        self._jitter = jitter_ms / 1_000

    async def invoke(
            self, func: Callable[P, Awaitable[T]], /, *args: P.args, **kwargs: P.kwargs
    ) -> T:
        delay = self._initial_delay

        logger.info("ModelRotateRetryStrategy invoked, kwargs: %s", kwargs)

        # set max attempt based on the number of models
        for n_attempt in range(0, len(self._model_list)):
            try:
                # Change model based on attempt count
                kwargs["model"] = self._model_list[n_attempt]
                logger.info("Trying web request with model %s", kwargs["model"])
                return await func(*args, **kwargs)
            except Exception:
                if n_attempt == len(self._model_list):
                    raise
            await asyncio.sleep(delay)
            delay = self._next_delay(delay)
        raise ValueError(f"incorrect max retries: {len(self._model_list)}")

    def _next_delay(self, current_delay: float) -> float:
        current_delay *= self._backoff_factor
        current_delay += random.random() * self._jitter
        return current_delay
