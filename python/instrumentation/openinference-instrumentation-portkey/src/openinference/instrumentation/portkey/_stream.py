import inspect
import logging
from types import TracebackType
from typing import (
    Any,
    AsyncIterator,
    Iterator,
    Optional,
    Protocol,
    Tuple,
    Type,
)

from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue
from wrapt import ObjectProxy

from openinference.instrumentation.portkey._utils import _finish_tracing
from openinference.instrumentation.portkey._with_span import _WithSpan

__all__ = (
    "_Stream",
    "_ResponseAccumulator",
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _ResponseAccumulator(Protocol):
    def process_chunk(self, chunk: Any) -> None: ...

    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]: ...

    def get_extra_attributes(self) -> Iterator[Tuple[str, AttributeValue]]: ...


class _Stream(ObjectProxy):  # type: ignore[misc,name-defined,type-arg,unused-ignore]
    """
    Wraps the stream returned by `create(stream=True)`: an `openai.Stream` / `AsyncStream`
    vendored by portkey, portkey's own `Stream` / `AsyncStream`, or a plain (async) generator
    in older releases of the SDK. The span ends once the stream is exhausted, fails, is
    closed, or is abandoned.
    """

    __slots__ = (
        "_self_with_span",
        "_self_iteration_count",
        "_self_is_finished",
        "_self_response_accumulator",
    )

    def __init__(
        self,
        stream: Any,
        with_span: _WithSpan,
        response_accumulator: Optional[_ResponseAccumulator] = None,
    ) -> None:
        super().__init__(stream)
        self._self_with_span = with_span
        self._self_iteration_count = 0
        self._self_is_finished = with_span.is_finished
        self._self_response_accumulator = response_accumulator

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        # pass through mistaken calls
        if not hasattr(self.__wrapped__, "__next__"):
            self.__wrapped__.__next__()
        try:
            chunk: Any = self.__wrapped__.__next__()
        except BaseException as exception:
            if not self._self_is_finished:
                if isinstance(exception, StopIteration):
                    status = trace_api.Status(status_code=trace_api.StatusCode.OK)
                else:
                    status = trace_api.Status(
                        status_code=trace_api.StatusCode.ERROR,
                        # Follow the format in OTEL SDK for description, see:
                        # https://github.com/open-telemetry/opentelemetry-python/blob/2b9dcfc5d853d1c10176937a6bcaade54cda1a31/opentelemetry-api/src/opentelemetry/trace/__init__.py#L588  # noqa E501
                        description=f"{type(exception).__name__}: {exception}",
                    )
                    self._self_with_span.record_exception(exception)
                self._finish_tracing(status=status)
            raise
        else:
            self._process_chunk(chunk)
            return chunk

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        # pass through mistaken calls
        if not hasattr(self.__wrapped__, "__anext__"):
            self.__wrapped__.__anext__()
        try:
            chunk: Any = await self.__wrapped__.__anext__()
        except BaseException as exception:
            if not self._self_is_finished:
                if isinstance(exception, StopAsyncIteration):
                    status = trace_api.Status(status_code=trace_api.StatusCode.OK)
                else:
                    status = trace_api.Status(
                        status_code=trace_api.StatusCode.ERROR,
                        # Follow the format in OTEL SDK for description, see:
                        # https://github.com/open-telemetry/opentelemetry-python/blob/2b9dcfc5d853d1c10176937a6bcaade54cda1a31/opentelemetry-api/src/opentelemetry/trace/__init__.py#L588  # noqa E501
                        description=f"{type(exception).__name__}: {exception}",
                    )
                    self._self_with_span.record_exception(exception)
                self._finish_tracing(status=status)
            raise
        else:
            self._process_chunk(chunk)
            return chunk

    def __enter__(self) -> Any:
        # The SDK's streams return themselves from __enter__, which would let the
        # unwrapped stream escape; return the proxy so iteration inside a `with` block
        # is still traced.
        obj = self.__wrapped__.__enter__()
        if obj is self.__wrapped__:
            return self
        return obj

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        try:
            self.__wrapped__.__exit__(exc_type, exc_val, exc_tb)
        except BaseException as exception:
            self._finish_tracing_with_error(exception)
            raise
        self._finish_tracing_on_exit(exc_val)

    async def __aenter__(self) -> Any:
        obj = await self.__wrapped__.__aenter__()
        if obj is self.__wrapped__:
            return self
        return obj

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        try:
            await self.__wrapped__.__aexit__(exc_type, exc_val, exc_tb)
        except BaseException as exception:
            self._finish_tracing_with_error(exception)
            raise
        self._finish_tracing_on_exit(exc_val)

    def close(self) -> Any:
        is_awaitable = False
        try:
            close = getattr(self.__wrapped__, "close", None)
            if not callable(close):
                # The async generator of older releases of the SDK only has `aclose`.
                close = getattr(self.__wrapped__, "aclose", None)
            result = close() if callable(close) else None
            is_awaitable = inspect.isawaitable(result)
        except BaseException as exception:
            self._finish_tracing_with_error(exception)
            raise
        finally:
            if not is_awaitable:
                self._finish_tracing(status=None)
        # `Stream.close` is synchronous while `AsyncStream.close` is a coroutine, so an
        # awaited close finishes the span once the wrapped close has run.
        if is_awaitable:
            return self._close_and_finish(result)
        return result

    async def aclose(self) -> None:
        try:
            # `AsyncStream` only has `close`, the async generator of older releases only
            # has `aclose`.
            close = getattr(self.__wrapped__, "aclose", None)
            if not callable(close):
                close = getattr(self.__wrapped__, "close", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result
        except BaseException as exception:
            self._finish_tracing_with_error(exception)
            raise
        finally:
            self._finish_tracing(status=None)

    async def _close_and_finish(self, awaitable: Any) -> None:
        try:
            await awaitable
        except BaseException as exception:
            self._finish_tracing_with_error(exception)
            raise
        finally:
            self._finish_tracing(status=None)

    def __del__(self) -> None:
        # Abandoned without any explicit teardown (e.g. `break` out of the loop), possibly
        # never iterated: the span must still be ended, otherwise the call is missing from
        # the trace entirely.
        try:
            self._finish_tracing(status=None)
        except BaseException:
            pass

    def _process_chunk(self, chunk: Any) -> None:
        if not self._self_iteration_count:
            try:
                self._self_with_span.add_event("First Token Stream Event")
            except Exception:
                logger.exception("Failed to add event to span")
        self._self_iteration_count += 1
        if self._self_response_accumulator is not None:
            try:
                self._self_response_accumulator.process_chunk(chunk)
            except Exception:
                logger.exception("Failed to accumulate response")

    def _finish_tracing(
        self,
        status: Optional[trace_api.Status] = None,
    ) -> None:
        if self._self_is_finished:
            return
        _finish_tracing(
            with_span=self._self_with_span,
            attributes=self.get_attributes(),
            extra_attributes=self.get_extra_attributes(),
            status=status,
        )
        self._self_is_finished = True

    def _finish_tracing_on_exit(self, exception: Optional[BaseException]) -> None:
        # Leaving the context manager without exhausting the stream ends the HTTP response
        # but never raises StopIteration, so finish the span here. The status stays UNSET to
        # distinguish a truncated stream from a completed one.
        # GeneratorExit: a generator holding the context was closed, which leaves the stream
        # early in the same way.
        if exception is None or isinstance(exception, GeneratorExit):
            self._finish_tracing(status=None)
        else:
            self._finish_tracing_with_error(exception)

    def _finish_tracing_with_error(self, exception: BaseException) -> None:
        # A stream that fails or is cancelled is an error, unlike one that is just left early.
        self._self_with_span.record_exception(exception)
        self._finish_tracing(
            status=trace_api.Status(
                status_code=trace_api.StatusCode.ERROR,
                description=f"{type(exception).__name__}: {exception}",
            )
        )

    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        if self._self_response_accumulator is not None:
            yield from self._self_response_accumulator.get_attributes()

    def get_extra_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        if self._self_response_accumulator is not None:
            yield from self._self_response_accumulator.get_extra_attributes()
