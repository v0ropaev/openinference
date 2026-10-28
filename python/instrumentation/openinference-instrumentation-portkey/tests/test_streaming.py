import gc
import inspect
import json
from importlib import import_module
from importlib.metadata import version
from typing import Any, AsyncIterator, Dict, Iterator, List

import httpx
import pytest
import respx
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.trace import StatusCode

from openinference.instrumentation.portkey import PortkeyInstrumentor
from openinference.semconv.trace import MessageAttributes, SpanAttributes, ToolCallAttributes

_OUTPUT_MESSAGE = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
_PORTKEY_MAJOR = int(version("portkey-ai").split(".")[0])
_MESSAGES = [{"role": "user", "content": "Hi"}]


def _chunk(delta: Dict[str, Any], finish_reason: Any = None, **extra: Any) -> Dict[str, Any]:
    return {
        "id": "chatcmpl-stream",
        "object": "chat.completion.chunk",
        "created": 1750000000,
        "model": "gpt-4o-mini-2024-07-18",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        **extra,
    }


_TEXT_CHUNKS = [
    _chunk({"role": "assistant", "content": "Hel"}),
    _chunk({"content": "lo!"}),
    _chunk({}, finish_reason="stop"),
    {
        **_chunk({}),
        "choices": [],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    },
]


def _sse(chunks: List[Dict[str, Any]], done: bool = True) -> bytes:
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    return (body + ("data: [DONE]\n\n" if done else "")).encode()


def _stream_response(content: Any) -> httpx.Response:
    kwargs: Dict[str, Any] = (
        {"content": content} if isinstance(content, bytes) else {"stream": content}
    )
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, **kwargs)


class _DroppedStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    """Sends the first chunk, then the connection drops."""

    def __iter__(self) -> Iterator[bytes]:
        yield _sse(_TEXT_CHUNKS[:1], done=False)
        raise httpx.ReadError("connection dropped")

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield _sse(_TEXT_CHUNKS[:1], done=False)
        raise httpx.ReadError("connection dropped")


def _assert_text_output(attributes: Dict[str, Any]) -> None:
    assert attributes[f"{_OUTPUT_MESSAGE}.{MessageAttributes.MESSAGE_ROLE}"] == "assistant"
    assert attributes[f"{_OUTPUT_MESSAGE}.{MessageAttributes.MESSAGE_CONTENT}"] == "Hello!"
    assert attributes[SpanAttributes.LLM_MODEL_NAME] == "gpt-4o-mini-2024-07-18"
    assert attributes[SpanAttributes.LLM_FINISH_REASON] == "stop"
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 5
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 2
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 7
    assert attributes[SpanAttributes.OUTPUT_MIME_TYPE] == "application/json"
    output = json.loads(attributes[SpanAttributes.OUTPUT_VALUE])
    assert output["choices"][0]["message"] == {"role": "assistant", "content": "Hello!"}
    assert output["usage"]["total_tokens"] == 7


def test_chat_completion_stream(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_portkey_instrumentation: None,
) -> None:
    client = import_module("portkey_ai").Portkey(api_key="REDACTED", virtual_key="REDACTED")
    with respx.mock(base_url="https://api.portkey.ai", assert_all_called=True) as respx_mock:
        respx_mock.post("/v1/chat/completions").mock(
            return_value=_stream_response(_sse(_TEXT_CHUNKS))
        )
        stream = client.chat.completions.create(
            messages=_MESSAGES, model="gpt-4o-mini", stream=True
        )
        assert not in_memory_span_exporter.get_finished_spans()
        text = "".join(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices)

    assert text == "Hello!"
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code is StatusCode.OK
    assert [event.name for event in span.events] == ["First Token Stream Event"]
    _assert_text_output(dict(span.attributes or {}))


async def test_async_chat_completion_stream(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_portkey_instrumentation: None,
) -> None:
    client = import_module("portkey_ai").AsyncPortkey(api_key="REDACTED", virtual_key="REDACTED")
    with respx.mock(base_url="https://api.portkey.ai", assert_all_called=True) as respx_mock:
        respx_mock.post("/v1/chat/completions").mock(
            return_value=_stream_response(_sse(_TEXT_CHUNKS))
        )
        stream = await client.chat.completions.create(
            messages=_MESSAGES, model="gpt-4o-mini", stream=True
        )
        assert not in_memory_span_exporter.get_finished_spans()
        text = "".join(
            [chunk.choices[0].delta.content or "" async for chunk in stream if chunk.choices]
        )

    assert text == "Hello!"
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code is StatusCode.OK
    _assert_text_output(dict(span.attributes or {}))


def test_chat_completion_stream_with_tool_calls(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_portkey_instrumentation: None,
) -> None:
    tool_call = {"index": 0, "id": "call_1", "type": "function"}
    chunks = [
        _chunk(
            {
                "role": "assistant",
                "tool_calls": [{**tool_call, "function": {"name": "get_weather", "arguments": ""}}],
            }
        ),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"city": '}}]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]}),
        _chunk({}, finish_reason="tool_calls"),
    ]
    client = import_module("portkey_ai").Portkey(api_key="REDACTED", virtual_key="REDACTED")
    with respx.mock(base_url="https://api.portkey.ai", assert_all_called=True) as respx_mock:
        respx_mock.post("/v1/chat/completions").mock(return_value=_stream_response(_sse(chunks)))
        for _ in client.chat.completions.create(
            messages=_MESSAGES, model="gpt-4o-mini", stream=True
        ):
            pass

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})
    prefix = f"{_OUTPUT_MESSAGE}.{MessageAttributes.MESSAGE_TOOL_CALLS}.0"
    assert attributes[f"{prefix}.{ToolCallAttributes.TOOL_CALL_ID}"] == "call_1"
    assert attributes[f"{prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"] == "get_weather"
    assert (
        attributes[f"{prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"]
        == '{"city": "Paris"}'
    )
    assert attributes[SpanAttributes.LLM_FINISH_REASON] == "tool_calls"


_PROMPT_TEXT_CHUNKS = [
    # a text model streams `text` instead of a `delta`
    {
        "id": "cmpl-stream",
        "object": "text_completion",
        "created": 1750000000,
        "model": "m",
        "choices": [{"index": 0, "text": "Hel", "finish_reason": None}],
    },
    {
        "id": "cmpl-stream",
        "object": "text_completion",
        "created": 1750000000,
        "model": "m",
        "choices": [{"index": 0, "text": "lo!", "finish_reason": "stop"}],
    },
]


def _assert_prompt_span(spans: Any, chunks: str) -> None:
    assert len(spans) == 1
    assert spans[0].status.status_code is StatusCode.OK
    attributes = dict(spans[0].attributes or {})
    assert attributes[SpanAttributes.PROMPT_ID] == "pp-test"
    output = json.loads(attributes[SpanAttributes.OUTPUT_VALUE])
    if chunks == "text":
        assert output["choices"][0]["text"] == "Hello!"
        assert attributes[SpanAttributes.LLM_FINISH_REASON] == "stop"
    else:
        assert attributes[f"{_OUTPUT_MESSAGE}.{MessageAttributes.MESSAGE_CONTENT}"] == "Hello!"
        assert attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 7


@pytest.mark.parametrize("chunks", ["delta", "text"])
def test_prompt_completion_stream(
    chunks: str,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_portkey_instrumentation: None,
) -> None:
    body = _sse(_TEXT_CHUNKS if chunks == "delta" else _PROMPT_TEXT_CHUNKS)
    client = import_module("portkey_ai").Portkey(api_key="REDACTED", virtual_key="REDACTED")
    with respx.mock(base_url="https://api.portkey.ai", assert_all_called=True) as respx_mock:
        respx_mock.post("/v1/prompts/pp-test/completions").mock(return_value=_stream_response(body))
        stream = client.prompts.completions.create(prompt_id="pp-test", stream=True)
        assert not in_memory_span_exporter.get_finished_spans()
        for _ in stream:
            pass

    _assert_prompt_span(in_memory_span_exporter.get_finished_spans(), chunks)


@pytest.mark.parametrize("chunks", ["delta", "text"])
async def test_async_prompt_completion_stream(
    chunks: str,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_portkey_instrumentation: None,
) -> None:
    body = _sse(_TEXT_CHUNKS if chunks == "delta" else _PROMPT_TEXT_CHUNKS)
    client = import_module("portkey_ai").AsyncPortkey(api_key="REDACTED", virtual_key="REDACTED")
    with respx.mock(base_url="https://api.portkey.ai", assert_all_called=True) as respx_mock:
        respx_mock.post("/v1/prompts/pp-test/completions").mock(return_value=_stream_response(body))
        stream = await client.prompts.completions.create(prompt_id="pp-test", stream=True)
        assert not in_memory_span_exporter.get_finished_spans()
        async for _ in stream:
            pass

    _assert_prompt_span(in_memory_span_exporter.get_finished_spans(), chunks)


@pytest.mark.parametrize("teardown", ["break", "close"])
def test_chat_completion_stream_left_early(
    teardown: str,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_portkey_instrumentation: None,
) -> None:
    client = import_module("portkey_ai").Portkey(api_key="REDACTED", virtual_key="REDACTED")
    with respx.mock(base_url="https://api.portkey.ai", assert_all_called=True) as respx_mock:
        respx_mock.post("/v1/chat/completions").mock(
            return_value=_stream_response(_sse(_TEXT_CHUNKS))
        )
        stream = client.chat.completions.create(
            messages=_MESSAGES, model="gpt-4o-mini", stream=True
        )
        for _ in stream:
            break
        assert not in_memory_span_exporter.get_finished_spans()
        if teardown == "close":
            stream.close()
        else:
            del stream
            gc.collect()

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    # A stream left early is truncated, not failed: the status stays UNSET.
    assert span.status.status_code is StatusCode.UNSET
    attributes = dict(span.attributes or {})
    assert attributes[f"{_OUTPUT_MESSAGE}.{MessageAttributes.MESSAGE_CONTENT}"] == "Hel"


@pytest.mark.parametrize("teardown", ["break", "close", "aclose", "async_with"])
async def test_async_chat_completion_stream_left_early(
    teardown: str,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_portkey_instrumentation: None,
) -> None:
    if teardown == "async_with" and _PORTKEY_MAJOR < 2:
        pytest.skip("streams are plain async generators before portkey-ai 2")
    client = import_module("portkey_ai").AsyncPortkey(api_key="REDACTED", virtual_key="REDACTED")
    with respx.mock(base_url="https://api.portkey.ai", assert_all_called=True) as respx_mock:
        respx_mock.post("/v1/chat/completions").mock(
            return_value=_stream_response(_sse(_TEXT_CHUNKS))
        )
        stream = await client.chat.completions.create(
            messages=_MESSAGES, model="gpt-4o-mini", stream=True
        )
        raw = stream.__wrapped__
        if teardown == "async_with":
            async with stream as entered:
                async for _ in entered:
                    break
                assert not in_memory_span_exporter.get_finished_spans()
        else:
            async for _ in stream:
                break
            assert not in_memory_span_exporter.get_finished_spans()
            if teardown == "close":
                await stream.close()
            elif teardown == "aclose":
                await stream.aclose()
            else:
                del stream
                gc.collect()
        if teardown != "break":
            # the underlying stream is closed, not just the span
            if inspect.isasyncgen(raw):
                assert raw.ag_frame is None
            else:
                assert raw.response.is_closed

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code is StatusCode.UNSET
    attributes = dict(span.attributes or {})
    assert attributes[f"{_OUTPUT_MESSAGE}.{MessageAttributes.MESSAGE_CONTENT}"] == "Hel"


def test_chat_completion_stream_not_recording(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    tracer_provider = TracerProvider(sampler=ALWAYS_OFF)
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    PortkeyInstrumentor().instrument(tracer_provider=tracer_provider)
    try:
        client = import_module("portkey_ai").Portkey(api_key="REDACTED", virtual_key="REDACTED")
        with respx.mock(base_url="https://api.portkey.ai", assert_all_called=True) as respx_mock:
            respx_mock.post("/v1/chat/completions").mock(
                return_value=_stream_response(_sse(_TEXT_CHUNKS))
            )
            stream = client.chat.completions.create(
                messages=_MESSAGES, model="gpt-4o-mini", stream=True
            )
            # a sampled-out span never reads the merged chunks, so none are merged
            assert stream._self_response_accumulator is None
            text = "".join(
                chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices
            )
    finally:
        PortkeyInstrumentor().uninstrument()

    assert text == "Hello!"
    assert not in_memory_span_exporter.get_finished_spans()


@pytest.mark.skipif(
    _PORTKEY_MAJOR < 2,
    reason="streams are plain generators before portkey-ai 2",
)
def test_chat_completion_stream_as_context_manager(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_portkey_instrumentation: None,
) -> None:
    client = import_module("portkey_ai").Portkey(api_key="REDACTED", virtual_key="REDACTED")
    with respx.mock(base_url="https://api.portkey.ai", assert_all_called=True) as respx_mock:
        respx_mock.post("/v1/chat/completions").mock(
            return_value=_stream_response(_sse(_TEXT_CHUNKS))
        )
        with client.chat.completions.create(
            messages=_MESSAGES, model="gpt-4o-mini", stream=True
        ) as stream:
            assert isinstance(stream.response, httpx.Response)
            for _ in stream:
                pass

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code is StatusCode.OK
    _assert_text_output(dict(spans[0].attributes or {}))


def test_chat_completion_stream_error(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_portkey_instrumentation: None,
) -> None:
    client = import_module("portkey_ai").Portkey(api_key="REDACTED", virtual_key="REDACTED")
    with respx.mock(base_url="https://api.portkey.ai", assert_all_called=True) as respx_mock:
        respx_mock.post("/v1/chat/completions").mock(
            return_value=_stream_response(_DroppedStream())
        )
        stream = client.chat.completions.create(
            messages=_MESSAGES, model="gpt-4o-mini", stream=True
        )
        with pytest.raises(httpx.ReadError):
            for _ in stream:
                pass

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description == "ReadError: connection dropped"
    assert [event.name for event in span.events] == ["First Token Stream Event", "exception"]
    attributes = dict(span.attributes or {})
    assert attributes[f"{_OUTPUT_MESSAGE}.{MessageAttributes.MESSAGE_CONTENT}"] == "Hel"


async def test_async_chat_completion_stream_error(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_portkey_instrumentation: None,
) -> None:
    client = import_module("portkey_ai").AsyncPortkey(api_key="REDACTED", virtual_key="REDACTED")
    with respx.mock(base_url="https://api.portkey.ai", assert_all_called=True) as respx_mock:
        respx_mock.post("/v1/chat/completions").mock(
            return_value=_stream_response(_DroppedStream())
        )
        stream = await client.chat.completions.create(
            messages=_MESSAGES, model="gpt-4o-mini", stream=True
        )
        with pytest.raises(httpx.ReadError):
            async for _ in stream:
                pass

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code is StatusCode.ERROR
    assert spans[0].status.description == "ReadError: connection dropped"
