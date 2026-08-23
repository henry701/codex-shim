from __future__ import annotations

import asyncio

from codex_shim.net.sse import (
    DownstreamWriter,
    PING_BYTES,
    keepalive_interval,
    sse_lines,
    write_bytes,
)


class _FakeContent:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def readany(self):
        await asyncio.sleep(0)
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


class _RecordingResponse:
    prepared = True

    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    async def write(self, data: bytes):
        self.chunks.append(data)

    async def write_eof(self):
        pass


def test_keepalive_interval_env_and_floor(monkeypatch):
    monkeypatch.delenv("CODEX_SHIM_SSE_KEEPALIVE_INTERVAL", raising=False)
    assert keepalive_interval() == 15.0
    monkeypatch.setenv("CODEX_SHIM_SSE_KEEPALIVE_INTERVAL", "7.5")
    assert keepalive_interval() == 7.5
    monkeypatch.setenv("CODEX_SHIM_SSE_KEEPALIVE_INTERVAL", "0.01")
    assert keepalive_interval() == 0.05
    assert keepalive_interval(3.0) == 3.0


async def test_sse_lines_skips_comment_lines():
    class Upstream:
        content = _FakeContent(
            [
                b": ping\n",
                b'data: {"type":"response.created"}\n\n',
                b": keep-alive\n",
                b"data: [DONE]\n\n",
            ]
        )

    lines = [line async for line in sse_lines(Upstream())]
    assert lines == ['{"type":"response.created"}', "[DONE]"]


async def test_write_bytes_goes_through_active_writer():
    response = _RecordingResponse()
    writer = DownstreamWriter(response)
    writer.activate()
    try:
        await write_bytes(response, b"hello")
        await writer.ping()
    finally:
        writer.deactivate()
    assert writer.content_written is True
    assert writer.bytes_written == 5
    assert writer.ping_count == 1
    assert b"hello" in response.chunks
    assert PING_BYTES in response.chunks


async def test_ping_does_not_count_as_content():
    response = _RecordingResponse()
    writer = DownstreamWriter(response)
    await writer.ping()
    assert writer.content_written is False
    assert writer.ping_count == 1
