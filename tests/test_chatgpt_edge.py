from __future__ import annotations

from aiohttp import ClientOSError

from codex_shim.chatgpt_edge import (
    EXHAUSTED_EDGE_MESSAGE,
    is_retryable_chatgpt_edge,
    post_chatgpt_with_retry,
)

HTML_SITE_DOWN = """<!DOCTYPE html>
<html>
<body>
Unable to load site
Visit status.openai.com
Ray ID: 9abc123
</body>
</html>
"""

JSON_403 = '{"error":{"message":"insufficient_quota","type":"insufficient_quota"}}'


class FakeResponse:
    def __init__(self, status: int, text: str, content_type: str) -> None:
        self.status = status
        self.content_type = content_type
        self.headers = {"Content-Type": content_type}
        self._text = text
        self.closed = False

    async def text(self):
        return self._text

    def close(self):
        self.closed = True

    def release(self):
        pass


class FakeSession:
    def __init__(self, outcomes: list[FakeResponse | BaseException]) -> None:
        self.outcomes = list(outcomes)
        self.urls: list[str] = []

    async def post(self, url, json=None, headers=None):
        self.urls.append(str(url))
        item = self.outcomes.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def test_json_403_is_not_retryable():
    assert (
        is_retryable_chatgpt_edge(
            status=403,
            content_type="application/json",
            body=JSON_403,
        )
        is False
    )


def test_html_unable_to_load_site_403_is_retryable():
    assert (
        is_retryable_chatgpt_edge(
            status=403,
            content_type="text/html",
            body=HTML_SITE_DOWN,
        )
        is True
    )


def test_envoy_503_is_retryable():
    assert (
        is_retryable_chatgpt_edge(
            status=503,
            content_type="text/plain",
            body="upstream connect error or disconnect/reset before headers. reset reason: connection failure, transport failure reason: delayed connect error: Connection refused",
        )
        is True
    )


def test_client_os_error_errno_110_is_retryable():
    assert is_retryable_chatgpt_edge(exc=ClientOSError(110, "Connection timed out")) is True


def test_client_os_error_errno_104_is_retryable():
    assert is_retryable_chatgpt_edge(exc=ClientOSError(104, "Connection reset by peer")) is True


def test_connection_reset_error_is_retryable():
    from aiohttp.client_exceptions import ClientConnectionResetError

    assert is_retryable_chatgpt_edge(exc=ClientConnectionResetError("Cannot write to closing transport")) is True


async def test_post_retries_503_then_returns_200(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("codex_shim.net.retry.asyncio.sleep", fake_sleep)
    session = FakeSession(
        [
            FakeResponse(503, "envoy unavailable", "text/plain"),
            FakeResponse(200, "", "text/event-stream"),
        ]
    )
    posted = await post_chatgpt_with_retry(
        session,
        "https://chatgpt.com/backend-api/codex/responses",
        json={"model": "gpt-5.5"},
        headers={},
        attempts=3,
        backoff_base=0.01,
        backoff_factor=2.0,
    )
    assert posted.status == 200
    assert posted.error_text is None
    assert posted.response.status == 200
    assert len(session.urls) == 2
    assert sleeps == [0.01]


async def test_post_exhausts_html_403_as_502_without_html_body(monkeypatch):
    async def fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr("codex_shim.net.retry.asyncio.sleep", fake_sleep)
    session = FakeSession(
        [
            FakeResponse(403, HTML_SITE_DOWN, "text/html"),
            FakeResponse(403, HTML_SITE_DOWN, "text/html"),
            FakeResponse(403, HTML_SITE_DOWN, "text/html"),
        ]
    )
    posted = await post_chatgpt_with_retry(
        session,
        "https://chatgpt.com/backend-api/codex/responses",
        json={"model": "gpt-5.5"},
        headers={},
        attempts=3,
        backoff_base=0.01,
    )
    assert posted.status == 502
    assert posted.error_text == EXHAUSTED_EDGE_MESSAGE
    assert "<html" not in (posted.error_text or "")
    assert "text/html" not in posted.content_type
    assert len(session.urls) == 3


async def test_post_does_not_retry_json_403(monkeypatch):
    session = FakeSession([FakeResponse(403, JSON_403, "application/json")])
    posted = await post_chatgpt_with_retry(
        session,
        "https://chatgpt.com/backend-api/codex/responses",
        json={"model": "gpt-5.5"},
        headers={},
        attempts=3,
    )
    assert posted.status == 403
    assert posted.error_text == JSON_403
    assert len(session.urls) == 1


async def test_post_retries_connect_timeout_then_succeeds(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("codex_shim.net.retry.asyncio.sleep", fake_sleep)
    session = FakeSession(
        [
            ClientOSError(110, "Connection timed out"),
            FakeResponse(200, "", "application/json"),
        ]
    )
    posted = await post_chatgpt_with_retry(
        session,
        "https://chatgpt.com/backend-api/codex/responses",
        json={"model": "gpt-5.5"},
        headers={},
        attempts=3,
        backoff_base=0.25,
        backoff_factor=2.0,
    )
    assert posted.status == 200
    assert len(session.urls) == 2
    assert sleeps == [0.25]


async def test_post_retries_econnreset_then_succeeds(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("codex_shim.net.retry.asyncio.sleep", fake_sleep)
    session = FakeSession(
        [
            ClientOSError(104, "Connection reset by peer"),
            FakeResponse(200, "", "application/json"),
        ]
    )
    posted = await post_chatgpt_with_retry(
        session,
        "https://chatgpt.com/backend-api/codex/responses",
        json={"model": "gpt-5.5"},
        headers={},
        attempts=3,
        backoff_base=0.25,
        backoff_factor=2.0,
    )
    assert posted.status == 200
    assert len(session.urls) == 2
    assert sleeps == [0.25]


async def test_compact_404_is_not_retryable(monkeypatch):
    session = FakeSession([FakeResponse(404, '{"detail":"Not Found"}', "application/json")])
    posted = await post_chatgpt_with_retry(
        session,
        "https://chatgpt.com/backend-api/codex/responses/compact",
        json={"model": "gpt-5.5"},
        headers={},
        attempts=3,
    )
    assert posted.status == 404
    assert posted.error_text == '{"detail":"Not Found"}'
    assert len(session.urls) == 1


def test_html_403_classifier_remains_edge_owned():
    from codex_shim.net.errors import is_retryable_status

    assert is_retryable_status(403) is False
    assert is_retryable_chatgpt_edge(status=403, content_type="text/html", body=HTML_SITE_DOWN) is True
    assert is_retryable_chatgpt_edge(status=403, content_type="application/json", body=JSON_403) is False


async def test_chatgpt_terminal_without_done_only_appends_done():
    from codex_shim.net.emitters import ChatgptRelayEmitter

    class Recording:
        prepared = True

        def __init__(self) -> None:
            self.chunks: list[bytes] = []

        async def write(self, data: bytes):
            self.chunks.append(data)

        async def write_eof(self):
            pass

    response = Recording()
    emitter = ChatgptRelayEmitter(model="test")
    emitter.observe({"type": "response.failed", "response": {"id": "r1", "status": "failed"}})
    await emitter.fail(response, "upstream closed", code="upstream_disconnect")
    await emitter.complete(response, upstream_saw_done=False)
    joined = b"".join(response.chunks)
    assert joined.count(b"response.failed") == 0
    assert joined.count(b"data: [DONE]") == 1
