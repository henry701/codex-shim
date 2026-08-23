from __future__ import annotations

from io import BytesIO
import json
from urllib.error import HTTPError, URLError

import pytest

from codex_shim.net.errors import is_retryable_exception, parse_upstream_error
from codex_shim.net.retry import RetryPolicy, request_urllib, retry_aiohttp_post, retry_policy_from_env


class _Resp:
    def __init__(self, body: bytes = b'{"ok":true}', status: int = 200) -> None:
        self._body = body
        self.status = status
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._body


class _AiohttpResponse:
    def __init__(
        self,
        status: int,
        text: str,
        content_type: str = "text/plain",
        headers: dict[str, str] | None = None,
        text_error: BaseException | None = None,
    ) -> None:
        self.status = status
        self.content_type = content_type
        self.headers = headers or {}
        self._text = text
        self._text_error = text_error
        self.closed = False

    async def text(self):
        if self._text_error is not None:
            raise self._text_error
        return self._text

    def close(self):
        self.closed = True

    def release(self):
        pass


class _AiohttpSession:
    def __init__(self, outcomes: list[_AiohttpResponse | BaseException]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    async def post(self, url, json=None, headers=None):
        self.calls += 1
        item = self.outcomes.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def test_request_urllib_retries_urlerror_then_succeeds():
    sleeps: list[float] = []
    attempts = {"n": 0}

    def fake_urlopen(request, timeout=20.0):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise URLError("temporary")
        return _Resp()

    result = request_urllib(
        "https://example.test/models",
        policy=RetryPolicy(attempts=3, backoff_base=0.01, backoff_factor=2.0),
        urlopen_fn=fake_urlopen,
        sleep_fn=sleeps.append,
        ensure_json=True,
    )
    assert json.loads(result.body.decode()) == {"ok": True}
    assert attempts["n"] == 3
    assert sleeps == pytest.approx([0.01, 0.02])


def test_request_urllib_retries_502_then_succeeds():
    attempts = {"n": 0}

    def fake_urlopen(request, timeout=20.0):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise HTTPError(
                "https://example.test/models",
                502,
                "Bad Gateway",
                hdrs=None,
                fp=BytesIO(b"nope"),
            )
        return _Resp()

    result = request_urllib(
        "https://example.test/models",
        policy=RetryPolicy(attempts=3, backoff_base=0.0),
        urlopen_fn=fake_urlopen,
        sleep_fn=lambda _delay: None,
    )
    assert attempts["n"] == 2
    assert result.status == 200


def test_request_urllib_does_not_retry_400():
    attempts = {"n": 0}

    def fake_urlopen(request, timeout=20.0):
        attempts["n"] += 1
        raise HTTPError(
            "https://example.test/models",
            400,
            "Bad Request",
            hdrs=None,
            fp=BytesIO(b'{"error":"nope"}'),
        )

    with pytest.raises(HTTPError) as exc:
        request_urllib(
            "https://example.test/models",
            policy=RetryPolicy(attempts=3, backoff_base=0.0),
            urlopen_fn=fake_urlopen,
            sleep_fn=lambda _delay: None,
        )
    assert exc.value.code == 400
    assert attempts["n"] == 1


def test_request_urllib_attempts_1_does_not_retry_oserror():
    attempts = {"n": 0}

    def fake_urlopen(request, timeout=20.0):
        attempts["n"] += 1
        raise OSError("portal down")

    with pytest.raises(OSError, match="portal down"):
        request_urllib(
            "https://example.test/oauth",
            policy=RetryPolicy(attempts=1, retry_json_decode=False),
            urlopen_fn=fake_urlopen,
        )
    assert attempts["n"] == 1


def test_request_urllib_raise_on_http_error_false_returns_status():
    def fake_urlopen(request, timeout=20.0):
        raise HTTPError(
            "https://example.test/go",
            401,
            "Unauthorized",
            hdrs=None,
            fp=BytesIO(b'{"error":"nope"}'),
        )

    result = request_urllib(
        "https://example.test/go",
        policy=RetryPolicy(attempts=2, backoff_base=0.0),
        raise_on_http_error=False,
        urlopen_fn=fake_urlopen,
        sleep_fn=lambda _delay: None,
    )
    assert result.status == 401
    assert result.body == b'{"error":"nope"}'


def test_request_urllib_retries_invalid_json_when_ensure_json():
    attempts = {"n": 0}

    def fake_urlopen(request, timeout=20.0):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return _Resp(b"not-json")
        return _Resp(b'{"ok":true}')

    result = request_urllib(
        "https://example.test/models",
        policy=RetryPolicy(attempts=3, backoff_base=0.0, retry_json_decode=True),
        urlopen_fn=fake_urlopen,
        sleep_fn=lambda _delay: None,
        ensure_json=True,
    )
    assert json.loads(result.body.decode()) == {"ok": True}
    assert attempts["n"] == 2


def test_retry_policy_from_env(monkeypatch):
    monkeypatch.setenv("CODEX_SHIM_RETRY_ATTEMPTS", "5")
    monkeypatch.setenv("CODEX_SHIM_RETRY_BACKOFF_BASE", "0.25")
    monkeypatch.setenv("CODEX_SHIM_RETRY_BACKOFF_FACTOR", "3")
    policy = retry_policy_from_env()
    assert policy.attempts == 5
    assert policy.backoff_base == 0.25
    assert policy.backoff_factor == 3.0
    assert policy.delay_for(0) == 0.25
    assert policy.delay_for(1) == 0.75


def test_parse_upstream_error_and_retryable_exception():
    code, message = parse_upstream_error('{"error":{"message":"nope","type":"overflow"}}', 400)
    assert code == "overflow"
    assert message == "nope"
    assert is_retryable_exception(TimeoutError("late")) is True
    assert is_retryable_exception(ValueError("nope")) is False


def test_parse_upstream_error_unwraps_openrouter_metadata_raw():
    body = json.dumps(
        {
            "error": {
                "message": "Provider returned error",
                "code": 429,
                "metadata": {
                    "raw": (
                        "google/gemma-4-26b-a4b-it:free is temporarily rate-limited "
                        "upstream. Please retry shortly."
                    ),
                    "provider_name": "Google AI Studio",
                    "provider_error_code": "upstream_429",
                    "limit_source": "upstream_provider_shared_pool",
                    "retry_after_seconds": 5,
                },
            }
        }
    )
    code, message = parse_upstream_error(body, 429)
    assert code == "upstream_429"
    assert "temporarily rate-limited" in message
    assert message != "Provider returned error"


def test_parse_upstream_error_keeps_specific_message_when_raw_is_present():
    body = json.dumps(
        {
            "error": {
                "message": "Free promotion has ended for MiniMax M3 Free",
                "type": "ModelError",
                "metadata": {"raw": "provider detail"},
            }
        }
    )
    code, message = parse_upstream_error(body, 400)
    assert code == "ModelError"
    assert message == "Free promotion has ended for MiniMax M3 Free"


async def test_retry_aiohttp_post_retries_503(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("codex_shim.net.retry.asyncio.sleep", fake_sleep)
    session = _AiohttpSession(
        [
            _AiohttpResponse(503, "envoy unavailable"),
            _AiohttpResponse(200, "", "text/event-stream"),
        ]
    )
    posted = await retry_aiohttp_post(
        session,
        "https://example.test/v1",
        json={"model": "x"},
        policy=RetryPolicy(attempts=3, backoff_base=0.5, backoff_factor=2.0),
        label="test-post",
    )
    assert posted.status == 200
    assert session.calls == 2
    assert sleeps == [0.5]


async def test_retry_aiohttp_post_honors_retry_after_header(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("codex_shim.net.retry.asyncio.sleep", fake_sleep)
    session = _AiohttpSession(
        [
            _AiohttpResponse(429, "slow down", headers={"Retry-After": "5"}),
            _AiohttpResponse(200, "", "text/event-stream"),
        ]
    )
    posted = await retry_aiohttp_post(
        session,
        "https://openrouter.ai/api/v1/chat/completions",
        json={"model": "x"},
        policy=RetryPolicy(attempts=3, backoff_base=0.5, backoff_factor=2.0),
        label="or-test",
    )
    assert posted.status == 200
    assert session.calls == 2
    assert sleeps == [5.0]


async def test_retry_aiohttp_post_honors_openrouter_retry_after_body(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("codex_shim.net.retry.asyncio.sleep", fake_sleep)
    body = json.dumps(
        {
            "error": {
                "message": "Provider returned error",
                "code": 429,
                "metadata": {
                    "retry_after_seconds": 5,
                    "headers": {"Retry-After": "5"},
                },
            }
        }
    )
    session = _AiohttpSession(
        [
            _AiohttpResponse(429, body, content_type="application/json"),
            _AiohttpResponse(200, "", "text/event-stream"),
        ]
    )
    posted = await retry_aiohttp_post(
        session,
        "https://openrouter.ai/api/v1/chat/completions",
        json={"model": "x"},
        policy=RetryPolicy(attempts=3, backoff_base=0.5, backoff_factor=2.0),
    )
    assert posted.status == 200
    assert sleeps == [5.0]


async def test_retry_aiohttp_post_extends_past_attempts_when_retry_after(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("codex_shim.net.retry.asyncio.sleep", fake_sleep)
    body = json.dumps(
        {
            "error": {
                "message": "Provider returned error",
                "code": 429,
                "metadata": {"retry_after_seconds": 5, "headers": {"Retry-After": "5"}},
            }
        }
    )
    session = _AiohttpSession(
        [_AiohttpResponse(429, body, content_type="application/json")] * 4
        + [_AiohttpResponse(200, "", "text/event-stream")]
    )
    posted = await retry_aiohttp_post(
        session,
        "https://openrouter.ai/api/v1/chat/completions",
        json={"model": "x"},
        policy=RetryPolicy(attempts=3, backoff_base=0.5, backoff_factor=2.0, wait_budget=20.0),
    )
    assert posted.status == 200
    assert session.calls == 5
    assert sleeps == [5.0, 5.0, 5.0, 5.0]


async def test_retry_aiohttp_post_attempts_1_does_not_extend_on_retry_after(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("codex_shim.net.retry.asyncio.sleep", fake_sleep)
    session = _AiohttpSession(
        [_AiohttpResponse(429, "slow", headers={"Retry-After": "5"})]
    )
    posted = await retry_aiohttp_post(
        session,
        "https://example.test/oauth",
        json={"model": "x"},
        policy=RetryPolicy(attempts=1, wait_budget=30.0),
    )
    assert posted.status == 429
    assert session.calls == 1
    assert sleeps == []


async def test_retry_aiohttp_post_defaults_429_wait_without_retry_after(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("codex_shim.net.retry.asyncio.sleep", fake_sleep)
    session = _AiohttpSession(
        [
            _AiohttpResponse(
                429,
                '{"type":"error","error":{"type":"FreeUsageLimitError","message":"Rate limit exceeded"}}',
                content_type="application/json",
            ),
            _AiohttpResponse(200, "", "text/event-stream"),
        ]
    )
    posted = await retry_aiohttp_post(
        session,
        "https://opencode.ai/zen/v1/chat/completions",
        json={"model": "x"},
        policy=RetryPolicy(attempts=3, backoff_base=0.5, backoff_factor=2.0),
    )
    assert posted.status == 200
    assert sleeps == [5.0]


async def test_aiohttp_retryable_status_body_read_failure_retries_and_closes_response(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("codex_shim.net.retry.asyncio.sleep", fake_sleep)
    first = _AiohttpResponse(503, "", text_error=ConnectionError("truncated"))
    session = _AiohttpSession([first, _AiohttpResponse(200, "", "text/event-stream")])
    posted = await retry_aiohttp_post(
        session,
        "https://example.test/v1",
        json={"model": "x"},
        policy=RetryPolicy(attempts=3, backoff_base=0.0),
    )
    assert posted.status == 200
    assert first.closed is True
    assert session.calls == 2


async def test_aiohttp_nonretryable_status_body_read_failure_closes_then_raises(monkeypatch):
    first = _AiohttpResponse(400, "", text_error=ConnectionError("truncated"))
    session = _AiohttpSession([first])
    with pytest.raises(ConnectionError, match="truncated"):
        await retry_aiohttp_post(
            session,
            "https://example.test/v1",
            json={"model": "x"},
            policy=RetryPolicy(attempts=3, backoff_base=0.0),
        )
    assert first.closed is True
    assert session.calls == 1


def test_urllib_http_error_closed_before_retry():
    closes: list[int] = []
    attempts = {"n": 0}

    class _CountedError(HTTPError):
        def close(self):
            closes.append(1)
            super().close()

    def fake_urlopen(request, timeout=20.0):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _CountedError(
                "https://example.test/models",
                502,
                "Bad Gateway",
                hdrs=None,
                fp=BytesIO(b"nope"),
            )
        return _Resp()

    result = request_urllib(
        "https://example.test/models",
        policy=RetryPolicy(attempts=3, backoff_base=0.0),
        urlopen_fn=fake_urlopen,
        sleep_fn=lambda _delay: None,
    )
    assert result.status == 200
    assert closes == [1]


def test_urllib_budget_exhaustion_preserves_final_error_body():
    def fake_urlopen(request, timeout=20.0):
        raise HTTPError(
            "https://example.test/go",
            429,
            "Too Many",
            hdrs=None,
            fp=BytesIO(b'{"error":"slow"}'),
        )

    result = request_urllib(
        "https://example.test/go",
        policy=RetryPolicy(attempts=1, wait_budget=30.0),
        raise_on_http_error=False,
        urlopen_fn=fake_urlopen,
        sleep_fn=lambda _delay: None,
    )
    assert result.status == 429
    assert result.body == b'{"error":"slow"}'


async def test_tiny_retry_after_cannot_create_unbounded_extension_attempts(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("codex_shim.net.retry.asyncio.sleep", fake_sleep)
    session = _AiohttpSession(
        [_AiohttpResponse(429, "slow", headers={"Retry-After": "0.001"})] * 40
    )
    posted = await retry_aiohttp_post(
        session,
        "https://example.test/v1",
        json={"model": "x"},
        policy=RetryPolicy(attempts=3, backoff_base=0.0, wait_budget=30.0),
    )
    assert posted.status == 429
    assert session.calls <= 12
    assert session.calls >= 3


def test_one_attempt_policy_never_uses_wait_budget_extension():
    attempts = {"n": 0}

    def fake_urlopen(request, timeout=20.0):
        attempts["n"] += 1
        raise HTTPError(
            "https://example.test/oauth",
            429,
            "Too Many",
            hdrs={"Retry-After": "5"},
            fp=BytesIO(b"nope"),
        )

    with pytest.raises(HTTPError):
        request_urllib(
            "https://example.test/oauth",
            policy=RetryPolicy(attempts=1, wait_budget=30.0),
            urlopen_fn=fake_urlopen,
            sleep_fn=lambda _delay: None,
        )
    assert attempts["n"] == 1


def test_parse_retry_after_accepts_http_date_and_caps_delay():
    from datetime import datetime, timedelta, timezone
    from email.utils import format_datetime

    from codex_shim.net.retry import parse_retry_after

    future = datetime.now(timezone.utc) + timedelta(seconds=12)
    parsed = parse_retry_after(type("R", (), {"headers": {"Retry-After": format_datetime(future)}})())
    assert parsed is not None
    assert 10.0 <= parsed <= 13.0

    far = datetime.now(timezone.utc) + timedelta(seconds=120)
    capped = parse_retry_after(type("R", (), {"headers": {"Retry-After": format_datetime(far)}})())
    assert capped == 60.0


def test_aiohttp_and_urllib_share_retry_decision_sequence():
    from codex_shim.net.retry import _RetryState

    policy = RetryPolicy(attempts=3, backoff_base=0.5, backoff_factor=2.0, wait_budget=30.0)
    events = [(True, 5.0)] * 12

    def waits_for() -> list[float | None]:
        state = _RetryState()
        out: list[float | None] = []
        for retryable, retry_after in events:
            wait = state.next_wait(policy, retryable=retryable, retry_after=retry_after)
            out.append(wait)
            if wait is None:
                break
            state.record(wait, extension=state.in_extension(policy))
        return out

        first = waits_for()
        second = waits_for()
        assert first == second
        assert first[:2] == [5.0, 5.0]
        assert first[-1] is None
        assert 3 <= len(first) <= 11
        assert all(wait is None or wait >= 5.0 for wait in first)
