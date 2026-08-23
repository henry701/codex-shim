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
    def __init__(self, status: int, text: str, content_type: str = "text/plain") -> None:
        self.status = status
        self.content_type = content_type
        self._text = text
        self.closed = False

    async def text(self):
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
