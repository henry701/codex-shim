from __future__ import annotations

from io import BytesIO
import json
from urllib.error import HTTPError, URLError

import pytest

from codex_shim.net.errors import (
    chat_chunk_upstream_error,
    is_retryable_exception,
    parse_upstream_error,
)
from codex_shim.net.retry import (
    RetryPolicy,
    configure_origin_backoff,
    get_origin_backoff,
    request_urllib,
    reset_origin_backoff,
    retry_aiohttp_post,
    retry_aiohttp_ws_connect,
    retry_policy_from_env,
)


@pytest.fixture(autouse=True)
def _reset_origin_backoff_between_tests():
    reset_origin_backoff()
    yield
    reset_origin_backoff()


def _rate_limit_policy(**overrides):
    policy = RetryPolicy(
        attempts=3,
        backoff_base=0.5,
        backoff_factor=2.0,
        rate_limit_min=60.0,
        rate_limit_max=3600.0,
        rate_limit_jitter=0.0,
    )
    for key, value in overrides.items():
        setattr(policy, key, value)
    return policy


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
    monkeypatch.setenv("CODEX_SHIM_RETRY_RATE_LIMIT_MIN", "90")
    monkeypatch.setenv("CODEX_SHIM_RETRY_RATE_LIMIT_MAX", "1800")
    policy = retry_policy_from_env()
    assert policy.attempts == 5
    assert policy.backoff_base == 0.25
    assert policy.backoff_factor == 3.0
    assert policy.delay_for(0) == 0.25
    assert policy.delay_for(1) == 0.75
    assert policy.rate_limit_min == 90.0
    assert policy.rate_limit_max == 1800.0


def test_parse_upstream_error_and_retryable_exception():
    code, message = parse_upstream_error('{"error":{"message":"nope","type":"overflow"}}', 400)
    assert code == "overflow"
    assert message == "nope"
    assert is_retryable_exception(TimeoutError("late")) is True
    assert is_retryable_exception(ValueError("nope")) is False


def test_chat_chunk_upstream_error_detects_native_network_error():
    chunk = {
        "choices": [
            {
                "delta": {"content": "", "role": "assistant"},
                "finish_reason": "stop",
                "native_finish_reason": "network_error",
                "message": {"role": "assistant", "content": None},
            }
        ]
    }
    code, message = chat_chunk_upstream_error(chunk)
    assert code == "network_error"
    assert "network_error" in message


def test_chat_chunk_upstream_error_detects_embedded_error_object():
    chunk = {"error": {"message": "The requested model is temporarily at capacity upstream.", "code": 429}}
    code, message = chat_chunk_upstream_error(chunk)
    assert "429" in code or code == "upstream_http_502"
    assert "capacity" in message


def test_chat_chunk_upstream_error_ignores_normal_stop():
    chunk = {"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}
    assert chat_chunk_upstream_error(chunk) is None


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
        policy=_rate_limit_policy(),
        label="or-test",
    )
    assert posted.status == 200
    assert session.calls == 2
    assert sleeps == pytest.approx([60.0], abs=0.05)


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
        policy=_rate_limit_policy(),
    )
    assert posted.status == 200
    assert sleeps == pytest.approx([60.0], abs=0.05)


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
        policy=_rate_limit_policy(wait_budget=20.0),
    )
    assert posted.status == 200
    assert session.calls == 5
    assert sleeps == pytest.approx([60.0, 120.0, 240.0, 480.0], abs=0.05)


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
        policy=_rate_limit_policy(),
    )
    assert posted.status == 200
    assert sleeps == pytest.approx([60.0], abs=0.05)


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
        [_AiohttpResponse(503, "slow", headers={"Retry-After": "0.001"})] * 40
    )
    posted = await retry_aiohttp_post(
        session,
        "https://example.test/v1",
        json={"model": "x"},
        policy=RetryPolicy(attempts=3, backoff_base=0.0, wait_budget=30.0),
    )
    assert posted.status == 503
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
    parsed_far = parse_retry_after(type("R", (), {"headers": {"Retry-After": format_datetime(far)}})())
    assert parsed_far is not None
    assert 110.0 <= parsed_far <= 130.0

    huge = datetime.now(timezone.utc) + timedelta(seconds=7200)
    capped = parse_retry_after(type("R", (), {"headers": {"Retry-After": format_datetime(huge)}})())
    assert capped == 3600.0


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


_FREE_USAGE_BODY = (
    '{"type":"error","error":{"type":"FreeUsageLimitError","message":"Rate limit exceeded"}}'
)


def _quota_body(*, resets_in_seconds: int) -> str:
    return json.dumps(
        {
            "error": {
                "type": "usage_limit_reached",
                "message": "You've hit your usage limit",
                "plan_type": "plus",
                "resets_in_seconds": resets_in_seconds,
            }
        }
    )


async def test_retry_aiohttp_post_rate_limit_ramps_then_caps_endlessly(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("codex_shim.net.retry.asyncio.sleep", fake_sleep)
    session = _AiohttpSession(
        [_AiohttpResponse(429, _FREE_USAGE_BODY, content_type="application/json")] * 8
        + [_AiohttpResponse(200, "", "text/event-stream")]
    )
    posted = await retry_aiohttp_post(
        session,
        "https://opencode.ai/zen/v1/responses",
        json={"model": "x"},
        policy=_rate_limit_policy(),
    )
    assert posted.status == 200
    assert session.calls == 9
    assert sleeps[:2] == pytest.approx([60.0, 120.0], abs=0.05)
    assert sleeps[-2:] == pytest.approx([3600.0, 3600.0], abs=0.05)
    assert max(sleeps) == pytest.approx(3600.0, abs=0.05)


async def test_retry_aiohttp_post_retry_after_7200_clamps_to_max(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("codex_shim.net.retry.asyncio.sleep", fake_sleep)
    session = _AiohttpSession(
        [
            _AiohttpResponse(429, "slow", headers={"Retry-After": "7200"}),
            _AiohttpResponse(200, "", "text/event-stream"),
        ]
    )
    posted = await retry_aiohttp_post(
        session,
        "https://opencode.ai/zen/v1/responses",
        json={"model": "x"},
        policy=_rate_limit_policy(),
    )
    assert posted.status == 200
    assert sleeps == pytest.approx([3600.0], abs=0.05)


async def test_retry_aiohttp_post_http_origin_gate_blocks_second_post(monkeypatch):
    sleeps: list[float] = []
    post_order: list[str] = []
    a_waiting = __import__("asyncio").Event()
    b_finished = __import__("asyncio").Event()

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        task = __import__("asyncio").current_task()
        name = task.get_name() if task is not None else ""
        if name == "gate-a" and not a_waiting.is_set():
            a_waiting.set()
            await b_finished.wait()

    monkeypatch.setattr("codex_shim.net.retry.asyncio.sleep", fake_sleep)

    class _SharedSession:
        def __init__(self) -> None:
            self.calls = 0

        async def post(self, url, json=None, headers=None):
            task = __import__("asyncio").current_task()
            post_order.append(task.get_name() if task is not None else "")
            self.calls += 1
            if self.calls == 1:
                return _AiohttpResponse(429, _FREE_USAGE_BODY, content_type="application/json")
            return _AiohttpResponse(200, "", "text/event-stream")

    session = _SharedSession()
    url = "https://opencode.ai/zen/v1/responses"

    async def run_a():
        return await retry_aiohttp_post(
            session, url, json={"model": "x"}, policy=_rate_limit_policy()
        )

    async def run_b():
        await a_waiting.wait()
        posted = await retry_aiohttp_post(
            session, url, json={"model": "x"}, policy=_rate_limit_policy()
        )
        b_finished.set()
        return posted

    import asyncio

    posted_a, posted_b = await asyncio.gather(
        asyncio.create_task(run_a(), name="gate-a"),
        asyncio.create_task(run_b(), name="gate-b"),
    )
    assert posted_a.status == 200
    assert posted_b.status == 200
    assert "gate-b" in sleeps or any(name == "gate-b" for name in post_order)
    assert sleeps[0] == pytest.approx(60.0, abs=0.05)
    assert any(wait >= 59.0 for wait in sleeps[1:])
    assert post_order[0] == "gate-a"
    assert session.calls >= 3


async def test_retry_aiohttp_post_origin_gate_does_not_span_hosts(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("codex_shim.net.retry.asyncio.sleep", fake_sleep)
    configure_origin_backoff("https://opencode.ai/zen/v1/responses", delay=90.0)
    session = _AiohttpSession([_AiohttpResponse(200, "", "text/event-stream")])
    posted = await retry_aiohttp_post(
        session,
        "https://chatgpt.com/backend-api/codex/responses",
        json={"model": "x"},
        policy=_rate_limit_policy(),
    )
    assert posted.status == 200
    assert session.calls == 1
    assert sleeps == []


async def test_retry_aiohttp_post_seeded_origin_gate_sleeps_before_send(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("codex_shim.net.retry.asyncio.sleep", fake_sleep)
    configure_origin_backoff("https://opencode.ai/zen/v1/responses", delay=90.0)
    session = _AiohttpSession([_AiohttpResponse(200, "", "text/event-stream")])
    posted = await retry_aiohttp_post(
        session,
        "https://opencode.ai/zen/v1/chat/completions",
        json={"model": "x"},
        policy=_rate_limit_policy(),
    )
    assert posted.status == 200
    assert session.calls == 1
    assert sleeps == pytest.approx([90.0], abs=0.05)


async def test_retry_aiohttp_post_quota_waits_max_endlessly(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("codex_shim.net.retry.asyncio.sleep", fake_sleep)
    session = _AiohttpSession(
        [_AiohttpResponse(429, _quota_body(resets_in_seconds=600000), content_type="application/json")] * 2
        + [_AiohttpResponse(200, "", "text/event-stream")]
    )
    posted = await retry_aiohttp_post(
        session,
        "https://chatgpt.com/backend-api/codex/responses",
        json={"model": "x"},
        policy=_rate_limit_policy(attempts=3),
    )
    assert posted.status == 200
    assert session.calls == 3
    assert sleeps == pytest.approx([3600.0, 3600.0], abs=0.05)


async def test_retry_aiohttp_post_quota_honors_short_reset(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("codex_shim.net.retry.asyncio.sleep", fake_sleep)
    session = _AiohttpSession(
        [
            _AiohttpResponse(429, _quota_body(resets_in_seconds=90), content_type="application/json"),
            _AiohttpResponse(200, "", "text/event-stream"),
        ]
    )
    posted = await retry_aiohttp_post(
        session,
        "https://chatgpt.com/backend-api/codex/responses",
        json={"model": "x"},
        policy=_rate_limit_policy(),
    )
    assert posted.status == 200
    assert sleeps == pytest.approx([90.0], abs=0.05)


async def test_retry_aiohttp_post_gives_up_when_client_disconnects(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("codex_shim.net.retry.asyncio.sleep", fake_sleep)
    configure_origin_backoff("https://opencode.ai/zen/v1/responses", delay=60.0)
    session = _AiohttpSession([_AiohttpResponse(200, "", "text/event-stream")])
    from codex_shim.net.sse import ClientDisconnected

    with pytest.raises(ClientDisconnected):
        await retry_aiohttp_post(
            session,
            "https://opencode.ai/zen/v1/responses",
            json={"model": "x"},
            policy=_rate_limit_policy(),
            disconnect_fn=lambda: True,
        )
    assert session.calls == 0
    assert sleeps == []


async def test_retry_aiohttp_post_emits_pings_during_rate_limit_wait(monkeypatch):
    sleeps: list[float] = []
    pings: list[int] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    async def ping_fn() -> None:
        pings.append(1)

    monkeypatch.setattr("codex_shim.net.retry.asyncio.sleep", fake_sleep)
    session = _AiohttpSession(
        [
            _AiohttpResponse(429, _FREE_USAGE_BODY, content_type="application/json"),
            _AiohttpResponse(200, "", "text/event-stream"),
        ]
    )
    posted = await retry_aiohttp_post(
        session,
        "https://opencode.ai/zen/v1/responses",
        json={"model": "x"},
        policy=_rate_limit_policy(),
        ping_fn=ping_fn,
        keepalive=15.0,
    )
    assert posted.status == 200
    assert sum(sleeps) == pytest.approx(60.0, abs=0.05)
    assert pings


class _WsHandshakeError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"handshake {status}")
        self.status = status


class _WsSession:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    async def ws_connect(self, url, headers=None, heartbeat=None):
        self.calls += 1
        item = self.outcomes.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


async def test_retry_aiohttp_ws_connect_sleeps_per_connection_not_origin_gate(monkeypatch):
    import asyncio

    sleeps: list[tuple[str, float]] = []
    a_waiting = asyncio.Event()
    b_connected = asyncio.Event()

    async def fake_sleep(seconds: float) -> None:
        task = asyncio.current_task()
        name = task.get_name() if task is not None else ""
        sleeps.append((name, seconds))
        if name == "ws-a":
            a_waiting.set()
            await b_connected.wait()

    monkeypatch.setattr("codex_shim.net.retry.asyncio.sleep", fake_sleep)
    session_a = _WsSession([_WsHandshakeError(429), object()])
    session_b = _WsSession([object()])

    async def run_a():
        return await retry_aiohttp_ws_connect(
            session_a,
            "wss://chatgpt.com/backend-api/codex/responses",
            policy=_rate_limit_policy(),
            ws_session="a",
        )

    async def run_b():
        await a_waiting.wait()
        result = await retry_aiohttp_ws_connect(
            session_b,
            "wss://chatgpt.com/backend-api/codex/responses",
            policy=_rate_limit_policy(),
            ws_session="b",
        )
        b_connected.set()
        return result

    ws_a, ws_b = await asyncio.gather(
        asyncio.create_task(run_a(), name="ws-a"),
        asyncio.create_task(run_b(), name="ws-b"),
    )
    assert ws_a is not None
    assert ws_b is not None
    assert session_b.calls == 1
    assert session_a.calls == 2
    assert sleeps[0][0] == "ws-a"
    assert sleeps[0][1] == pytest.approx(60.0, abs=0.05)
    assert all(name != "ws-b" for name, _delay in sleeps)
    origin = get_origin_backoff("wss://chatgpt.com/backend-api/codex/responses")
    assert origin is not None
    assert origin.kind == "none" or origin.exponent in {1, 2}


def test_request_urllib_rate_limit_ramps_then_succeeds():
    sleeps: list[float] = []
    attempts = {"n": 0}

    def fake_urlopen(request, timeout=20.0):
        attempts["n"] += 1
        if attempts["n"] < 4:
            raise HTTPError(
                "https://opencode.ai/zen/v1/models",
                429,
                "Too Many",
                hdrs=None,
                fp=BytesIO(_FREE_USAGE_BODY.encode()),
            )
        return _Resp()

    result = request_urllib(
        "https://opencode.ai/zen/v1/models",
        policy=_rate_limit_policy(),
        urlopen_fn=fake_urlopen,
        sleep_fn=sleeps.append,
        ensure_json=True,
    )
    assert result.status == 200
    assert attempts["n"] == 4
    assert sleeps == pytest.approx([60.0, 120.0, 240.0], abs=0.05)


def test_request_urllib_seeded_origin_gate_sleeps_before_send():
    sleeps: list[float] = []
    configure_origin_backoff("https://opencode.ai/zen/v1/models", delay=90.0)

    def fake_urlopen(request, timeout=20.0):
        return _Resp()

    result = request_urllib(
        "https://opencode.ai/zen/v1/chat/completions",
        policy=_rate_limit_policy(),
        urlopen_fn=fake_urlopen,
        sleep_fn=sleeps.append,
        ensure_json=True,
    )
    assert result.status == 200
    assert sleeps == pytest.approx([90.0], abs=0.05)


def test_request_urllib_quota_waits_max_then_succeeds():
    sleeps: list[float] = []
    attempts = {"n": 0}

    def fake_urlopen(request, timeout=20.0):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise HTTPError(
                "https://chatgpt.com/backend-api/codex/responses",
                429,
                "Too Many",
                hdrs=None,
                fp=BytesIO(_quota_body(resets_in_seconds=600000).encode()),
            )
        return _Resp()

    result = request_urllib(
        "https://chatgpt.com/backend-api/codex/responses",
        policy=_rate_limit_policy(),
        urlopen_fn=fake_urlopen,
        sleep_fn=sleeps.append,
        ensure_json=True,
    )
    assert result.status == 200
    assert attempts["n"] == 3
    assert sleeps == pytest.approx([3600.0, 3600.0], abs=0.05)


def test_request_urllib_503_does_not_seed_origin_gate():
    sleeps: list[float] = []
    attempts = {"n": 0}

    def fake_urlopen(request, timeout=20.0):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise HTTPError(
                "https://opencode.ai/zen/v1/models",
                503,
                "Unavailable",
                hdrs=None,
                fp=BytesIO(b"nope"),
            )
        return _Resp()

    result = request_urllib(
        "https://opencode.ai/zen/v1/models",
        policy=_rate_limit_policy(backoff_base=0.5),
        urlopen_fn=fake_urlopen,
        sleep_fn=sleeps.append,
        ensure_json=True,
    )
    assert result.status == 200
    assert sleeps == pytest.approx([0.5], abs=0.05)
    origin = get_origin_backoff("https://opencode.ai/zen/v1/models")
    assert origin is None or origin.kind == "none"
    assert origin is None or origin.not_before == 0


async def test_retry_aiohttp_post_503_does_not_seed_origin_gate(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("codex_shim.net.retry.asyncio.sleep", fake_sleep)
    session = _AiohttpSession(
        [
            _AiohttpResponse(503, "nope"),
            _AiohttpResponse(200, "", "text/event-stream"),
        ]
    )
    posted = await retry_aiohttp_post(
        session,
        "https://opencode.ai/zen/v1/responses",
        json={"model": "x"},
        policy=_rate_limit_policy(),
    )
    assert posted.status == 200
    assert sleeps == pytest.approx([0.5], abs=0.05)
    origin = get_origin_backoff("https://opencode.ai/zen/v1/responses")
    assert origin is None or origin.kind == "none"


async def test_retry_aiohttp_ws_connect_pings_during_rate_limit_wait(monkeypatch):
    sleeps: list[float] = []
    pings: list[int] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    async def ping_fn() -> None:
        pings.append(1)

    monkeypatch.setattr("codex_shim.net.retry.asyncio.sleep", fake_sleep)
    session = _WsSession([_WsHandshakeError(429), object()])
    ws = await retry_aiohttp_ws_connect(
        session,
        "wss://opencode.ai/zen",
        policy=_rate_limit_policy(),
        ping_fn=ping_fn,
        keepalive=15.0,
    )
    assert ws is not None
    assert session.calls == 2
    assert sum(sleeps) == pytest.approx(60.0, abs=0.05)
    assert pings


async def test_retry_aiohttp_ws_connect_gives_up_when_client_disconnects(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("codex_shim.net.retry.asyncio.sleep", fake_sleep)
    session = _WsSession([_WsHandshakeError(429), object()])
    from codex_shim.net.sse import ClientDisconnected

    def disconnected() -> bool:
        return True

    with pytest.raises(ClientDisconnected):
        await retry_aiohttp_ws_connect(
            session,
            "wss://opencode.ai/zen",
            policy=_rate_limit_policy(),
            disconnect_fn=disconnected,
        )
    assert session.calls == 0
    assert sleeps == []


async def test_await_ws_throttle_sleeps_on_429_connect_error(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("codex_shim.net.retry.asyncio.sleep", fake_sleep)
    monkeypatch.setenv("CODEX_SHIM_RETRY_RATE_LIMIT_JITTER", "0")
    from codex_shim.ws_passthrough import WsPassthroughConnectError, await_ws_throttle

    waited = await await_ws_throttle(
        object(),
        "wss://opencode.ai/zen",
        WsPassthroughConnectError("FreeUsageLimitError", status=429),
    )
    assert waited is True
    assert sleeps == pytest.approx([60.0], abs=0.05)


async def test_await_ws_throttle_ignores_non_throttle_connect_error(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("codex_shim.net.retry.asyncio.sleep", fake_sleep)
    from codex_shim.ws_passthrough import WsPassthroughConnectError, await_ws_throttle

    class _Session:
        def client_disconnected(self) -> bool:
            return False

    waited = await await_ws_throttle(
        _Session(),
        "wss://opencode.ai/zen",
        WsPassthroughConnectError("bad request", status=400),
    )
    assert waited is False
    assert sleeps == []
