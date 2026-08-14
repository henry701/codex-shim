from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from multidict import CIMultiDict

from codex_shim.server import ShimServer
from codex_shim.ws_passthrough import WsPassthroughConnectError


@pytest.mark.asyncio
async def test_chatgpt_ws_relay_reset_falls_back_to_http(tmp_path, monkeypatch):
    settings = tmp_path / "settings.json"
    settings.write_text('{"models": []}')
    shim = ShimServer(settings)
    http_fallback = AsyncMock()
    monkeypatch.setattr(shim, "_handle_chatgpt_response_create_websocket_http", http_fallback)
    monkeypatch.setattr(
        shim,
        "_prepare_chatgpt_passthrough_body_ws",
        lambda *args, **kwargs: {"model": "gpt-5.5", "input": [], "stream": True},
    )

    passthrough = MagicMock()
    passthrough.client_ws = AsyncMock()
    passthrough.client_session = AsyncMock()
    passthrough.connect_upstream = AsyncMock(return_value=({}, False))
    passthrough.last_upstream_chained_response_id = MagicMock(return_value=None)
    passthrough.send_response_create = AsyncMock()
    passthrough.relay_until_terminal = AsyncMock(
        side_effect=WsPassthroughConnectError("Cannot write to closing transport")
    )
    passthrough.note_thread_id = MagicMock()

    request = MagicMock()
    request.headers = CIMultiDict()
    payload = {"type": "response.create", "model": "codex-gpt-5-5", "input": []}
    target = ShimServer._WsPassthroughTarget(
        kind="chatgpt",
        requested_slug="codex-gpt-5-5",
        upstream_model="gpt-5.5",
        response_model_override="codex-gpt-5-5",
        access_token="tok",
        account_id="acc",
    )

    handled = await shim._handle_chatgpt_ws_passthrough_response_create(
        request,
        passthrough,
        payload,
        target,
    )

    assert handled is True
    http_fallback.assert_awaited_once()
    passthrough.send_response_create.assert_awaited_once()
    passthrough.relay_until_terminal.assert_awaited_once()
