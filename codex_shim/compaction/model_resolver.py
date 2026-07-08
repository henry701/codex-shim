from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .config import CompactionSettings

TertiarySkipReason = str  # not_configured | route_error | no_credentials


@dataclass(frozen=True)
class ResolvedCompactionModels:
    native_slug: str
    summarization_slug: str
    tertiary_slug: str | None
    tertiary_configured_slug: str | None = None
    tertiary_skip_reason: TertiarySkipReason | None = None


class CompactionModelResolver:
    def __init__(
        self,
        settings: CompactionSettings,
        *,
        route_fn: Callable[[dict[str, Any]], Any | Awaitable[Any]] | None = None,
        has_credentials_fn: Callable[[Any], bool] | None = None,
    ) -> None:
        self._settings = settings
        self._route_fn = route_fn
        self._has_credentials_fn = has_credentials_fn

    async def resolve(
        self,
        *,
        requested_slug: str,
        body: dict[str, Any],
        passthrough_fallback_slug: str | None = None,
    ) -> ResolvedCompactionModels:
        native_slug = requested_slug
        summarization_slug = requested_slug
        configured = self._settings.model
        if configured and (self._settings.override_current_model or configured != requested_slug):
            summarization_slug = configured
        configured_tertiary = self._settings.tertiary_fallback_slug or passthrough_fallback_slug
        tertiary = configured_tertiary
        skip_reason: TertiarySkipReason | None = None
        if not configured_tertiary:
            tertiary = None
            skip_reason = "not_configured"
        elif not self._route_fn or not self._has_credentials_fn:
            tertiary = configured_tertiary
            skip_reason = None
        else:
            try:
                route_result = self._route_fn({**body, "model": configured_tertiary})
                if inspect.isawaitable(route_result):
                    route = await route_result
                else:
                    route = route_result
                if not self._has_credentials_fn(route):
                    tertiary = None
                    skip_reason = "no_credentials"
            except Exception:
                tertiary = None
                skip_reason = "route_error"
        return ResolvedCompactionModels(
            native_slug=native_slug,
            summarization_slug=summarization_slug,
            tertiary_slug=tertiary,
            tertiary_configured_slug=configured_tertiary,
            tertiary_skip_reason=skip_reason,
        )
