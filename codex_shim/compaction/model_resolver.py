from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .config import CompactionSettings


@dataclass(frozen=True)
class ResolvedCompactionModels:
    native_slug: str
    summarization_slug: str
    tertiary_slug: str | None


class CompactionModelResolver:
    def __init__(
        self,
        settings: CompactionSettings,
        *,
        route_fn: Callable[[dict[str, Any]], Any] | None = None,
        has_credentials_fn: Callable[[Any], bool] | None = None,
    ) -> None:
        self._settings = settings
        self._route_fn = route_fn
        self._has_credentials_fn = has_credentials_fn

    def resolve(
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
        tertiary = self._settings.tertiary_fallback_slug or passthrough_fallback_slug
        if tertiary and self._route_fn and self._has_credentials_fn:
            try:
                route = self._route_fn({**body, "model": tertiary})
                if not self._has_credentials_fn(route):
                    tertiary = None
            except Exception:
                tertiary = None
        return ResolvedCompactionModels(
            native_slug=native_slug,
            summarization_slug=summarization_slug,
            tertiary_slug=tertiary,
        )
