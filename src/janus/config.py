"""Validated configuration loading for Janus."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.json"


class CacheSettings(BaseModel):
    """Cache behavior and policy settings."""

    model_config = ConfigDict(extra="forbid")

    exact_enabled: bool = True
    semantic_enabled: bool = True
    ttl_seconds: Annotated[int, Field(ge=60, le=604800)] = 3600
    max_semantic_distance: Annotated[float, Field(ge=0.0, le=0.3)] = 0.08


class PerformanceSettings(BaseModel):
    """Resource limits for embedding and streaming operations."""

    model_config = ConfigDict(extra="forbid")

    max_concurrent_embeds: Annotated[int, Field(ge=1)] = 2
    max_stream_buffer_bytes: Annotated[int, Field(ge=1)] = 1_000_000


class AppSettings(BaseModel):
    """Application behavior loaded from settings.json."""

    model_config = ConfigDict(extra="forbid")

    cache: CacheSettings = Field(default_factory=CacheSettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)


class EnvironmentSettings(BaseSettings):
    """Infrastructure secrets and optional environment overrides."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "NVIDIA_API_KEY"),
    )
    base_url: str | None = None
    redis_url: str = "redis://localhost:6379"
    proxy_keys: str = Field(
        default="",
        validation_alias=AliasChoices("PROXY_KEYS", "PROXY_API_KEYS"),
    )
    config_path: Path | None = None

    cache_exact_enabled: bool | None = Field(
        default=None,
        validation_alias=AliasChoices("CACHE_EXACT_ENABLED", "EXACT_CACHE_ENABLED"),
    )
    cache_semantic_enabled: bool | None = Field(
        default=None,
        validation_alias=AliasChoices("CACHE_SEMANTIC_ENABLED", "SEMANTIC_CACHE_ENABLED"),
    )
    cache_ttl_seconds: int | None = None
    max_semantic_distance: float | None = Field(
        default=None,
        validation_alias=AliasChoices("MAX_SEMANTIC_DISTANCE", "SEMANTIC_MAX_DISTANCE"),
    )
    max_concurrent_embeds: int | None = None
    max_stream_buffer_bytes: int | None = Field(
        default=None,
        validation_alias=AliasChoices("MAX_STREAM_BUFFER_BYTES", "MAX_STREAM_BUFFER"),
    )

    @field_validator("openai_api_key")
    @classmethod
    def validate_api_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and not value.get_secret_value().strip():
            raise ValueError("OPENAI_API_KEY must not be empty")
        return value

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip().rstrip("/")
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("BASE_URL must be an absolute http:// or https:// URL")
        return value + "/"

    @field_validator("redis_url")
    @classmethod
    def validate_redis_url(cls, value: str) -> str:
        value = value.strip()
        parsed = urlsplit(value)
        if parsed.scheme not in {"redis", "rediss", "unix"}:
            raise ValueError("REDIS_URL must use the redis://, rediss://, or unix:// scheme")
        if parsed.scheme == "unix":
            if not parsed.path:
                raise ValueError("REDIS_URL must include a Unix socket path")
        elif not parsed.netloc:
            raise ValueError("REDIS_URL must include a Redis host")
        return value

    def require_runtime_values(self) -> None:
        """Raise a readable error when required deployment settings are absent."""
        missing = []
        if self.openai_api_key is None:
            missing.append("OPENAI_API_KEY")
        if self.base_url is None:
            missing.append("BASE_URL")
        if not self.proxy_key_values:
            missing.append("PROXY_KEYS")
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    @property
    def upstream_api_key(self) -> str:
        """Return the configured upstream key after runtime validation."""
        if self.openai_api_key is None:
            raise RuntimeError("OPENAI_API_KEY is required to initialize the proxy")
        return self.openai_api_key.get_secret_value()

    @property
    def upstream_base_url(self) -> str:
        """Return the configured upstream URL after runtime validation."""
        if self.base_url is None:
            raise RuntimeError("BASE_URL is required to initialize the proxy")
        return self.base_url

    @property
    def proxy_key_values(self) -> tuple[str, ...]:
        """Return the comma-separated proxy keys as a normalized tuple."""
        return tuple(key.strip() for key in self.proxy_keys.split(",") if key.strip())


class ProxyConfiguration(BaseModel):
    """The complete, process-wide proxy configuration."""

    environment: EnvironmentSettings
    app: AppSettings


def _environment_overrides(environment: EnvironmentSettings) -> dict[str, dict[str, Any]]:
    cache: dict[str, Any] = {}
    performance: dict[str, Any] = {}
    for name in ("exact_enabled", "semantic_enabled", "ttl_seconds"):
        value = getattr(environment, f"cache_{name}")
        if value is not None:
            cache[name] = value
    if environment.max_semantic_distance is not None:
        cache["max_semantic_distance"] = environment.max_semantic_distance
    if environment.max_concurrent_embeds is not None:
        performance["max_concurrent_embeds"] = environment.max_concurrent_embeds
    if environment.max_stream_buffer_bytes is not None:
        performance["max_stream_buffer_bytes"] = environment.max_stream_buffer_bytes
    return {"cache": cache, "performance": performance}


def _merge_settings(
    file_settings: Mapping[str, Any],
    environment_settings: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(file_settings)
    for section, values in environment_settings.items():
        if values:
            current = merged.get(section, {})
            if not isinstance(current, dict):
                raise ValueError(f"Configuration section '{section}' must be a JSON object")
            merged[section] = {**current, **values}
    return merged


def load_app_settings(
    config_path: str | Path | None = None,
    *,
    environment: EnvironmentSettings | None = None,
) -> AppSettings:
    """Load, merge, and validate JSON settings and optional environment overrides."""
    env = environment or EnvironmentSettings()
    path = Path(config_path) if config_path is not None else env.config_path
    path = path or DEFAULT_SETTINGS_PATH

    file_settings: dict[str, Any] = {}
    try:
        with path.open(encoding="utf-8") as settings_file:
            loaded = json.load(settings_file)
    except FileNotFoundError:
        loaded = {}
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Malformed JSON in configuration file {path} "
            f"(line {exc.lineno}, column {exc.colno}): {exc.msg}"
        ) from exc
    except OSError as exc:
        raise ValueError(f"Could not read configuration file {path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration file {path} must contain a JSON object")
    file_settings = loaded

    overrides = _environment_overrides(env)
    if not any(overrides.values()):
        overrides = {}
    try:
        return AppSettings.model_validate(_merge_settings(file_settings, overrides))
    except ValidationError as exc:
        raise ValueError(f"Invalid settings in configuration file {path} or environment: {exc}") from exc


def load_configuration(
    environment: EnvironmentSettings | None = None,
) -> ProxyConfiguration:
    """Load and validate all configuration once during application startup."""
    environment = environment or EnvironmentSettings()
    environment.require_runtime_values()
    app_settings = load_app_settings(environment=environment)
    return ProxyConfiguration(environment=environment, app=app_settings)
