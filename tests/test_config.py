import json

import pytest
from pydantic import SecretStr
from pydantic_settings import SettingsConfigDict

from janus.config import EnvironmentSettings, load_app_settings, load_configuration


class IsolatedEnvironmentSettings(EnvironmentSettings):
    model_config = SettingsConfigDict(
        env_file=None,
        extra="ignore",
        populate_by_name=True,
    )


ENVIRONMENT_VARIABLES = (
    "OPENAI_API_KEY",
    "NVIDIA_API_KEY",
    "BASE_URL",
    "REDIS_URL",
    "PROXY_KEYS",
    "PROXY_API_KEYS",
    "CONFIG_PATH",
    "CACHE_EXACT_ENABLED",
    "EXACT_CACHE_ENABLED",
    "CACHE_SEMANTIC_ENABLED",
    "SEMANTIC_CACHE_ENABLED",
    "CACHE_TTL_SECONDS",
    "MAX_SEMANTIC_DISTANCE",
    "SEMANTIC_MAX_DISTANCE",
    "MAX_CONCURRENT_EMBEDS",
    "MAX_STREAM_BUFFER_BYTES",
    "MAX_STREAM_BUFFER",
)


@pytest.fixture(autouse=True)
def clear_configuration_environment(monkeypatch):
    for name in ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def test_missing_file_uses_builtin_defaults(tmp_path):
    settings = load_app_settings(
        tmp_path / "missing.json",
        environment=IsolatedEnvironmentSettings(),
    )

    assert settings.cache.exact_enabled is True
    assert settings.cache.semantic_enabled is True
    assert settings.cache.ttl_seconds == 3600
    assert settings.cache.max_semantic_distance == 0.08
    assert settings.performance.max_concurrent_embeds == 2
    assert settings.performance.max_stream_buffer_bytes == 1_000_000


def test_loads_settings_file_and_environment_overrides(tmp_path):
    config_path = tmp_path / "settings.json"
    config_path.write_text(
        json.dumps(
            {
                "cache": {
                    "exact_enabled": False,
                    "semantic_enabled": False,
                    "ttl_seconds": 1800,
                    "max_semantic_distance": 0.12,
                },
                "performance": {
                    "max_concurrent_embeds": 4,
                    "max_stream_buffer_bytes": 2048,
                },
            }
        ),
        encoding="utf-8",
    )
    environment = IsolatedEnvironmentSettings(
        cache_exact_enabled=True,
        cache_ttl_seconds=120,
        max_concurrent_embeds=3,
    )

    settings = load_app_settings(config_path, environment=environment)

    assert settings.cache.exact_enabled is True
    assert settings.cache.semantic_enabled is False
    assert settings.cache.ttl_seconds == 120
    assert settings.cache.max_semantic_distance == 0.12
    assert settings.performance.max_concurrent_embeds == 3
    assert settings.performance.max_stream_buffer_bytes == 2048


def test_config_path_and_infrastructure_environment_variables(monkeypatch, tmp_path):
    config_path = tmp_path / "custom.json"
    config_path.write_text('{"cache": {"ttl_seconds": 600}}', encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "upstream-secret")
    monkeypatch.setenv("BASE_URL", "https://provider.example/v1")
    monkeypatch.setenv("REDIS_URL", "redis://redis.example:6379/2")
    monkeypatch.setenv("PROXY_KEYS", "first-key, second-key")
    monkeypatch.setenv("CONFIG_PATH", str(config_path))

    environment = IsolatedEnvironmentSettings()
    settings = load_app_settings(environment=environment)

    assert environment.openai_api_key == SecretStr("upstream-secret")
    assert environment.upstream_base_url == "https://provider.example/v1/"
    assert environment.redis_url == "redis://redis.example:6379/2"
    assert environment.proxy_key_values == ("first-key", "second-key")
    assert settings.cache.ttl_seconds == 600


def test_load_configuration_combines_runtime_and_application_settings(tmp_path):
    environment = IsolatedEnvironmentSettings(
        openai_api_key="upstream-secret",
        base_url="https://provider.example/v1",
        proxy_keys="proxy-secret",
        config_path=tmp_path / "missing.json",
    )

    configuration = load_configuration(environment)

    assert configuration.environment.upstream_api_key == "upstream-secret"
    assert configuration.app.cache.ttl_seconds == 3600


def test_load_configuration_requires_infrastructure_environment():
    with pytest.raises(ValueError, match="Missing required environment variables"):
        load_configuration(IsolatedEnvironmentSettings())


def test_legacy_environment_names_remain_supported(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "legacy-secret")
    monkeypatch.setenv("BASE_URL", "http://localhost:9000/v1")
    monkeypatch.setenv("PROXY_API_KEYS", "legacy-proxy")
    monkeypatch.setenv("SEMANTIC_MAX_DISTANCE", "0.2")

    environment = IsolatedEnvironmentSettings()
    settings = load_app_settings(environment=environment)

    assert environment.upstream_api_key == "legacy-secret"
    assert environment.proxy_key_values == ("legacy-proxy",)
    assert settings.cache.max_semantic_distance == 0.2


def test_malformed_json_reports_path_and_location(tmp_path):
    config_path = tmp_path / "bad.json"
    config_path.write_text('{"cache": ', encoding="utf-8")

    with pytest.raises(ValueError, match=r"Malformed JSON.*bad.json.*line 1"):
        load_app_settings(config_path, environment=IsolatedEnvironmentSettings())


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("cache", "ttl_seconds", 59),
        ("cache", "ttl_seconds", 604801),
        ("cache", "max_semantic_distance", -0.01),
        ("cache", "max_semantic_distance", 0.31),
        ("performance", "max_concurrent_embeds", 0),
        ("performance", "max_stream_buffer_bytes", 0),
    ],
)
def test_invalid_settings_are_rejected(tmp_path, section, key, value):
    config_path = tmp_path / "invalid.json"
    config_path.write_text(json.dumps({section: {key: value}}), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid settings"):
        load_app_settings(config_path, environment=IsolatedEnvironmentSettings())


def test_invalid_environment_override_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("CACHE_TTL_SECONDS", "59")

    with pytest.raises(ValueError, match="Invalid settings"):
        load_app_settings(
            tmp_path / "missing.json",
            environment=IsolatedEnvironmentSettings(),
        )


def test_non_object_settings_document_is_rejected(tmp_path):
    config_path = tmp_path / "array.json"
    config_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a JSON object"):
        load_app_settings(config_path, environment=IsolatedEnvironmentSettings())
