import tempfile

import yaml

from agent_memory_server.config import Settings, load_yaml_settings


def test_defaults(monkeypatch):
    # Clear env vars
    monkeypatch.delenv("APP_CONFIG_FILE", raising=False)
    monkeypatch.delenv("redis_url", raising=False)
    # No YAML file
    monkeypatch.chdir(tempfile.gettempdir())
    settings = Settings()
    assert settings.redis_url == "redis://localhost:6379"
    assert settings.port == 8000
    assert settings.log_level == "INFO"


def test_yaml_loading(tmp_path, monkeypatch):
    config = {"redis_url": "redis://test:6379", "port": 1234, "log_level": "DEBUG"}
    yaml_path = tmp_path / "config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f)
    monkeypatch.setenv("APP_CONFIG_FILE", str(yaml_path))
    # Remove env var overrides
    monkeypatch.delenv("redis_url", raising=False)
    monkeypatch.delenv("port", raising=False)
    monkeypatch.delenv("log_level", raising=False)
    loaded = load_yaml_settings()
    settings = Settings(**loaded)
    assert settings.redis_url == "redis://test:6379"
    assert settings.port == 1234
    assert settings.log_level == "DEBUG"


def test_env_overrides_yaml(tmp_path, monkeypatch):
    config = {"redis_url": "redis://yaml:6379", "port": 1111}
    yaml_path = tmp_path / "config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f)
    monkeypatch.setenv("APP_CONFIG_FILE", str(yaml_path))
    monkeypatch.setenv("redis_url", "redis://env:6379")
    monkeypatch.setenv("port", "2222")
    loaded = load_yaml_settings()
    settings = Settings(**loaded)
    # Env vars should override YAML
    assert settings.redis_url == "redis://env:6379"
    assert settings.port == 2222  # Pydantic auto-casts


def test_custom_config_path(tmp_path, monkeypatch):
    config = {"redis_url": "redis://custom:6379"}
    custom_path = tmp_path / "custom.yaml"
    with open(custom_path, "w") as f:
        yaml.dump(config, f)
    monkeypatch.setenv("APP_CONFIG_FILE", str(custom_path))
    loaded = load_yaml_settings()
    settings = Settings(**loaded)
    assert settings.redis_url == "redis://custom:6379"
