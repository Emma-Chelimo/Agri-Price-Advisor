"""
Settings for the web API.

Reads the same `.streamlit/secrets.toml` the Streamlit app uses, so there is
one settings file, with environment variables as a fallback:

    ADMIN_PASSWORD      administrator sign-in
    APP_SECRET          signs sign-in tokens; created in data/app_secret.key if not set
    ANTHROPIC_API_KEY / OPENAI_API_KEY / ...   the AI assistant (see src/assistant.py)
"""

from __future__ import annotations

import os
import secrets as pysecrets
import tomllib
from functools import lru_cache

from src import config as cfg

SECRETS_FILE = cfg.ROOT / ".streamlit" / "secrets.toml"
SECRET_KEY_FILE = cfg.DATA_DIR / "app_secret.key"


class SettingsError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def file_settings() -> dict:
    """Values from secrets.toml. Raises SettingsError if the file cannot be read."""
    if not SECRETS_FILE.exists():
        return {}
    try:
        data = tomllib.loads(SECRETS_FILE.read_text(encoding="utf-8-sig"))
    except tomllib.TOMLDecodeError as e:
        raise SettingsError(
            f".streamlit/secrets.toml could not be read ({e}). Comment lines must start with #, "
            'and values need quotes, for example ADMIN_PASSWORD = "...". Fix it and restart.') from None
    return {k: str(v) for k, v in data.items() if isinstance(v, (str, int, float))}


def settings_problem() -> str | None:
    try:
        file_settings()
        return None
    except SettingsError as e:
        return str(e)


def get(key: str, default: str | None = None) -> str | None:
    try:
        value = file_settings().get(key)
    except SettingsError:
        value = None
    return value or os.environ.get(key, default)


def llm_settings() -> dict:
    try:
        return dict(file_settings())
    except SettingsError:
        return {}


@lru_cache(maxsize=1)
def app_secret() -> bytes:
    """Key for signing sign-in tokens. Kept in data/ (private, not in version control)."""
    configured = get("APP_SECRET")
    if configured:
        return configured.encode()
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SECRET_KEY_FILE.exists():
        SECRET_KEY_FILE.write_text(pysecrets.token_hex(32))
    return SECRET_KEY_FILE.read_text().strip().encode()
