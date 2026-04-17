from __future__ import annotations

import json
from typing import Any

import keyring

SERVICE_NAME = "dbx-model-planner"


class KeyringError(Exception):
    pass


class KeyringNotAvailableError(KeyringError):
    pass


def _check_keyring_available() -> None:
    try:
        keyring.get_password(SERVICE_NAME, "_health_check")
    except Exception as exc:
        raise KeyringNotAvailableError(
            "System keyring is not available. "
            "Please ensure a keyring backend is installed "
            "(e.g., Windows Credential Manager on Windows, "
            "libsecret on Linux, or Keychain on macOS)."
        ) from exc


def save_credential(credential_name: str, data: dict[str, Any]) -> None:
    try:
        _check_keyring_available()
        payload = json.dumps(data, sort_keys=True)
        keyring.set_password(SERVICE_NAME, credential_name, payload)
    except KeyringNotAvailableError:
        raise
    except Exception as exc:
        raise KeyringError(f"Failed to save credential '{credential_name}': {exc}") from exc


def load_credential(credential_name: str) -> dict[str, Any] | None:
    try:
        _check_keyring_available()
        payload = keyring.get_password(SERVICE_NAME, credential_name)
        if payload is None:
            return None
        return json.loads(payload)
    except KeyringNotAvailableError:
        raise
    except json.JSONDecodeError as exc:
        raise KeyringError(f"Credential '{credential_name}' contains invalid data: {exc}") from exc
    except Exception as exc:
        raise KeyringError(f"Failed to load credential '{credential_name}': {exc}") from exc


def delete_credential(credential_name: str) -> bool:
    try:
        _check_keyring_available()
        existing = keyring.get_password(SERVICE_NAME, credential_name)
        if existing is None:
            return False
        keyring.delete_password(SERVICE_NAME, credential_name)
        return True
    except KeyringNotAvailableError:
        raise
    except Exception as exc:
        raise KeyringError(f"Failed to delete credential '{credential_name}': {exc}") from exc


# Well-known credential names used by this application.
_KNOWN_CREDENTIAL_NAMES: list[str] = ["databricks", "huggingface"]


def list_credentials() -> list[str]:
    """Return names of credentials that are currently stored in the keyring."""
    try:
        _check_keyring_available()
    except KeyringNotAvailableError:
        return []
    found: list[str] = []
    for name in _KNOWN_CREDENTIAL_NAMES:
        try:
            if keyring.get_password(SERVICE_NAME, name) is not None:
                found.append(name)
        except Exception:
            continue
    return found


def credential_exists(credential_name: str) -> bool:
    try:
        return keyring.get_password(SERVICE_NAME, credential_name) is not None
    except KeyringNotAvailableError:
        return False
    except Exception:
        return False
