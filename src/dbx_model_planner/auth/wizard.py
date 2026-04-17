from __future__ import annotations

import getpass
import re
from typing import Callable

from .credentials import DatabricksCredentials, HuggingFaceCredentials
from .keyring import KeyringNotAvailableError, credential_exists, delete_credential, load_credential, save_credential

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


DATABRICKS_CREDENTIAL_NAME = "databricks"
HUGGINGFACE_CREDENTIAL_NAME = "huggingface"


def _is_valid_databricks_url(url: str) -> bool:
    pattern = r"^https?://[a-zA-Z0-9][a-zA-Z0-9.\-]*\.azuredatabricks\.net/?$"
    return bool(re.match(pattern, url.strip()))


def _validate_databricks_connection(host: str, token: str) -> tuple[bool, str]:
    try:
        import urllib.request
        import json
        
        url = f"{host.rstrip('/')}/api/2.0/current-user"
        request = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status == 200:
                data = json.loads(response.read())
                user_name = data.get("user_name", data.get("display_name", "your account"))
                return True, user_name
            return False, f"Unexpected status: {response.status}"
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            return False, "Invalid token or unauthorized"
        return False, f"HTTP error: {exc.code}"
    except Exception as exc:
        return False, f"Connection failed: {exc}"


def _validate_huggingface_token(token: str | None) -> tuple[bool, str]:
    if not token:
        return True, "Skipped (public repos only)"
    try:
        import urllib.request
        
        url = "https://huggingface.co/api/whoami-v2"
        request = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {token}"}
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status == 200:
                import json
                data = json.loads(response.read())
                username = data.get("name", "authenticated user")
                return True, f"Authenticated as {username}"
            return False, f"Unexpected status: {response.status}"
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            return False, "Invalid token"
        return False, f"HTTP error: {exc.code}"
    except Exception as exc:
        return False, f"Validation failed: {exc}"


def _prompt_databricks_credentials(
    input_fn: InputFn,
    output_fn: OutputFn,
) -> DatabricksCredentials | None:
    output_fn("")
    output_fn("=== Databricks Setup ===")
    output_fn("")

    while True:
        output_fn("Enter your Databricks workspace URL:")
        output_fn("  (e.g., https://adb-1234567890123456.7.azuredatabricks.net)")
        host = input_fn("host> ").strip()
        
        if not host:
            output_fn("Host URL is required.")
            output_fn("")
            continue
        
        if not host.startswith(("http://", "https://")):
            host = "https://" + host
        
        if not _is_valid_databricks_url(host):
            output_fn("Warning: URL doesn't look like a standard Azure Databricks workspace.")
            output_fn("Proceed anyway? (y/n)")
            if input_fn("confirm> ").strip().lower() != "y":
                continue
        
        break

    output_fn("")
    output_fn("Enter your Databricks API token:")
    output_fn("  (Generate at: Workspace > Settings > Developer > Access tokens)")
    
    try:
        token = getpass.getpass("token> ").strip()
    except EOFError:
        token = input_fn("token> ").strip()
    
    if not token:
        output_fn("Token is required for live inventory sync.")
        return None
    
    output_fn("")
    output_fn("Validating credentials...")
    
    valid, message = _validate_databricks_connection(host, token)
    if not valid:
        output_fn(f"Validation failed: {message}")
        output_fn("Credentials will still be saved, but inventory sync may fail.")
        output_fn("")
    
    output_fn(f"  {message}")
    
    return DatabricksCredentials(host=host, token=token)


def _prompt_huggingface_credentials(
    input_fn: InputFn,
    output_fn: OutputFn,
    required: bool = False,
) -> HuggingFaceCredentials | None:
    output_fn("")
    output_fn("=== HuggingFace Setup (Optional) ===")
    output_fn("")
    
    if not required:
        output_fn("A HuggingFace token is needed for gated repos (e.g., Llama, Mistral).")
        output_fn("Press Enter to skip if you only need public models.")
        output_fn("")
    
    try:
        token_input = getpass.getpass("HF token (or Enter to skip)> ").strip()
    except EOFError:
        token_input = input_fn("HF token (or Enter to skip)> ").strip()
    
    if not token_input:
        if required:
            output_fn("Token is required.")
            return None
        return HuggingFaceCredentials(token=None)
    
    output_fn("")
    output_fn("Validating token...")
    
    valid, message = _validate_huggingface_token(token_input)
    if not valid:
        output_fn(f"Warning: {message}")
        output_fn("")
    
    output_fn(f"  {message}")
    
    return HuggingFaceCredentials(token=token_input if token_input else None)


def run_auth_wizard(
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
) -> tuple[DatabricksCredentials | None, HuggingFaceCredentials | None]:
    try:
        from .keyring import _check_keyring_available
        _check_keyring_available()
    except KeyringNotAvailableError:
        output_fn("Error: System keyring is not available.")
        output_fn("Cannot save credentials securely. Please install/configure your system's keyring.")
        return None, None

    output_fn("")
    output_fn("=== dbx-model-planner Authentication ===")
    output_fn("")
    
    dbx_creds = _prompt_databricks_credentials(input_fn, output_fn)
    if dbx_creds is None:
        return None, None
    
    hf_creds = _prompt_huggingface_credentials(input_fn, output_fn, required=False)
    
    output_fn("")
    output_fn("=== Saving Credentials ===")
    output_fn("")
    
    try:
        save_credential(DATABRICKS_CREDENTIAL_NAME, {
            "host": dbx_creds.host,
            "token": dbx_creds.token,
        })
        output_fn(f"  Databricks: {dbx_creds.host} ({dbx_creds.masked_token()})")
    except Exception as exc:
        output_fn(f"  Failed to save Databricks credentials: {exc}")
        return None, None
    
    if hf_creds and hf_creds.token:
        try:
            save_credential(HUGGINGFACE_CREDENTIAL_NAME, {
                "token": hf_creds.token,
            })
            output_fn(f"  HuggingFace: {hf_creds.masked_token()}")
        except Exception as exc:
            output_fn(f"  Warning: Failed to save HuggingFace credentials: {exc}")
    
    output_fn("")
    output_fn("Credentials saved to system keyring.")
    output_fn("")
    
    return dbx_creds, hf_creds


def load_stored_credentials() -> tuple[DatabricksCredentials | None, HuggingFaceCredentials | None]:
    dbx_data = load_credential(DATABRICKS_CREDENTIAL_NAME)
    hf_data = load_credential(HUGGINGFACE_CREDENTIAL_NAME)
    
    dbx_creds = None
    if dbx_data:
        dbx_creds = DatabricksCredentials(
            host=dbx_data.get("host", ""),
            token=dbx_data.get("token", ""),
        )
    
    hf_creds = None
    if hf_data:
        hf_creds = HuggingFaceCredentials(
            token=hf_data.get("token"),
        )
    
    return dbx_creds, hf_creds


def clear_stored_credentials(
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
) -> None:
    output_fn("")
    output_fn("This will remove all stored credentials from the system keyring.")
    output_fn("You'll need to run 'dbx-model-planner auth login' again.")
    output_fn("")
    
    confirm = input_fn("Proceed? (y/n)> ").strip().lower()
    if confirm != "y":
        output_fn("Cancelled.")
        return
    
    if credential_exists(DATABRICKS_CREDENTIAL_NAME):
        delete_credential(DATABRICKS_CREDENTIAL_NAME)
        output_fn("  Databricks credentials removed.")
    
    if credential_exists(HUGGINGFACE_CREDENTIAL_NAME):
        delete_credential(HUGGINGFACE_CREDENTIAL_NAME)
        output_fn("  HuggingFace credentials removed.")
    
    output_fn("")
    output_fn("All credentials cleared.")


def show_credential_status(output_fn: OutputFn = print) -> None:
    output_fn("")
    output_fn("=== Credential Status ===")
    output_fn("")
    
    dbx_data = load_credential(DATABRICKS_CREDENTIAL_NAME)
    if dbx_data:
        host = dbx_data.get("host", "unknown")
        token = dbx_data.get("token", "")
        masked = token[:4] + "..." + token[-4:] if len(token) > 8 else "***"
        output_fn(f"  Databricks: {host} ({masked})")
    else:
        output_fn("  Databricks: Not configured")
    
    hf_data = load_credential(HUGGINGFACE_CREDENTIAL_NAME)
    if hf_data:
        token = hf_data.get("token")
        if token:
            masked = token[:4] + "..." + token[-4:] if len(token) > 8 else "***"
            output_fn(f"  HuggingFace: {masked}")
        else:
            output_fn("  HuggingFace: Configured (no token)")
    else:
        output_fn("  HuggingFace: Not configured")
    
    output_fn("")
