from __future__ import annotations

from collections.abc import Callable

from .adapters.huggingface import GatedRepoError, HuggingFaceAPIError, fetch_huggingface_metadata, normalize_huggingface_repo_metadata
from .auth import (
    DatabricksCredentials,
    HuggingFaceCredentials,
    KeyringNotAvailableError,
    credential_exists,
    load_stored_credentials,
    run_auth_wizard,
)
from .collectors.databricks import DatabricksAPIError, DatabricksInventoryCollector
from .config import AppConfig
from .domain import WorkloadProfile, WorkspaceInventorySnapshot
from .planners import recommend_compute_for_model
from .presentation import render_inventory, render_model_recommendation


InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


def run_terminal_app(
    *,
    config: AppConfig,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
) -> int:
    """Run the interactive terminal app.

    Flow:
      1. Load or prompt for credentials
      2. Sync live inventory from Databricks
      3. Interactive menu loop
    """

    output_fn("")
    output_fn("dbx-model-planner")
    output_fn("Terminal planner for Azure Databricks model fit")
    output_fn("")

    # -- Step 1: Credentials ------------------------------------------------
    dbx_creds, hf_creds = _ensure_credentials(input_fn, output_fn)
    if dbx_creds is None:
        output_fn("Cannot continue without Databricks credentials.")
        return 1

    # -- Step 2: Live inventory sync ----------------------------------------
    output_fn("")
    output_fn("Syncing workspace inventory...")
    try:
        collector = DatabricksInventoryCollector(credentials=dbx_creds)
        collection = collector.collect()
        inventory = collection.snapshot
        for note in collection.notes:
            output_fn(f"  {note}")
        output_fn("")
    except DatabricksAPIError as exc:
        output_fn(f"Error: {exc}")
        return 1

    # -- Step 3: Interactive menu -------------------------------------------
    while True:
        output_fn("Choose an action:")
        output_fn("  1. Show workspace inventory")
        output_fn("  2. Model -> compute fit")
        output_fn("  q. Quit")
        choice = input_fn("> ").strip().lower()

        if choice in {"q", "quit", "exit"}:
            output_fn("Bye.")
            return 0
        if choice == "1":
            _action_show_inventory(inventory, output_fn)
        elif choice == "2":
            _action_model_fit(config, inventory, hf_creds, input_fn, output_fn)
        else:
            output_fn("Unknown choice. Enter 1, 2, or q.")
            output_fn("")


def _ensure_credentials(
    input_fn: InputFn,
    output_fn: OutputFn,
) -> tuple[DatabricksCredentials | None, HuggingFaceCredentials | None]:
    """Load stored credentials or run the auth wizard."""

    dbx_creds: DatabricksCredentials | None = None
    hf_creds: HuggingFaceCredentials | None = None

    if credential_exists("databricks"):
        try:
            dbx_creds, hf_creds = load_stored_credentials()
            if dbx_creds:
                output_fn(f"Loaded Databricks credentials: {dbx_creds.host}")
                if hf_creds and hf_creds.has_token:
                    output_fn(f"Loaded HuggingFace token: {hf_creds.masked_token()}")
                return dbx_creds, hf_creds
        except KeyringNotAvailableError:
            output_fn("System keyring unavailable.")

    output_fn("No stored credentials found. Starting setup...")
    output_fn("")
    try:
        dbx_creds, hf_creds = run_auth_wizard(input_fn=input_fn, output_fn=output_fn)
    except KeyringNotAvailableError as exc:
        output_fn(f"Error: {exc}")

    return dbx_creds, hf_creds


def _action_show_inventory(inventory: WorkspaceInventorySnapshot, output_fn: OutputFn) -> None:
    output_fn("")
    output_fn(render_inventory(inventory))
    output_fn("")


def _action_model_fit(
    config: AppConfig,
    inventory: WorkspaceInventorySnapshot,
    hf_creds: HuggingFaceCredentials | None,
    input_fn: InputFn,
    output_fn: OutputFn,
) -> None:
    repo_id = _prompt_model_ref(input_fn, output_fn)
    if repo_id is None:
        return

    model = _fetch_and_normalize_model(repo_id, hf_creds, output_fn)
    if model is None:
        return

    recommendation = recommend_compute_for_model(
        config=config,
        inventory=inventory,
        model=model,
        workload=WorkloadProfile(workload_name="terminal-app", online=True),
    )
    output_fn("")
    output_fn(render_model_recommendation(recommendation))
    output_fn("")


def _prompt_model_ref(input_fn: InputFn, output_fn: OutputFn) -> str | None:
    output_fn("")
    output_fn("Enter a HuggingFace model ID:")
    output_fn("  (e.g., meta-llama/Llama-3.1-8B-Instruct)")
    output_fn("  Or press Enter to cancel")
    choice = input_fn("model> ").strip()
    if not choice:
        output_fn("")
        return None
    return choice


def _fetch_and_normalize_model(
    repo_id: str,
    credentials: HuggingFaceCredentials | None,
    output_fn: OutputFn,
):
    """Fetch model metadata from HuggingFace and normalize it."""

    output_fn(f"Fetching model metadata for '{repo_id}'...")

    try:
        raw_metadata = fetch_huggingface_metadata(repo_id, credentials=credentials)
        normalized = normalize_huggingface_repo_metadata(raw_metadata)
        profile = normalized.model_profile
        output_fn(f"  Family: {profile.family.value}")
        param_str = f"{profile.parameter_count:,}" if profile.parameter_count else "unknown"
        output_fn(f"  Parameters: {param_str}")
        return profile
    except GatedRepoError as exc:
        output_fn(f"Error: {exc}")
        if not credentials or not credentials.token:
            output_fn("Run 'dbx-model-planner auth login' to add your HuggingFace token.")
        return None
    except HuggingFaceAPIError as exc:
        output_fn(f"Error: {exc}")
        return None
    except Exception as exc:
        output_fn(f"Unexpected error: {exc}")
        return None
