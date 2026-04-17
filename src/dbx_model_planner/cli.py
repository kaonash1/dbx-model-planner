"""CLI entrypoint for dbx-model-planner.

Running bare ``dbx-model-planner`` launches the interactive TUI.
Subcommands under ``auth`` manage keyring credentials.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from dbx_model_planner.auth import (
    KeyringNotAvailableError,
    clear_stored_credentials,
    run_auth_wizard,
    show_credential_status,
)
from dbx_model_planner.config import load_app_config
from dbx_model_planner.tui import run_tui

console = Console()

app = typer.Typer(
    invoke_without_command=True,
    help="Interactive model-to-compute planner for Azure Databricks.",
)
auth_app = typer.Typer(help="Manage credentials stored in system keyring.")
app.add_typer(auth_app, name="auth")


# -- Default command: launch TUI ---------------------------------------------


@app.callback()
def main(
    ctx: typer.Context,
    config_path: Path | None = typer.Option(None, "--config-path", help="Optional config TOML path."),
) -> None:
    """Launch the interactive TUI (default when no subcommand is given)."""
    if ctx.invoked_subcommand is not None:
        # A subcommand (e.g. "auth login") was given; let it run.
        return

    config = load_app_config(config_path=config_path)
    raise typer.Exit(run_tui(config=config))


# -- Auth commands -----------------------------------------------------------


@auth_app.command("login")
def auth_login() -> None:
    """Interactively configure Databricks and HuggingFace credentials."""
    try:
        run_auth_wizard(input_fn=input, output_fn=console.print)
    except KeyringNotAvailableError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)


@auth_app.command("logout")
def auth_logout() -> None:
    """Remove all stored credentials from system keyring."""
    try:
        clear_stored_credentials(input_fn=input, output_fn=console.print)
    except KeyringNotAvailableError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)


@auth_app.command("status")
def auth_status() -> None:
    """Show current credential status."""
    try:
        show_credential_status(output_fn=console.print)
    except KeyringNotAvailableError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
