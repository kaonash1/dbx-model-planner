from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from dbx_model_planner.adapters.azure import build_azure_cost_profile, enrich_azure_compute
from dbx_model_planner.adapters.huggingface import (
    GatedRepoError,
    HuggingFaceAPIError,
    fetch_huggingface_metadata,
    normalize_huggingface_repo_metadata,
)
from dbx_model_planner.auth import (
    KeyringNotAvailableError,
    clear_stored_credentials,
    load_stored_credentials,
    run_auth_wizard,
    show_credential_status,
)
from dbx_model_planner.collectors.databricks import DatabricksAPIError, DatabricksInventoryCollector
from dbx_model_planner.collectors.databricks.inventory import enrich_dbu_rates
from dbx_model_planner.adapters.azure.dbu_rates import (
    build_dbu_rate_cache,
    fetch_dbu_rates,
    fetch_dbu_unit_prices,
    load_dbu_cache,
    save_dbu_cache,
)
from dbx_model_planner.config import AppConfig, load_app_config
from dbx_model_planner.domain import WorkloadProfile, WorkspaceComputeProfile, WorkspaceInventorySnapshot
from dbx_model_planner.planners import build_deployment_hint, recommend_compute_for_model
from dbx_model_planner.presentation import (
    render_deployment_hint,
    render_inventory,
    render_json,
    render_model_recommendation,
)
from dbx_model_planner.runtime import build_runtime_context
from dbx_model_planner.storage import SQLiteSnapshotStore
from dbx_model_planner.terminal_app import run_terminal_app
from dbx_model_planner.tui import run_tui

console = Console()

app = typer.Typer(
    no_args_is_help=True,
    help="Plan model-to-compute fit, pricing, and deployment hints for Azure Databricks.",
)
auth_app = typer.Typer(help="Manage credentials stored in system keyring.")
inventory_app = typer.Typer(help="Sync or inspect Databricks inventory.")
model_app = typer.Typer(help="Run model-first planning commands.")
price_app = typer.Typer(help="Estimate cost for a compute profile.")
deploy_app = typer.Typer(help="Generate a minimal deployment hint.")

app.add_typer(auth_app, name="auth")
app.add_typer(inventory_app, name="inventory")
app.add_typer(model_app, name="model")
app.add_typer(price_app, name="price")
app.add_typer(deploy_app, name="deploy")


ConfigPathOption = typer.Option(None, "--config-path", help="Optional config TOML path.")
DataDirOption = typer.Option(None, "--data-dir", help="Optional data directory for local snapshots.")
JsonOption = typer.Option(False, "--json", help="Emit JSON instead of text.")


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


# -- Interactive app ---------------------------------------------------------


@app.command("app")
def terminal_app_cmd(
    config_path: Path | None = ConfigPathOption,
    classic: bool = typer.Option(False, "--classic", help="Use text-based terminal app instead of TUI."),
) -> None:
    """Open the interactive terminal planner (credentials required)."""

    config = _load_config(config_path)
    if classic:
        raise typer.Exit(run_terminal_app(config=config))
    raise typer.Exit(run_tui(config=config))


# -- Inventory ---------------------------------------------------------------


@inventory_app.command("sync")
def inventory_sync(
    config_path: Path | None = ConfigPathOption,
    data_dir: Path | None = DataDirOption,
    json_output: bool = JsonOption,
) -> None:
    """Sync Databricks inventory from live workspace API."""

    _, store = _load_runtime(config_path=config_path, data_dir=data_dir)
    dbx_creds = _require_databricks_credentials()

    try:
        collector = DatabricksInventoryCollector(credentials=dbx_creds)
        collection = collector.collect()
    except DatabricksAPIError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    _enrich_snapshot_dbu_rates(collection.snapshot)
    store.save_inventory_snapshot(collection.snapshot)

    if json_output:
        console.print(render_json({"snapshot": collection.snapshot, "notes": collection.notes}))
    else:
        console.print(render_inventory(collection.snapshot))
        console.print(f"\nSnapshot saved to: {store.path}")
        for note in collection.notes:
            console.print(f"  - {note}")


# -- Model -------------------------------------------------------------------


@model_app.command("fit")
def model_fit(
    model_ref: str = typer.Argument(..., help="HuggingFace repository ID (e.g., meta-llama/Llama-3.1-8B-Instruct)."),
    config_path: Path | None = ConfigPathOption,
    data_dir: Path | None = DataDirOption,
    json_output: bool = JsonOption,
    batch: bool = typer.Option(False, "--batch", help="Plan for batch compute instead of online inference."),
    expected_qps: float | None = typer.Option(None, "--expected-qps", help="Optional QPS hint."),
    target_concurrency: int | None = typer.Option(None, "--target-concurrency", help="Optional concurrency hint."),
    azure_pricing: bool = typer.Option(False, "--azure-pricing", help="Try public Azure Retail Prices lookup."),
) -> None:
    """Show candidate compute for a HuggingFace model."""

    config, store = _load_runtime(config_path=config_path, data_dir=data_dir)
    snapshot = _load_inventory(store)
    normalized = _fetch_model(model_ref)
    model = normalized.model_profile

    workload = WorkloadProfile(
        workload_name="model-fit",
        online=not batch,
        expected_qps=expected_qps,
        target_concurrency=target_concurrency,
    )

    vm_pricing = {}
    if azure_pricing:
        vm_pricing, _ = _azure_vm_pricing_map(config, snapshot)

    recommendation = recommend_compute_for_model(
        config=config,
        inventory=snapshot,
        model=model,
        workload=workload,
        vm_pricing=vm_pricing,
        dbu_pricing=_dbu_pricing_map(config, snapshot),
    )

    if json_output:
        console.print(render_json({"model": normalized, "recommendation": recommendation}))
    else:
        console.print(render_model_recommendation(recommendation))


# -- Price -------------------------------------------------------------------


@price_app.command("estimate")
def price_estimate(
    node_type: str = typer.Argument(..., help="Databricks node type id."),
    vm_rate: float | None = typer.Option(None, "--vm-rate", help="Explicit VM hourly rate override."),
    dbu_rate: float | None = typer.Option(None, "--dbu-rate", help="Explicit DBU hourly rate override."),
    azure_pricing: bool = typer.Option(False, "--azure-pricing", help="Try public Azure Retail Prices lookup."),
    config_path: Path | None = ConfigPathOption,
    data_dir: Path | None = DataDirOption,
    json_output: bool = JsonOption,
) -> None:
    """Estimate price from Azure or explicit rates."""

    config, store = _load_runtime(config_path=config_path, data_dir=data_dir)
    snapshot = _load_inventory(store)
    compute = _resolve_compute(snapshot, node_type)

    if azure_pricing:
        enrichment = enrich_azure_compute(
            compute,
            dbu_hourly_rate=dbu_rate if dbu_rate is not None else None,
            discount_rate=config.pricing.discount_rate,
            vat_rate=config.pricing.vat_rate,
            currency_code=config.pricing.currency_code,
        )
        console.print(render_json(enrichment))
        return

    if vm_rate is None:
        raise typer.BadParameter("Provide --vm-rate or enable --azure-pricing for a price estimate.")

    cost = build_azure_cost_profile(
        vm_hourly_rate=vm_rate,
        dbu_hourly_rate=dbu_rate if dbu_rate is not None else None,
        discount_rate=config.pricing.discount_rate,
        vat_rate=config.pricing.vat_rate,
        currency_code=config.pricing.currency_code,
    )
    console.print(render_json({"node_type_id": compute.node_type_id, "cost": cost}))


# -- Deploy ------------------------------------------------------------------


@deploy_app.command("plan")
def deploy_plan(
    model_ref: str = typer.Argument(..., help="HuggingFace repository ID."),
    config_path: Path | None = ConfigPathOption,
    data_dir: Path | None = DataDirOption,
    json_output: bool = JsonOption,
    azure_pricing: bool = typer.Option(False, "--azure-pricing", help="Try public Azure Retail Prices lookup."),
) -> None:
    """Build a UC volume and cluster hint for a HuggingFace model."""

    config, store = _load_runtime(config_path=config_path, data_dir=data_dir)
    snapshot = _load_inventory(store)
    normalized = _fetch_model(model_ref)
    model = normalized.model_profile

    vm_pricing = {}
    if azure_pricing:
        vm_pricing, _ = _azure_vm_pricing_map(config, snapshot)

    recommendation = recommend_compute_for_model(
        config=config,
        inventory=snapshot,
        model=model,
        workload=WorkloadProfile(workload_name="deploy-plan", online=True),
        vm_pricing=vm_pricing,
        dbu_pricing=_dbu_pricing_map(config, snapshot),
    )
    hint = build_deployment_hint(config, snapshot, model, recommendation)

    if json_output:
        console.print(render_json({"recommendation": recommendation, "hint": hint}))
    else:
        console.print(render_deployment_hint(hint))


# -- Helpers -----------------------------------------------------------------


def _load_config(config_path: Path | None = None) -> AppConfig:
    return load_app_config(config_path=config_path)


def _load_runtime(
    config_path: Path | None = None,
    data_dir: Path | None = None,
) -> tuple[AppConfig, SQLiteSnapshotStore]:
    config = load_app_config(config_path=config_path)
    try:
        context = build_runtime_context(config, config_path=config_path, data_dir=data_dir)
    except OSError:
        fallback_dir = Path(tempfile.gettempdir()) / "dbx-model-planner"
        context = build_runtime_context(config, config_path=config_path, data_dir=fallback_dir)
    store = SQLiteSnapshotStore(context.paths.snapshot_db_path)
    return config, store


def _require_databricks_credentials():
    """Load Databricks credentials or exit with an error."""
    try:
        dbx_creds, _ = load_stored_credentials()
    except KeyringNotAvailableError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    if dbx_creds is None:
        console.print("[red]Error:[/red] No Databricks credentials found.")
        console.print("Run 'dbx-model-planner auth login' first.")
        raise typer.Exit(code=1)
    return dbx_creds


def _load_inventory(store: SQLiteSnapshotStore) -> WorkspaceInventorySnapshot:
    """Load inventory from snapshot store, or fail with guidance."""
    stored = store.load_inventory_snapshot()
    if stored is not None:
        # Ensure DBU rates are populated (may be missing on old snapshots)
        has_dbu = any(c.dbu_per_hour is not None for c in stored.compute)
        if not has_dbu:
            _enrich_snapshot_dbu_rates(stored)
        return stored
    console.print("[yellow]No inventory snapshot found.[/yellow]")
    console.print("Run 'dbx-model-planner inventory sync' first.")
    raise typer.Exit(code=1)


def _fetch_model(model_ref: str):
    """Fetch and normalize a HuggingFace model, or exit with an error."""
    try:
        hf_creds = None
        try:
            _, hf_creds = load_stored_credentials()
        except KeyringNotAvailableError:
            pass

        raw_metadata = fetch_huggingface_metadata(model_ref, credentials=hf_creds)
        return normalize_huggingface_repo_metadata(raw_metadata)
    except GatedRepoError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        console.print("Run 'dbx-model-planner auth login' to add your HuggingFace token.")
        raise typer.Exit(code=1)
    except HuggingFaceAPIError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)


def _resolve_compute(snapshot: WorkspaceInventorySnapshot, node_type: str) -> WorkspaceComputeProfile:
    for compute in snapshot.compute:
        if compute.node_type_id == node_type:
            return compute
    return WorkspaceComputeProfile(node_type_id=node_type, region=snapshot.region)


def _dbu_pricing_map(config: AppConfig, snapshot: WorkspaceInventorySnapshot) -> dict[str, float]:
    """Build {node_type_id: dbu_hourly_cost} for all compute nodes.

    Uses the per-DBU unit price from the DBU rate cache (fetched from
    the Azure Retail Prices API in USD) if available,
    falling back to ``config.databricks.dbu_rate_per_unit``.

    Formula: DBU_cost = DBU_Count × per_DBU_unit_rate.
    """
    rate = config.databricks.dbu_rate_per_unit

    # Check if the DBU rate cache has API-derived per-DBU prices
    cache = load_dbu_cache()
    if cache is not None and cache.dbu_unit_prices and cache.unit_price_currency == config.pricing.currency_code:
        wt = config.databricks.workload_type
        api_rate = cache.dbu_unit_prices.get(wt)
        if api_rate is not None:
            rate = api_rate

    if rate <= 0:
        return {}
    result: dict[str, float] = {}
    for compute in snapshot.compute:
        dbu_per_hour = compute.dbu_per_hour
        if dbu_per_hour is not None and dbu_per_hour > 0:
            result[compute.node_type_id] = round(dbu_per_hour * rate, 4)
    return result


def _enrich_snapshot_dbu_rates(snapshot: WorkspaceInventorySnapshot) -> None:
    """Enrich snapshot compute nodes with real DBU rates from Azure pricing page."""
    cache = load_dbu_cache()
    if cache is not None and cache.is_populated:
        enrich_dbu_rates(snapshot.compute, cache.as_dict())
        return
    try:
        entries = fetch_dbu_rates(timeout=60.0)
        if entries:
            new_cache = build_dbu_rate_cache(entries)
            enrich_dbu_rates(snapshot.compute, new_cache.as_dict())
            try:
                save_dbu_cache(new_cache)
            except Exception:
                pass
    except Exception:
        pass


def _azure_vm_pricing_map(
    config: AppConfig,
    snapshot: WorkspaceInventorySnapshot,
) -> tuple[dict[str, float], dict[str, Any]]:
    prices: dict[str, float] = {}
    details: dict[str, Any] = {}
    for compute in snapshot.compute:
        try:
            enrichment = enrich_azure_compute(
                compute,
                dbu_hourly_rate=None,
                discount_rate=config.pricing.discount_rate,
                vat_rate=config.pricing.vat_rate,
                currency_code=config.pricing.currency_code,
            )
        except Exception as exc:
            details[compute.node_type_id] = {"error": str(exc)}
            continue
        if enrichment.selected_price is not None:
            prices[compute.node_type_id] = enrichment.selected_price.unit_price
        details[compute.node_type_id] = render_json(enrichment)
    return prices, details


if __name__ == "__main__":
    app()
