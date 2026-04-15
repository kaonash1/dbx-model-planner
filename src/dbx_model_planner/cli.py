from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from dbx_model_planner.adapters.azure import build_azure_cost_profile, enrich_azure_compute
from dbx_model_planner.adapters.huggingface import normalize_huggingface_repo_metadata
from dbx_model_planner.catalog import list_example_models, resolve_example_model
from dbx_model_planner.collectors.databricks import DatabricksInventoryCollector
from dbx_model_planner.config import AppConfig, load_app_config
from dbx_model_planner.domain import WorkloadProfile, WorkspaceComputeProfile, WorkspaceInventorySnapshot
from dbx_model_planner.planners import build_deployment_hint, recommend_compute_for_model, recommend_models_for_compute
from dbx_model_planner.presentation import (
    render_compute_fit,
    render_deployment_hint,
    render_inventory,
    render_json,
    render_model_recommendation,
)
from dbx_model_planner.runtime import build_runtime_context
from dbx_model_planner.storage import SQLiteSnapshotStore
from dbx_model_planner.terminal_app import run_terminal_app

console = Console()

app = typer.Typer(
    no_args_is_help=True,
    help="Plan model-to-compute fit, pricing, and deployment hints for Azure Databricks.",
)
inventory_app = typer.Typer(help="Sync or inspect Databricks inventory.")
model_app = typer.Typer(help="Run model-first planning commands.")
compute_app = typer.Typer(help="Run compute-first planning commands.")
price_app = typer.Typer(help="Estimate cost for a compute profile.")
deploy_app = typer.Typer(help="Generate a minimal deployment hint.")

app.add_typer(inventory_app, name="inventory")
app.add_typer(model_app, name="model")
app.add_typer(compute_app, name="compute")
app.add_typer(price_app, name="price")
app.add_typer(deploy_app, name="deploy")


ConfigPathOption = typer.Option(None, "--config-path", help="Optional config TOML path.")
DataDirOption = typer.Option(None, "--data-dir", help="Optional data directory for local snapshots.")
JsonOption = typer.Option(False, "--json", help="Emit JSON instead of text.")


@app.command("app")
def terminal_app(
    config_path: Path | None = ConfigPathOption,
    data_dir: Path | None = DataDirOption,
) -> None:
    """Open the lightweight interactive terminal app."""

    config, store = _load_runtime(config_path=config_path, data_dir=data_dir)
    snapshot, _ = _load_inventory_snapshot(store)
    raise typer.Exit(run_terminal_app(config=config, inventory=snapshot))


@inventory_app.command("sync")
def inventory_sync(
    fixture_path: Path | None = typer.Option(None, "--fixture-path", help="Path to a Databricks inventory JSON fixture."),
    config_path: Path | None = ConfigPathOption,
    data_dir: Path | None = DataDirOption,
    json_output: bool = JsonOption,
) -> None:
    """Load inventory from a fixture and store it locally."""

    _, store = _load_runtime(config_path=config_path, data_dir=data_dir)
    collector = DatabricksInventoryCollector(fixture_path=fixture_path)
    collection = collector.collect()
    store.save_inventory_snapshot(collection.snapshot)

    payload = {
        "snapshot": collection.snapshot,
        "notes": collection.notes,
        "pools": collection.pools,
        "snapshot_path": str(store.path),
    }
    _print_output(payload if json_output else collection.snapshot, as_json=json_output, text_renderer=render_inventory)
    if not json_output:
        console.print(f"Snapshot store: {store.path}")
        for note in collection.notes:
            console.print(f"- {note}")


@model_app.command("examples")
def model_examples() -> None:
    """List bundled example models."""

    for key, label, repo_id in list_example_models():
        console.print(f"{key}: {label} ({repo_id})")


@model_app.command("fit")
def model_fit(
    model_ref: str = typer.Argument(..., help="Example key, repo id, or model fixture reference."),
    model_fixture: Path | None = typer.Option(None, "--model-fixture", help="Path to a local Hugging Face metadata fixture JSON file."),
    config_path: Path | None = ConfigPathOption,
    data_dir: Path | None = DataDirOption,
    json_output: bool = JsonOption,
    batch: bool = typer.Option(False, "--batch", help="Plan for batch compute instead of online inference."),
    expected_qps: float | None = typer.Option(None, "--expected-qps", help="Optional QPS hint for embeddings or online inference."),
    target_concurrency: int | None = typer.Option(None, "--target-concurrency", help="Optional concurrency hint."),
    azure_pricing: bool = typer.Option(False, "--azure-pricing", help="Try public Azure Retail Prices lookup."),
) -> None:
    """Show candidate compute for a model."""

    config, store = _load_runtime(config_path=config_path, data_dir=data_dir)
    snapshot, inventory_source = _load_inventory_snapshot(store)
    normalized = _load_model(model_ref, model_fixture)
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
    payload = {
        "inventory_source": inventory_source,
        "model": normalized,
        "recommendation": recommendation,
    }
    _print_output(payload if json_output else recommendation, as_json=json_output, text_renderer=render_model_recommendation)


@compute_app.command("fit")
def compute_fit(
    node_type: str = typer.Argument(..., help="Databricks node type id, for example Standard_NC6s_v3."),
    config_path: Path | None = ConfigPathOption,
    data_dir: Path | None = DataDirOption,
    json_output: bool = JsonOption,
    azure_pricing: bool = typer.Option(False, "--azure-pricing", help="Try public Azure Retail Prices lookup."),
) -> None:
    """Show realistic bundled models for a compute profile."""

    config, store = _load_runtime(config_path=config_path, data_dir=data_dir)
    snapshot, inventory_source = _load_inventory_snapshot(store)
    compute = _resolve_compute(snapshot, node_type)
    models = [resolve_example_model(key).model_profile for key, _, _ in list_example_models()]

    vm_hourly_rate = None
    if azure_pricing:
        enrichment = enrich_azure_compute(
            compute,
            dbu_hourly_rate=config.databricks.dbu_hourly_rate or None,
            discount_rate=config.pricing.discount_rate,
            vat_rate=config.pricing.vat_rate,
            currency_code=config.pricing.currency_code,
        )
        if enrichment.selected_price is not None:
            vm_hourly_rate = enrichment.selected_price.unit_price

    report = recommend_models_for_compute(
        config=config,
        compute=compute,
        models=models,
        vm_hourly_rate=vm_hourly_rate,
        dbu_hourly_rate=config.databricks.dbu_hourly_rate or None,
    )
    payload = {
        "inventory_source": inventory_source,
        "report": report,
    }
    _print_output(payload if json_output else report, as_json=json_output, text_renderer=render_compute_fit)


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
    snapshot, inventory_source = _load_inventory_snapshot(store)
    compute = _resolve_compute(snapshot, node_type)

    if azure_pricing:
        enrichment = enrich_azure_compute(
            compute,
            dbu_hourly_rate=dbu_rate if dbu_rate is not None else (config.databricks.dbu_hourly_rate or None),
            discount_rate=config.pricing.discount_rate,
            vat_rate=config.pricing.vat_rate,
            currency_code=config.pricing.currency_code,
        )
        payload = {
            "inventory_source": inventory_source,
            "enrichment": enrichment,
        }
        _print_output(payload if json_output else enrichment, as_json=json_output, text_renderer=render_json)
        return

    if vm_rate is None:
        raise typer.BadParameter("Provide --vm-rate or enable --azure-pricing for a price estimate.")

    cost = build_azure_cost_profile(
        vm_hourly_rate=vm_rate,
        dbu_hourly_rate=dbu_rate if dbu_rate is not None else (config.databricks.dbu_hourly_rate or None),
        discount_rate=config.pricing.discount_rate,
        vat_rate=config.pricing.vat_rate,
        currency_code=config.pricing.currency_code,
    )
    payload = {
        "inventory_source": inventory_source,
        "node_type_id": compute.node_type_id,
        "cost": cost,
    }
    _print_output(payload, as_json=json_output, text_renderer=render_json)


@deploy_app.command("plan")
def deploy_plan(
    model_ref: str = typer.Argument(..., help="Example key, repo id, or model fixture reference."),
    model_fixture: Path | None = typer.Option(None, "--model-fixture", help="Path to a local Hugging Face metadata fixture JSON file."),
    config_path: Path | None = ConfigPathOption,
    data_dir: Path | None = DataDirOption,
    json_output: bool = JsonOption,
    azure_pricing: bool = typer.Option(False, "--azure-pricing", help="Try public Azure Retail Prices lookup."),
) -> None:
    """Build a simple UC volume and cluster hint."""

    config, store = _load_runtime(config_path=config_path, data_dir=data_dir)
    snapshot, inventory_source = _load_inventory_snapshot(store)
    normalized = _load_model(model_ref, model_fixture)
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
    payload = {
        "inventory_source": inventory_source,
        "recommendation": recommendation,
        "hint": hint,
    }
    _print_output(payload if json_output else hint, as_json=json_output, text_renderer=render_deployment_hint)


def _load_runtime(config_path: Path | None = None, data_dir: Path | None = None) -> tuple[AppConfig, SQLiteSnapshotStore]:
    config = load_app_config(config_path=config_path)
    try:
        context = build_runtime_context(config, config_path=config_path, data_dir=data_dir)
    except OSError:
        fallback_dir = Path(tempfile.gettempdir()) / "dbx-model-planner"
        context = build_runtime_context(config, config_path=config_path, data_dir=fallback_dir)
    store = SQLiteSnapshotStore(context.paths.snapshot_db_path)
    return config, store


def _load_inventory_snapshot(store: SQLiteSnapshotStore) -> tuple[WorkspaceInventorySnapshot, str]:
    stored = store.load_inventory_snapshot()
    if stored is not None:
        return stored, "stored"
    return DatabricksInventoryCollector().collect_snapshot(), "mock"


def _load_model(model_ref: str, model_fixture: Path | None) -> Any:
    if model_fixture:
        payload = json.loads(model_fixture.read_text(encoding="utf-8"))
        return normalize_huggingface_repo_metadata(payload)
    return resolve_example_model(model_ref)


def _resolve_compute(snapshot: WorkspaceInventorySnapshot, node_type: str) -> WorkspaceComputeProfile:
    for compute in snapshot.compute:
        if compute.node_type_id == node_type:
            return compute
    return WorkspaceComputeProfile(node_type_id=node_type, region=snapshot.region)


def _dbu_pricing_map(config: AppConfig, snapshot: WorkspaceInventorySnapshot) -> dict[str, float]:
    if config.databricks.dbu_hourly_rate <= 0:
        return {}
    return {compute.node_type_id: config.databricks.dbu_hourly_rate for compute in snapshot.compute}


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
                dbu_hourly_rate=config.databricks.dbu_hourly_rate or None,
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


def _print_output(payload: Any, *, as_json: bool, text_renderer) -> None:
    if as_json:
        console.print(render_json(payload))
        return
    console.print(text_renderer(payload))


if __name__ == "__main__":
    app()
