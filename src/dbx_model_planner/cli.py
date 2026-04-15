from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dbx-model-planner",
        description="Plan model-to-compute fit, pricing, and deployment hints for Azure Databricks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory_parser = subparsers.add_parser("inventory", help="Sync or inspect Databricks inventory.")
    inventory_subparsers = inventory_parser.add_subparsers(dest="inventory_command", required=True)
    inventory_sync = inventory_subparsers.add_parser("sync", help="Load inventory from a fixture and store it locally.")
    _add_common_runtime_args(inventory_sync)
    inventory_sync.add_argument("--fixture-path", help="Path to a Databricks inventory JSON fixture.")
    inventory_sync.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    inventory_sync.set_defaults(handler=_handle_inventory_sync)

    model_parser = subparsers.add_parser("model", help="Run model-first planning commands.")
    model_subparsers = model_parser.add_subparsers(dest="model_command", required=True)
    model_examples = model_subparsers.add_parser("examples", help="List bundled example models.")
    model_examples.set_defaults(handler=_handle_model_examples)

    model_fit = model_subparsers.add_parser("fit", help="Show candidate compute for a model.")
    _add_common_runtime_args(model_fit)
    model_fit.add_argument("model_ref", help="Example key, repo id, or model fixture reference.")
    model_fit.add_argument("--model-fixture", help="Path to a local Hugging Face metadata fixture JSON file.")
    model_fit.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    model_fit.add_argument("--batch", action="store_true", help="Plan for batch compute instead of online inference.")
    model_fit.add_argument("--expected-qps", type=float, help="Optional QPS hint for embeddings or online inference.")
    model_fit.add_argument("--target-concurrency", type=int, help="Optional concurrency hint.")
    model_fit.add_argument("--azure-pricing", action="store_true", help="Try public Azure Retail Prices lookup.")
    model_fit.set_defaults(handler=_handle_model_fit)

    compute_parser = subparsers.add_parser("compute", help="Run compute-first planning commands.")
    compute_subparsers = compute_parser.add_subparsers(dest="compute_command", required=True)
    compute_fit = compute_subparsers.add_parser("fit", help="Show realistic bundled models for a compute profile.")
    _add_common_runtime_args(compute_fit)
    compute_fit.add_argument("node_type", help="Databricks node type id, for example Standard_NC6s_v3.")
    compute_fit.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    compute_fit.add_argument("--azure-pricing", action="store_true", help="Try public Azure Retail Prices lookup.")
    compute_fit.set_defaults(handler=_handle_compute_fit)

    price_parser = subparsers.add_parser("price", help="Estimate cost for a compute profile.")
    price_subparsers = price_parser.add_subparsers(dest="price_command", required=True)
    price_estimate = price_subparsers.add_parser("estimate", help="Estimate price from Azure or explicit rates.")
    _add_common_runtime_args(price_estimate)
    price_estimate.add_argument("node_type", help="Databricks node type id.")
    price_estimate.add_argument("--vm-rate", type=float, help="Explicit VM hourly rate override.")
    price_estimate.add_argument("--dbu-rate", type=float, help="Explicit DBU hourly rate override.")
    price_estimate.add_argument("--azure-pricing", action="store_true", help="Try public Azure Retail Prices lookup.")
    price_estimate.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    price_estimate.set_defaults(handler=_handle_price_estimate)

    deploy_parser = subparsers.add_parser("deploy", help="Generate a minimal deployment hint.")
    deploy_subparsers = deploy_parser.add_subparsers(dest="deploy_command", required=True)
    deploy_plan = deploy_subparsers.add_parser("plan", help="Build a simple UC volume and cluster hint.")
    _add_common_runtime_args(deploy_plan)
    deploy_plan.add_argument("model_ref", help="Example key, repo id, or model fixture reference.")
    deploy_plan.add_argument("--model-fixture", help="Path to a local Hugging Face metadata fixture JSON file.")
    deploy_plan.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    deploy_plan.add_argument("--azure-pricing", action="store_true", help="Try public Azure Retail Prices lookup.")
    deploy_plan.set_defaults(handler=_handle_deploy_plan)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _add_common_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config-path", help="Optional config TOML path.")
    parser.add_argument("--data-dir", help="Optional data directory for local snapshots.")


def _load_runtime(args: argparse.Namespace) -> tuple[AppConfig, SQLiteSnapshotStore]:
    config = load_app_config(config_path=getattr(args, "config_path", None))
    try:
        context = build_runtime_context(
            config,
            config_path=getattr(args, "config_path", None),
            data_dir=getattr(args, "data_dir", None),
        )
    except OSError:
        fallback_dir = Path(tempfile.gettempdir()) / "dbx-model-planner"
        context = build_runtime_context(
            config,
            config_path=getattr(args, "config_path", None),
            data_dir=fallback_dir,
        )
    store = SQLiteSnapshotStore(context.paths.snapshot_db_path)
    return config, store


def _load_inventory_snapshot(store: SQLiteSnapshotStore) -> tuple[WorkspaceInventorySnapshot, str]:
    stored = store.load_inventory_snapshot()
    if stored is not None:
        return stored, "stored"
    return DatabricksInventoryCollector().collect_snapshot(), "mock"


def _load_model(model_ref: str, model_fixture: str | None) -> Any:
    if model_fixture:
        payload = json.loads(Path(model_fixture).read_text(encoding="utf-8"))
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
        print(render_json(payload))
        return
    print(text_renderer(payload))


def _handle_inventory_sync(args: argparse.Namespace) -> int:
    _, store = _load_runtime(args)
    collector = DatabricksInventoryCollector(fixture_path=args.fixture_path)
    collection = collector.collect()
    store.save_inventory_snapshot(collection.snapshot)

    payload = {
        "snapshot": collection.snapshot,
        "notes": collection.notes,
        "pools": collection.pools,
        "snapshot_path": str(store.path),
    }
    _print_output(payload if args.json else collection.snapshot, as_json=args.json, text_renderer=render_inventory)
    if not args.json:
        print(f"Snapshot store: {store.path}")
        for note in collection.notes:
            print(f"- {note}")
    return 0


def _handle_model_examples(_: argparse.Namespace) -> int:
    for key, label, repo_id in list_example_models():
        print(f"{key}: {label} ({repo_id})")
    return 0


def _handle_model_fit(args: argparse.Namespace) -> int:
    config, store = _load_runtime(args)
    snapshot, inventory_source = _load_inventory_snapshot(store)
    normalized = _load_model(args.model_ref, args.model_fixture)
    model = normalized.model_profile
    workload = WorkloadProfile(
        workload_name="model-fit",
        online=not args.batch,
        expected_qps=args.expected_qps,
        target_concurrency=args.target_concurrency,
    )

    vm_pricing = {}
    if args.azure_pricing:
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
    _print_output(payload if args.json else recommendation, as_json=args.json, text_renderer=render_model_recommendation)
    return 0


def _handle_compute_fit(args: argparse.Namespace) -> int:
    config, store = _load_runtime(args)
    snapshot, inventory_source = _load_inventory_snapshot(store)
    compute = _resolve_compute(snapshot, args.node_type)
    models = [resolve_example_model(key).model_profile for key, _, _ in list_example_models()]

    vm_hourly_rate = None
    if args.azure_pricing:
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
    _print_output(payload if args.json else report, as_json=args.json, text_renderer=render_compute_fit)
    return 0


def _handle_price_estimate(args: argparse.Namespace) -> int:
    config, store = _load_runtime(args)
    snapshot, inventory_source = _load_inventory_snapshot(store)
    compute = _resolve_compute(snapshot, args.node_type)

    if args.azure_pricing:
        enrichment = enrich_azure_compute(
            compute,
            dbu_hourly_rate=args.dbu_rate if args.dbu_rate is not None else (config.databricks.dbu_hourly_rate or None),
            discount_rate=config.pricing.discount_rate,
            vat_rate=config.pricing.vat_rate,
            currency_code=config.pricing.currency_code,
        )
        payload = {
            "inventory_source": inventory_source,
            "enrichment": enrichment,
        }
        _print_output(payload if args.json else enrichment, as_json=args.json, text_renderer=render_json)
        return 0

    if args.vm_rate is None:
        raise ValueError("Provide --vm-rate or enable --azure-pricing for a price estimate.")

    cost = build_azure_cost_profile(
        vm_hourly_rate=args.vm_rate,
        dbu_hourly_rate=args.dbu_rate if args.dbu_rate is not None else (config.databricks.dbu_hourly_rate or None),
        discount_rate=config.pricing.discount_rate,
        vat_rate=config.pricing.vat_rate,
        currency_code=config.pricing.currency_code,
    )
    payload = {
        "inventory_source": inventory_source,
        "node_type_id": compute.node_type_id,
        "cost": cost,
    }
    _print_output(payload if args.json else payload, as_json=args.json, text_renderer=render_json)
    return 0


def _handle_deploy_plan(args: argparse.Namespace) -> int:
    config, store = _load_runtime(args)
    snapshot, inventory_source = _load_inventory_snapshot(store)
    normalized = _load_model(args.model_ref, args.model_fixture)
    model = normalized.model_profile
    vm_pricing = {}
    if args.azure_pricing:
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
    _print_output(payload if args.json else hint, as_json=args.json, text_renderer=render_deployment_hint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
