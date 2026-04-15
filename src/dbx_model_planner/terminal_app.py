from __future__ import annotations

from collections.abc import Callable

from .catalog import list_example_models, resolve_example_model
from .config import AppConfig
from .domain import WorkloadProfile, WorkspaceComputeProfile, WorkspaceInventorySnapshot
from .planners import build_deployment_hint, recommend_compute_for_model, recommend_models_for_compute
from .presentation import render_compute_fit, render_deployment_hint, render_inventory, render_model_recommendation


InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


def run_terminal_app(
    *,
    config: AppConfig,
    inventory: WorkspaceInventorySnapshot,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
) -> int:
    """Run the lightweight terminal app.

    This deliberately stays dependency-free. It is a guided CLI mode, not a
    full-screen TUI, so the core command API remains easy to script.
    """

    output_fn("")
    output_fn("dbx-model-planner")
    output_fn("Terminal planner for Azure Databricks model fit")
    output_fn("")

    while True:
        output_fn("Choose an action:")
        output_fn("  1. Show Databricks inventory")
        output_fn("  2. Model -> compute fit")
        output_fn("  3. Compute -> model fit")
        output_fn("  4. Deployment hint")
        output_fn("  q. Quit")
        choice = input_fn("> ").strip().lower()

        if choice in {"q", "quit", "exit"}:
            output_fn("Bye.")
            return 0
        if choice == "1":
            output_fn("")
            output_fn(render_inventory(inventory))
            output_fn("")
            continue
        if choice == "2":
            model_ref = _choose_model(input_fn=input_fn, output_fn=output_fn)
            if model_ref is None:
                continue
            model = resolve_example_model(model_ref).model_profile
            recommendation = recommend_compute_for_model(
                config=config,
                inventory=inventory,
                model=model,
                workload=WorkloadProfile(workload_name="terminal-app", online=True),
            )
            output_fn("")
            output_fn(render_model_recommendation(recommendation))
            output_fn("")
            continue
        if choice == "3":
            compute = _choose_compute(inventory, input_fn=input_fn, output_fn=output_fn)
            if compute is None:
                continue
            models = [resolve_example_model(key).model_profile for key, _, _ in list_example_models()]
            report = recommend_models_for_compute(config=config, compute=compute, models=models)
            output_fn("")
            output_fn(render_compute_fit(report))
            output_fn("")
            continue
        if choice == "4":
            model_ref = _choose_model(input_fn=input_fn, output_fn=output_fn)
            if model_ref is None:
                continue
            model = resolve_example_model(model_ref).model_profile
            recommendation = recommend_compute_for_model(
                config=config,
                inventory=inventory,
                model=model,
                workload=WorkloadProfile(workload_name="terminal-app-deploy", online=True),
            )
            hint = build_deployment_hint(config, inventory, model, recommendation)
            output_fn("")
            output_fn(render_deployment_hint(hint))
            output_fn("")
            continue

        output_fn("Unknown choice. Enter 1, 2, 3, 4, or q.")
        output_fn("")


def _choose_model(*, input_fn: InputFn, output_fn: OutputFn) -> str | None:
    examples = list_example_models()
    output_fn("")
    output_fn("Choose a model:")
    for index, (key, label, repo_id) in enumerate(examples, start=1):
        output_fn(f"  {index}. {label} ({repo_id})")
    output_fn("  b. Back")
    choice = input_fn("model> ").strip().lower()
    if choice in {"b", "back", ""}:
        output_fn("")
        return None
    if choice.isdigit():
        index = int(choice)
        if 1 <= index <= len(examples):
            return examples[index - 1][0]
    return choice


def _choose_compute(
    inventory: WorkspaceInventorySnapshot,
    *,
    input_fn: InputFn,
    output_fn: OutputFn,
) -> WorkspaceComputeProfile | None:
    output_fn("")
    output_fn("Choose compute:")
    for index, compute in enumerate(inventory.compute, start=1):
        gpu = f"{compute.gpu_family or 'gpu'} x{compute.gpu_count}" if compute.gpu_count else "cpu-only"
        output_fn(f"  {index}. {compute.node_type_id} ({gpu})")
    output_fn("  b. Back")
    choice = input_fn("compute> ").strip()
    if choice.lower() in {"b", "back", ""}:
        output_fn("")
        return None
    if choice.isdigit():
        index = int(choice)
        if 1 <= index <= len(inventory.compute):
            return inventory.compute[index - 1]
    for compute in inventory.compute:
        if compute.node_type_id == choice:
            return compute
    output_fn(f"Compute '{choice}' is not in the current inventory.")
    output_fn("")
    return None
