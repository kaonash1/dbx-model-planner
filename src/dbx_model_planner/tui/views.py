"""TUI rendering using Rich -- minimal/monochrome theme with split-pane layout."""

from __future__ import annotations

import time

from rich.console import ConsoleRenderable
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..adapters.huggingface.catalog import CatalogEntry
from ..config import WORKLOAD_LABELS, WorkloadType
from ..domain import (
    CandidateCompute,
    FitLevel,
    ModelFamily,
    WorkspaceComputeProfile,
    WorkspaceInventorySnapshot,
)
from ..engines.fit import estimate_model_memory_gb, KvCacheQuant, MemoryEstimate
from ..engines.plan import QUANTIZATION_OPTIONS, CONTEXT_PRESETS
from .state import (
    FIT_FILTER_LABELS,
    FitFilter,
    InputMode,
    TuiState,
    View,
)


# -- Theme constants --------------------------------------------------------

# Single accent color: cyan.  Everything else is white/gray/dim.
ACCENT = "cyan"
ACCENT_BOLD = "bold cyan"
BORDER = "grey37"
BORDER_ACCENT = "cyan"
HEADER_BG = "grey11"
SELECTED_STYLE = f"bold {ACCENT} reverse"
DIM = "grey58"
MUTED = "grey70"
BRIGHT = "white"

FIT_COLORS = {
    FitLevel.SAFE: "green",
    FitLevel.BORDERLINE: "yellow",
    FitLevel.UNLIKELY: "red",
}


def _node_total_hourly(
    node_type_id: str,
    vm_pricing: dict[str, float],
    dbu_per_hour: float | None,
    dbu_rate_per_unit: float,
    discount_rate: float = 0.0,
    vat_rate: float = 0.0,
) -> float | None:
    """Compute final hourly cost for a node, or None if no VM price.

    Formula:
      list_price = VM_Price + DBU_Count × per_DBU_unit_rate
      discounted = list_price × (1 - discount_rate)
      final      = discounted × (1 + vat_rate)
    """
    vm_rate = vm_pricing.get(node_type_id)
    if vm_rate is None:
        return None
    dbu_cost = (dbu_per_hour or 0.0) * dbu_rate_per_unit
    list_price = vm_rate + dbu_cost
    discounted = list_price * (1.0 - discount_rate)
    return discounted * (1.0 + vat_rate)

# Spinner frames for loading indicator
_SPINNER = ["   ", ".  ", ".. ", "...", ".. ", ".  "]


# -- Top-level layout -------------------------------------------------------


def build_layout(state: TuiState, terminal_height: int = 40) -> ConsoleRenderable:
    """Build the full-screen Rich layout from current state."""

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=1),
        Layout(name="body"),
        Layout(name="footer", size=1),
    )

    layout["header"].update(_render_header(state))
    layout["footer"].update(_render_footer(state))

    if state.loading:
        layout["body"].update(_render_loading(state))
    elif state.view == View.INVENTORY:
        layout["body"].update(_render_inventory_layout(state, terminal_height - 2))
    elif state.view == View.MODEL_BROWSE:
        layout["body"].update(_render_browse_layout(state, terminal_height - 2))
    elif state.view == View.MODEL_INPUT:
        layout["body"].update(_render_model_input_view(state))
    elif state.view == View.MODEL_FIT:
        layout["body"].update(_render_model_fit_layout(state, terminal_height - 2))
    elif state.view == View.WHAT_IF:
        layout["body"].update(_render_whatif_view(state, terminal_height - 2))
    elif state.view == View.PRICING_SETUP:
        layout["body"].update(_render_pricing_setup_view(state))
    else:
        layout["body"].update(Text(""))

    return layout


# -- Header bar (single line, no panel border) ------------------------------


def _render_header(state: TuiState) -> Text:
    if state.inventory is None:
        return Text.from_markup(
            f"  [{ACCENT_BOLD}]dbx-model-planner[/{ACCENT_BOLD}]  [{DIM}]No inventory loaded[/{DIM}]"
        )

    inv = state.inventory
    gpu_count = len(state.gpu_nodes)
    cpu_count = len(state.cpu_nodes)
    ml_runtimes = sum(1 for r in inv.runtimes if r.ml_runtime)
    policies = len(inv.policies)

    url = inv.workspace_url or "unknown"
    if "azuredatabricks.net" in url:
        url = url.replace("https://", "")

    parts = [
        f"[{ACCENT_BOLD}]dbx-model-planner[/{ACCENT_BOLD}]",
        f"[{DIM}]{url}[/{DIM}]",
        f"[{BRIGHT}]{gpu_count}[/{BRIGHT}] [{DIM}]GPU[/{DIM}]",
        f"[{DIM}]{cpu_count} CPU[/{DIM}]",
        f"[{DIM}]{ml_runtimes} ML runtimes[/{DIM}]",
        f"[{DIM}]{policies} pol[/{DIM}]",
    ]
    if state.pricing_loaded:
        parts.append(f"[{DIM}]{state.pricing_node_count} priced[/{DIM}]")
    elif state.pricing_loading:
        parts.append(f"[{DIM}]pricing...[/{DIM}]")

    # Workload type indicator
    try:
        wt_label = WORKLOAD_LABELS[WorkloadType(state.workload_type)]
    except (ValueError, KeyError):
        wt_label = state.workload_type
    parts.append(f"[{DIM}]{wt_label}[/{DIM}]")

    return Text.from_markup("  " + f"  [{DIM}]|[/{DIM}]  ".join(parts))


# -- Footer / help bar (single line, no panel border) -----------------------


def _render_footer(state: TuiState) -> Text:
    if state.input_mode == InputMode.SEARCH:
        # Show browse search or inventory search depending on view
        query = state.browse_search if state.view == View.MODEL_BROWSE else state.search_query
        return Text.from_markup(
            f"  [{ACCENT_BOLD}]/[/{ACCENT_BOLD}] [{BRIGHT}]{query}[/{BRIGHT}]"
            f"[{ACCENT}]_[/{ACCENT}]"
            f"  [{DIM}]Enter: apply  Esc: cancel[/{DIM}]"
        )

    if state.input_mode == InputMode.MODEL_ID:
        return Text.from_markup(
            f"  [{ACCENT_BOLD}]model:[/{ACCENT_BOLD}] [{BRIGHT}]{state.input_buffer}[/{BRIGHT}]"
            f"[{ACCENT}]_[/{ACCENT}]"
            f"  [{DIM}]Enter: fetch  Esc: cancel[/{DIM}]"
        )

    if state.input_mode == InputMode.PRICING:
        step_labels = ["region", "discount %", "VAT %", "DBU rate"]
        step = state.pricing_setup_step
        return Text.from_markup(
            f"  [{ACCENT_BOLD}]{step_labels[step]}:[/{ACCENT_BOLD}] "
            f"[{BRIGHT}]{state.input_buffer}[/{BRIGHT}]"
            f"[{ACCENT}]_[/{ACCENT}]"
            f"  [{DIM}]Enter: next  Esc: cancel[/{DIM}]"
            f"  [{DIM}]step {step + 1}/4[/{DIM}]"
        )

    if state.status_message:
        return Text.from_markup(f"  [{DIM}]{state.status_message}[/{DIM}]")

    if state.view == View.INVENTORY:
        cpu_label = "CPU on" if state.show_cpu_nodes else "CPU off"
        price_hint = f"[{ACCENT}]$[/{ACCENT}] reconfigure prices  " if state.pricing_loaded or state.pricing_error else f"[{ACCENT}]$[/{ACCENT}] pricing setup  "
        try:
            wt_short = "AP" if WorkloadType(state.workload_type) == WorkloadType.ALL_PURPOSE else "JC"
        except ValueError:
            wt_short = "?"
        return Text.from_markup(
            f"  [{ACCENT}]j/k[/{ACCENT}] nav  "
            f"[{ACCENT}]PgUp/Dn[/{ACCENT}] page  "
            f"[{ACCENT}]/[/{ACCENT}] search  "
            f"[{ACCENT}]b[/{ACCENT}] browse models  "
            f"[{ACCENT}]m[/{ACCENT}] model ID  "
            f"[{ACCENT}]c[/{ACCENT}] [{MUTED}]{cpu_label}[/{MUTED}]  "
            f"[{ACCENT}]t[/{ACCENT}] [{MUTED}]{wt_short}[/{MUTED}]  "
            f"{price_hint}"
            f"[{ACCENT}]q[/{ACCENT}] quit"
        )

    if state.view == View.MODEL_BROWSE:
        cat = state.browse_category_filter or "All"
        discover_label = "loaded" if state.browse_discovered else "fetch"
        return Text.from_markup(
            f"  [{ACCENT}]j/k[/{ACCENT}] nav  "
            f"[{ACCENT}]PgUp/Dn[/{ACCENT}] page  "
            f"[{ACCENT}]/[/{ACCENT}] search  "
            f"[{ACCENT}]Enter[/{ACCENT}] fit model  "
            f"[{ACCENT}]Tab[/{ACCENT}] category:[{MUTED}]{cat}[/{MUTED}]  "
            f"[{ACCENT}]D[/{ACCENT}] discover:[{MUTED}]{discover_label}[/{MUTED}]  "
            f"[{ACCENT}]Esc[/{ACCENT}] back  "
            f"[{ACCENT}]q[/{ACCENT}] quit"
        )

    if state.view == View.MODEL_FIT:
        filter_label = FIT_FILTER_LABELS.get(state.fit_filter, "All")
        try:
            wt_short = "AP" if WorkloadType(state.workload_type) == WorkloadType.ALL_PURPOSE else "JC"
        except ValueError:
            wt_short = "?"
        return Text.from_markup(
            f"  [{ACCENT}]j/k[/{ACCENT}] nav  "
            f"[{ACCENT}]PgUp/Dn[/{ACCENT}] page  "
            f"[{ACCENT}]f[/{ACCENT}] filter:[{MUTED}]{filter_label}[/{MUTED}]  "
            f"[{ACCENT}]w[/{ACCENT}] what-if  "
            f"[{ACCENT}]t[/{ACCENT}] [{MUTED}]{wt_short}[/{MUTED}]  "
            f"[{ACCENT}]m[/{ACCENT}] new model  "
            f"[{ACCENT}]Esc[/{ACCENT}] back  "
            f"[{ACCENT}]q[/{ACCENT}] quit"
        )

    if state.view == View.WHAT_IF:
        selector = "quant" if state.whatif_selector_row == 0 else "context"
        try:
            wt_short = "AP" if WorkloadType(state.workload_type) == WorkloadType.ALL_PURPOSE else "JC"
        except ValueError:
            wt_short = "?"
        tq = f"[green]ON[/green]" if state.whatif_turboquant else "off"
        return Text.from_markup(
            f"  [{ACCENT}]←/→[/{ACCENT}] change {selector}  "
            f"[{ACCENT}]Tab[/{ACCENT}] toggle selector  "
            f"[{ACCENT}]K[/{ACCENT}] TurboQuant [{MUTED}]{tq}[/{MUTED}]  "
            f"[{ACCENT}]j/k[/{ACCENT}] nav  "
            f"[{ACCENT}]t[/{ACCENT}] [{MUTED}]{wt_short}[/{MUTED}]  "
            f"[{ACCENT}]Esc[/{ACCENT}] back  "
            f"[{ACCENT}]q[/{ACCENT}] quit"
        )

    if state.view == View.PRICING_SETUP:
        return Text.from_markup(
            f"  [{DIM}]Enter values for Azure pricing setup[/{DIM}]  "
            f"[{ACCENT}]Esc[/{ACCENT}] cancel"
        )

    return Text.from_markup(f"  [{ACCENT}]q[/{ACCENT}] quit")


# -- Loading spinner --------------------------------------------------------


def _render_loading(state: TuiState) -> Panel:
    frame_idx = int(time.time() * 4) % len(_SPINNER)
    spinner = _SPINNER[frame_idx]
    msg = state.status_message or "Loading"
    return Panel(
        Text.from_markup(
            f"\n\n  [{ACCENT}]{spinner}[/{ACCENT}] [{BRIGHT}]{msg}[/{BRIGHT}]\n"
        ),
        border_style=BORDER,
    )


# -- Inventory view with split-pane detail ----------------------------------


def _render_inventory_layout(state: TuiState, body_height: int) -> Layout:
    """Inventory table, optionally split with a detail sidebar on the right."""

    node = state.selected_node()

    # If a node is selected, show split pane
    if node is not None and state.view == View.INVENTORY:
        layout = Layout()
        layout.split_row(
            Layout(name="table", ratio=3),
            Layout(name="detail", ratio=2),
        )
        layout["table"].update(_render_inventory_table(state, body_height))
        layout["detail"].update(_render_node_sidebar(node, state))
        return layout

    return Layout(_render_inventory_table(state, body_height))


def _render_inventory_table(state: TuiState, max_rows: int) -> Panel:
    nodes = state.displayed_nodes
    if not nodes:
        if state.search_query:
            return Panel(
                Text.from_markup(
                    f"\n  [{DIM}]No nodes match[/{DIM}] [{BRIGHT}]'{state.search_query}'[/{BRIGHT}]"
                ),
                border_style=BORDER,
            )
        label = "GPU+CPU" if state.show_cpu_nodes else "GPU"
        return Panel(
            Text.from_markup(f"\n  [{DIM}]No {label} nodes in workspace[/{DIM}]"),
            border_style=BORDER,
        )

    table = Table(
        expand=True,
        show_edge=False,
        pad_edge=True,
        show_header=True,
        header_style=MUTED,
        row_styles=[""],  # Uniform, no alternating
    )

    # Columns
    table.add_column("Node Type", min_width=26, style=BRIGHT)
    table.add_column("GPUs", justify="center", width=5, style=MUTED)
    table.add_column("Family", width=10, style=MUTED)
    table.add_column("GPU Mem", justify="right", width=9, style=MUTED)
    table.add_column("vCPU", justify="right", width=5, style=MUTED)
    table.add_column("RAM", justify="right", width=7, style=DIM)
    table.add_column("DBUs", justify="right", width=7, style=DIM)
    if state.pricing_loaded:
        table.add_column("$/hr", justify="right", width=7, style=MUTED)

    # Determine visible window
    state.scroll_offset, start, end = state.compute_scroll_window(
        state.selected_index, state.scroll_offset, len(nodes), max_rows,
    )
    state.visible_rows = max(max_rows - 4, 5)

    for i in range(start, end):
        node = nodes[i]
        is_selected = i == state.selected_index

        total_gpu_mem = (node.gpu_memory_gb or 0.0) * max(node.gpu_count, 1) if node.gpu_memory_gb else 0.0
        gpu_mem = f"{total_gpu_mem:.0f} GB" if total_gpu_mem else "-"
        ram = f"{node.memory_gb:.0f} GB" if node.memory_gb else "-"
        vcpu = str(node.vcpu_count) if node.vcpu_count else "-"
        gpu_family = node.gpu_family or ("-" if node.gpu_count == 0 else "?")

        if is_selected:
            style = SELECTED_STYLE
        else:
            style = ""

        name_prefix = ">" if is_selected else " "

        row = [
            f"{name_prefix} {node.node_type_id}",
            str(node.gpu_count),
            gpu_family,
            gpu_mem,
            vcpu,
            ram,
            f"{node.dbu_per_hour:.1f}" if node.dbu_per_hour else "-",
        ]
        if state.pricing_loaded:
            total = _node_total_hourly(
                node.node_type_id, state.vm_pricing,
                node.dbu_per_hour, state.dbu_rate_per_unit,
                state.pricing_discount, state.pricing_vat,
            )
            row.append(f"{total:.2f}" if total is not None else "-")

        table.add_row(*row, style=style)

    # Title with scroll position
    total = len(nodes)
    pos = f"{start + 1}-{end} of {total}"
    label = "GPU + CPU" if state.show_cpu_nodes else "GPU"
    title = f"[{DIM}]{label} Compute[/{DIM}]"
    if state.search_query:
        title += f"  [{DIM}]filter: {state.search_query}[/{DIM}]"
    if state.pricing_loaded:
        title += f"  [{DIM}]{state.pricing_node_count} priced ({state.pricing_region})[/{DIM}]"

    return Panel(
        table,
        title=title,
        subtitle=f"[{DIM}]{pos}[/{DIM}]",
        title_align="left",
        subtitle_align="right",
        border_style=BORDER,
    )


# -- Node detail sidebar (right pane) --------------------------------------


def _render_node_sidebar(node: WorkspaceComputeProfile, state: TuiState) -> Panel:
    """Compact node detail for the split-pane sidebar."""
    lines: list[str] = []

    lines.append(f"[{ACCENT_BOLD}]{node.node_type_id}[/{ACCENT_BOLD}]")
    lines.append("")

    # Hardware
    lines.append(f"  [{MUTED}]GPU count[/{MUTED}]     [{BRIGHT}]{node.gpu_count}[/{BRIGHT}]")
    lines.append(f"  [{MUTED}]GPU family[/{MUTED}]    [{BRIGHT}]{node.gpu_family or 'none'}[/{BRIGHT}]")
    gpu_mem = f"{node.gpu_memory_gb:.0f} GB" if node.gpu_memory_gb else "-"
    total_gpu_mem = ""
    if node.gpu_memory_gb and node.gpu_count > 1:
        total = node.gpu_memory_gb * node.gpu_count
        total_gpu_mem = f"  [{DIM}]({total:.0f} GB total)[/{DIM}]"
    lines.append(f"  [{MUTED}]GPU memory[/{MUTED}]    [{BRIGHT}]{gpu_mem}[/{BRIGHT}]{total_gpu_mem}")
    lines.append(f"  [{MUTED}]vCPU[/{MUTED}]          [{BRIGHT}]{node.vcpu_count or '-'}[/{BRIGHT}]")
    ram = f"{node.memory_gb:.0f} GB" if node.memory_gb else "-"
    lines.append(f"  [{MUTED}]System RAM[/{MUTED}]    [{BRIGHT}]{ram}[/{BRIGHT}]")

    # Pricing section
    if state.pricing_loaded or node.dbu_per_hour:
        lines.append("")
        try:
            wt_label = WORKLOAD_LABELS[WorkloadType(state.workload_type)]
        except (ValueError, KeyError):
            wt_label = state.workload_type
        lines.append(
            f"  [{ACCENT}]Pricing[/{ACCENT}]  [{DIM}]({state.pricing_region or 'no region'}"
            f" | {wt_label})[/{DIM}]"
        )
        vm_rate: float | None = None
        if state.pricing_loaded:
            vm_rate = state.vm_pricing.get(node.node_type_id)
            if vm_rate is not None:
                lines.append(f"  [{MUTED}]VM cost[/{MUTED}]       [{BRIGHT}]${vm_rate:.2f}/hr[/{BRIGHT}]")
            elif node.gpu_count > 0:
                lines.append(f"  [{DIM}]No VM price found[/{DIM}]")
        if node.dbu_per_hour:
            dbu_cost = node.dbu_per_hour * state.dbu_rate_per_unit
            lines.append(
                f"  [{MUTED}]DBU Count[/{MUTED}]     [{BRIGHT}]{node.dbu_per_hour:.1f}[/{BRIGHT}]"
            )
            lines.append(
                f"  [{MUTED}]DBU Price[/{MUTED}]     [{BRIGHT}]${dbu_cost:.2f}/hr[/{BRIGHT}]"
                f"  [{DIM}]({node.dbu_per_hour:.1f} x ${state.dbu_rate_per_unit:.2f})[/{DIM}]"
            )
        if state.pricing_loaded and vm_rate is not None:
            dbu_cost = (node.dbu_per_hour or 0.0) * state.dbu_rate_per_unit
            total = vm_rate + dbu_cost
            discount = state.pricing_discount
            vat = state.pricing_vat
            discounted = total * (1.0 - discount)
            with_vat = discounted * (1.0 + vat)
            lines.append(f"  [{MUTED}]Total[/{MUTED}]         [{BRIGHT}]${total:.2f}/hr[/{BRIGHT}]")
            lines.append(f"  [{MUTED}]Total*[/{MUTED}]        [{BRIGHT}]${with_vat:.2f}/hr[/{BRIGHT}]")
            lines.append(f"  [{DIM}]* discount and VAT applied[/{DIM}]")

    # Runtimes section
    if state.inventory and state.inventory.runtimes:
        lines.append("")
        lines.append(f"  [{ACCENT}]Runtimes[/{ACCENT}]")
        if node.runtime_ids:
            # Show ML runtimes that match this node
            ml_rts = [
                r for r in state.inventory.runtimes
                if r.ml_runtime and r.runtime_id in node.runtime_ids
            ]
            for rt in ml_rts[:5]:
                cuda = f" cuda:{rt.cuda_version}" if rt.cuda_version else ""
                lines.append(f"    [{DIM}]{rt.runtime_id}{cuda}[/{DIM}]")
            remaining = len(node.runtime_ids) - len(ml_rts[:5])
            if remaining > 0:
                lines.append(f"    [{DIM}]+{remaining} more[/{DIM}]")
        else:
            lines.append(f"    [{DIM}]all runtimes[/{DIM}]")

    # Policies section
    if state.inventory and state.inventory.policies:
        matching_policies = [
            p for p in state.inventory.policies
            if not p.allowed_node_types or node.node_type_id in p.allowed_node_types
        ]
        if matching_policies:
            lines.append("")
            lines.append(f"  [{ACCENT}]Policies[/{ACCENT}]")
            for pol in matching_policies[:4]:
                lines.append(f"    [{DIM}]{pol.policy_name}[/{DIM}] [{DIM}]({pol.policy_id})[/{DIM}]")
            if len(matching_policies) > 4:
                lines.append(f"    [{DIM}]+{len(matching_policies) - 4} more[/{DIM}]")

    # Availability notes
    if node.availability_notes:
        lines.append("")
        lines.append(f"  [{ACCENT}]Notes[/{ACCENT}]")
        for note in node.availability_notes[:3]:
            lines.append(f"    [{DIM}]{note}[/{DIM}]")

    return Panel(
        Text.from_markup("\n".join(lines)),
        title=f"[{DIM}]Detail[/{DIM}]",
        title_align="left",
        border_style=BORDER,
    )


# -- Model browse view with split-pane detail -------------------------------


_CATEGORY_COLORS = {
    "LLM": BRIGHT,
    "Embedding": "magenta",
    "VLM": "yellow",
    "Code": "green",
}


def _render_browse_layout(state: TuiState, body_height: int) -> Layout:
    """Model catalog table, with entry detail in a right sidebar."""
    entry = state.selected_browse_entry()

    if entry is not None:
        layout = Layout()
        layout.split_row(
            Layout(name="table", ratio=3),
            Layout(name="detail", ratio=2),
        )
        layout["table"].update(_render_browse_table(state, body_height))
        layout["detail"].update(_render_browse_sidebar(entry))
        return layout

    return Layout(_render_browse_table(state, body_height))


def _render_browse_table(state: TuiState, max_rows: int) -> Panel:
    entries = state.browse_displayed
    if not entries:
        msg = "No models match filter" if state.browse_search or state.browse_category_filter else "No models in catalog"
        return Panel(
            Text.from_markup(f"\n  [{DIM}]{msg}[/{DIM}]"),
            border_style=BORDER,
        )

    table = Table(
        expand=True,
        show_edge=False,
        pad_edge=True,
        show_header=True,
        header_style=MUTED,
    )
    table.add_column("Model", min_width=36, style=BRIGHT, no_wrap=True, overflow="ellipsis")
    table.add_column("Params", justify="right", width=8, style=MUTED)
    table.add_column("Type", width=10)
    table.add_column("Provider", width=10, style=DIM)
    table.add_column("Use Case", min_width=20, style=DIM, no_wrap=True, overflow="ellipsis")

    # Visible window
    state.browse_scroll_offset, start, end = state.compute_scroll_window(
        state.browse_selected_index, state.browse_scroll_offset, len(entries), max_rows,
    )
    state.visible_rows = max(max_rows - 4, 5)

    for i in range(start, end):
        e = entries[i]
        is_selected = i == state.browse_selected_index
        style = SELECTED_STYLE if is_selected else ""
        prefix = ">" if is_selected else " "

        cat_color = _CATEGORY_COLORS.get(e.category, DIM)
        cat_text = f"[{cat_color}]{e.category}[/{cat_color}]"

        # Mark discovered (trending) entries
        name = e.model_id
        if e.discovered:
            name = f"{name} [{DIM}]*[/{DIM}]"

        table.add_row(
            f"{prefix} {name}",
            e.params_label,
            cat_text,
            e.provider,
            e.use_case,
            style=style,
        )

    total = len(entries)
    pos = f"{start + 1}-{end} of {total}"
    curated_count = sum(1 for e in entries if not e.discovered)
    discovered_count = sum(1 for e in entries if e.discovered)

    title = f"[{DIM}]Model Catalog[/{DIM}]"
    if state.browse_search:
        title += f"  [{DIM}]filter: {state.browse_search}[/{DIM}]"

    subtitle_parts = [f"[{DIM}]{curated_count} curated[/{DIM}]"]
    if discovered_count:
        subtitle_parts.append(f"[{DIM}]{discovered_count} trending[/{DIM}]")
    subtitle_parts.append(f"[{DIM}]{pos}[/{DIM}]")

    return Panel(
        table,
        title=title,
        subtitle="  ".join(subtitle_parts),
        title_align="left",
        subtitle_align="right",
        border_style=BORDER,
    )


def _render_browse_sidebar(entry: CatalogEntry) -> Panel:
    """Detail sidebar for a catalog entry."""
    lines: list[str] = []

    lines.append(f"[{ACCENT_BOLD}]{entry.model_id}[/{ACCENT_BOLD}]")
    lines.append("")

    cat_color = _CATEGORY_COLORS.get(entry.category, DIM)
    lines.append(f"  [{MUTED}]Type[/{MUTED}]          [{cat_color}]{entry.category}[/{cat_color}]")
    lines.append(f"  [{MUTED}]Provider[/{MUTED}]      [{BRIGHT}]{entry.provider}[/{BRIGHT}]")
    lines.append(f"  [{MUTED}]Parameters[/{MUTED}]    [{BRIGHT}]{entry.params_label}[/{BRIGHT}]")
    if entry.context_length:
        lines.append(f"  [{MUTED}]Context[/{MUTED}]       [{BRIGHT}]{entry.context_length:,}[/{BRIGHT}]")
    lines.append(f"  [{MUTED}]Use case[/{MUTED}]      [{BRIGHT}]{entry.use_case}[/{BRIGHT}]")

    if entry.downloads:
        dl = entry.downloads
        if dl >= 1_000_000:
            dl_str = f"{dl / 1_000_000:.1f}M"
        elif dl >= 1_000:
            dl_str = f"{dl / 1_000:.0f}K"
        else:
            dl_str = str(dl)
        lines.append(f"  [{MUTED}]Downloads[/{MUTED}]     [{DIM}]{dl_str}[/{DIM}]")

    if entry.gated:
        lines.append("")
        lines.append(f"  [{DIM}]Gated repo - HF token required[/{DIM}]")

    if entry.discovered:
        lines.append("")
        lines.append(f"  [{DIM}]* Discovered from HF trending[/{DIM}]")

    lines.append("")
    lines.append(f"  [{ACCENT}]Press Enter to run fit analysis[/{ACCENT}]")
    lines.append(f"  [{DIM}]against your workspace inventory[/{DIM}]")

    return Panel(
        Text.from_markup("\n".join(lines)),
        title=f"[{DIM}]Model Info[/{DIM}]",
        title_align="left",
        border_style=BORDER,
    )


# -- Model input view -------------------------------------------------------


def _render_model_input_view(state: TuiState) -> Panel:
    lines: list[str] = [
        "",
        f"  [{BRIGHT}]Enter a HuggingFace model ID:[/{BRIGHT}]",
        "",
    ]

    # Show model history if available
    if state.model_history:
        lines.append(f"  [{MUTED}]Recent:[/{MUTED}]")
        for h in state.model_history[:5]:
            lines.append(f"    [{DIM}]{h}[/{DIM}]")
        lines.append("")

    lines.append(f"  [{ACCENT}]>[/{ACCENT}] [{BRIGHT}]{state.input_buffer}[/{BRIGHT}][{ACCENT}]_[/{ACCENT}]")
    lines.append("")

    if state.status_message:
        lines.append(f"  [{DIM}]{state.status_message}[/{DIM}]")

    return Panel(
        Text.from_markup("\n".join(lines)),
        title=f"[{DIM}]Model Fit[/{DIM}]",
        title_align="left",
        border_style=BORDER,
    )


# -- Model fit view with split-pane detail ----------------------------------


def _render_model_fit_layout(state: TuiState, body_height: int) -> Layout:
    """Model fit table, with candidate detail in a right sidebar."""

    candidate = state.selected_candidate()

    if candidate is not None:
        layout = Layout()
        layout.split_row(
            Layout(name="table", ratio=3),
            Layout(name="detail", ratio=1),
        )
        layout["table"].update(_render_model_fit_table(state, body_height))
        layout["detail"].update(_render_candidate_sidebar(candidate, state))
        return layout

    return Layout(_render_model_fit_table(state, body_height))


def _render_model_fit_table(state: TuiState, max_rows: int) -> Panel:

    if state.model_recommendation is None or state.model_profile is None:
        return Panel(
            Text.from_markup(f"  [{DIM}]No model loaded[/{DIM}]"),
            border_style=BORDER,
        )

    model = state.model_profile
    rec = state.model_recommendation
    all_candidates = rec.candidates
    candidates = state.fit_displayed_candidates

    # Model summary line
    param_str = f"{model.parameter_count / 1e9:.1f}B" if model.parameter_count else "?"
    if model.context_length:
        ctx_str = f"{model.context_length:,}"
    else:
        ctx_str = "?"

    safe = sum(1 for c in all_candidates if c.fit_level == FitLevel.SAFE)
    border = sum(1 for c in all_candidates if c.fit_level == FitLevel.BORDERLINE)
    unlikely = sum(1 for c in all_candidates if c.fit_level == FitLevel.UNLIKELY)

    table = Table(
        expand=True,
        show_edge=False,
        pad_edge=True,
        show_header=True,
        header_style=MUTED,
    )
    table.add_column("Node Type", min_width=24, style=BRIGHT)
    table.add_column("Fit", width=11)
    table.add_column("Headroom", justify="right", width=9, style=MUTED)
    table.add_column("GPU Mem", justify="right", width=8, style=DIM)
    table.add_column("GPUs", justify="center", width=5, style=DIM)
    table.add_column("Mem%", justify="right", width=5, style=MUTED)
    table.add_column("Est. tok/s", justify="right", width=10, style=DIM)
    if state.pricing_loaded:
        table.add_column("$/hr", justify="right", width=7, style=MUTED)

    if not candidates:
        filter_label = FIT_FILTER_LABELS.get(state.fit_filter, "")
        title_parts = [
            f"[{ACCENT_BOLD}]{model.model_id}[/{ACCENT_BOLD}]",
            f"[{DIM}]{param_str}[/{DIM}]",
            f"[{DIM}]ctx {ctx_str}[/{DIM}]",
        ]
        if state.fit_filter != FitFilter.ALL:
            title_parts.append(f"[{ACCENT}]filter: {filter_label}[/{ACCENT}]")
        return Panel(
            Text.from_markup(f"\n  [{DIM}]No candidates match filter '{filter_label}'[/{DIM}]"),
            title="  ".join(title_parts),
            title_align="left",
            border_style=BORDER,
        )

    visible_rows = max(max_rows - 6, 5)
    state.fit_scroll_offset, start, end = state.compute_scroll_window(
        state.fit_selected_index, state.fit_scroll_offset, len(candidates), max_rows,
    )

    for i in range(start, end):
        c = candidates[i]
        is_selected = i == state.fit_selected_index

        fit_color = FIT_COLORS.get(c.fit_level, MUTED)
        fit_text = f"[{fit_color}]{c.fit_level.value}[/{fit_color}]"
        headroom = f"{c.estimated_headroom_gb:+.1f} GB" if c.estimated_headroom_gb is not None else "-"
        total_gpu_mem = (c.compute.gpu_memory_gb or 0.0) * max(c.compute.gpu_count, 1) if c.compute.gpu_memory_gb else 0.0
        gpu_mem = f"{total_gpu_mem:.0f} GB" if total_gpu_mem else "-"

        style = SELECTED_STYLE if is_selected else ""
        prefix = ">" if is_selected else " "

        row = [
            f"{prefix} {c.compute.node_type_id}",
        ]

        row += [
            fit_text,
            headroom,
            gpu_mem,
            str(c.compute.gpu_count),
        ]

        # Memory usage %
        if total_gpu_mem > 0 and c.estimated_memory_gb and not c.estimate_incomplete:
            mem_pct = (c.estimated_memory_gb / total_gpu_mem) * 100
            if mem_pct < 70:
                pct_color = "green"
            elif mem_pct <= 85:
                pct_color = "yellow"
            else:
                pct_color = "red"
            row.append(f"[{pct_color}]{mem_pct:.0f}%[/{pct_color}]")
        else:
            row.append("-")

        # Estimated tok/s
        if c.estimated_tok_s is not None:
            row.append(f"~{c.estimated_tok_s:,.0f}")
        else:
            row.append("-")

        if state.pricing_loaded:
            if c.cost and c.cost.vat_adjusted_hourly_rate is not None:
                row.append(f"{c.cost.vat_adjusted_hourly_rate:.2f}")
            else:
                # Fallback: compute from VM price with discount+VAT
                total = _node_total_hourly(
                    c.compute.node_type_id, state.vm_pricing,
                    c.compute.dbu_per_hour, state.dbu_rate_per_unit,
                    state.pricing_discount, state.pricing_vat,
                )
                row.append(f"{total:.2f}" if total is not None else "-")

        table.add_row(*row, style=style)

    # Scroll position
    total = len(candidates)
    pos = f"{start + 1}-{end} of {total}" if total > 0 else "0"

    title_parts = [
        f"[{ACCENT_BOLD}]{model.model_id}[/{ACCENT_BOLD}]",
        f"[{DIM}]{param_str}[/{DIM}]",
        f"[{DIM}]ctx {ctx_str}[/{DIM}]",
    ]
    if state.fit_filter != FitFilter.ALL:
        filter_label = FIT_FILTER_LABELS.get(state.fit_filter, "")
        fit_color = FIT_COLORS.get(FitLevel(state.fit_filter.value), ACCENT)
        title_parts.append(f"[{fit_color}]filter: {filter_label}[/{fit_color}]")
    subtitle_parts = [
        f"[green]{safe}ok[/green]",
        f"[yellow]{border}maybe[/yellow]",
        f"[red]{unlikely}no[/red]",
        f"[{DIM}]{pos}[/{DIM}]",
    ]

    return Panel(
        table,
        title="  ".join(title_parts),
        subtitle="  ".join(subtitle_parts),
        title_align="left",
        subtitle_align="right",
        border_style=BORDER,
    )


# -- Candidate detail sidebar (right pane in model fit) ---------------------


def _render_candidate_sidebar(candidate: CandidateCompute, state: TuiState) -> Panel:
    """Compact candidate detail for the split-pane sidebar."""
    c = candidate
    fit_color = FIT_COLORS.get(c.fit_level, MUTED)

    lines: list[str] = []
    lines.append(f"[{ACCENT_BOLD}]{c.compute.node_type_id}[/{ACCENT_BOLD}]")
    lines.append("")

    # Fit info
    lines.append(f"  [{MUTED}]Fit[/{MUTED}]           [{fit_color}]{c.fit_level.value}[/{fit_color}]")
    lines.append(f"  [{MUTED}]Risk[/{MUTED}]          [{BRIGHT}]{c.risk_level.value}[/{BRIGHT}]")
    lines.append(f"  [{MUTED}]Precision[/{MUTED}]     [{BRIGHT}]{c.recommended_quantization or 'fp16'}[/{BRIGHT}]")

    est = "? (insufficient metadata)" if c.estimate_incomplete else (f"{c.estimated_memory_gb:.1f} GB" if c.estimated_memory_gb else "-")
    lines.append(f"  [{MUTED}]Est. memory[/{MUTED}]   [{BRIGHT}]{est}[/{BRIGHT}]")

    headroom = f"{c.estimated_headroom_gb:+.1f} GB" if c.estimated_headroom_gb is not None else "-"
    lines.append(f"  [{MUTED}]Headroom[/{MUTED}]      [{BRIGHT}]{headroom}[/{BRIGHT}]")

    # Memory usage percentage
    total_gpu_mem_sidebar = (c.compute.gpu_memory_gb or 0.0) * max(c.compute.gpu_count, 1) if c.compute.gpu_memory_gb else 0.0
    if total_gpu_mem_sidebar > 0 and c.estimated_memory_gb and not c.estimate_incomplete:
        usage_pct = (c.estimated_memory_gb / total_gpu_mem_sidebar) * 100
        if usage_pct < 70:
            pct_color = "green"
        elif usage_pct <= 85:
            pct_color = "yellow"
        else:
            pct_color = "red"
        lines.append(f"  [{MUTED}]Mem Usage[/{MUTED}]     [{pct_color}]{usage_pct:.0f}%[/{pct_color}]")

    # Estimated tok/s
    if c.estimated_tok_s is not None:
        lines.append(f"  [{MUTED}]Est. tok/s[/{MUTED}]    [{BRIGHT}]~{c.estimated_tok_s:,.0f}[/{BRIGHT}]")

    # Context length from model profile
    if state.model_profile:
        native_ctx = state.model_profile.context_length
        if native_ctx:
            lines.append(f"  [{MUTED}]Context[/{MUTED}]       [{BRIGHT}]{native_ctx:,}[/{BRIGHT}]")

    if state.model_gated:
        lines.append("")
        lines.append(f"  [{DIM}]Gated repo - HF token required[/{DIM}]")

    # Hardware
    lines.append("")
    lines.append(f"  [{ACCENT}]Hardware[/{ACCENT}]")
    lines.append(f"  [{MUTED}]GPU count[/{MUTED}]     [{BRIGHT}]{c.compute.gpu_count}[/{BRIGHT}]")
    lines.append(f"  [{MUTED}]GPU family[/{MUTED}]    [{BRIGHT}]{c.compute.gpu_family or '-'}[/{BRIGHT}]")
    gpu_mem = f"{c.compute.gpu_memory_gb:.0f} GB" if c.compute.gpu_memory_gb else "-"
    total_gpu_mem = ""
    if c.compute.gpu_memory_gb and c.compute.gpu_count > 1:
        total = c.compute.gpu_memory_gb * c.compute.gpu_count
        total_gpu_mem = f"  [{DIM}]({total:.0f} GB total)[/{DIM}]"
    lines.append(f"  [{MUTED}]GPU memory[/{MUTED}]    [{BRIGHT}]{gpu_mem}[/{BRIGHT}]{total_gpu_mem}")

    # Cost
    if c.cost is not None:
        lines.append("")
        lines.append(f"  [{ACCENT}]Cost[/{ACCENT}]")
        if c.cost.vm_hourly_rate is not None:
            lines.append(f"  [{MUTED}]VM cost[/{MUTED}]       [{BRIGHT}]${c.cost.vm_hourly_rate:.2f}/hr[/{BRIGHT}]")
        if c.cost.dbu_hourly_rate is not None and c.cost.dbu_hourly_rate > 0:
            dbu_count = c.compute.dbu_per_hour or 0.0
            dbu_price = c.cost.dbu_hourly_rate / dbu_count if dbu_count > 0 else 0.0
            lines.append(
                f"  [{MUTED}]DBU Price[/{MUTED}]     [{BRIGHT}]${c.cost.dbu_hourly_rate:.2f}/hr[/{BRIGHT}]"
                f"  [{DIM}]({dbu_count:.1f} x ${dbu_price:.2f})[/{DIM}]"
            )
        if c.cost.estimated_hourly_rate is not None:
            lines.append(f"  [{MUTED}]Total[/{MUTED}]         [{BRIGHT}]${c.cost.estimated_hourly_rate:.2f}/hr[/{BRIGHT}]")
        if c.cost.vat_adjusted_hourly_rate is not None:
            lines.append(f"  [{MUTED}]Total*[/{MUTED}]        [{BRIGHT}]${c.cost.vat_adjusted_hourly_rate:.2f}/hr[/{BRIGHT}]")
            lines.append(f"  [{DIM}]* discount and VAT applied[/{DIM}]")
    elif c.compute.dbu_per_hour is not None:
        lines.append("")
        lines.append(f"  [{ACCENT}]DBU[/{ACCENT}]")
        lines.append(f"  [{MUTED}]DBU count[/{MUTED}]     [{BRIGHT}]{c.compute.dbu_per_hour:.1f}[/{BRIGHT}]")

    # Notes
    if c.notes:
        lines.append("")
        lines.append(f"  [{ACCENT}]Notes[/{ACCENT}]")
        for note in c.notes[:4]:
            lines.append(f"    [{DIM}]{note}[/{DIM}]")

    # What-if hint
    if state.model_profile is not None:
        lines.append("")
        lines.append(f"  [{DIM}]Press[/{DIM}] [{ACCENT}]w[/{ACCENT}] [{DIM}]for what-if analysis[/{DIM}]")

    return Panel(
        Text.from_markup("\n".join(lines)),
        title=f"[{DIM}]Fit Detail[/{DIM}]",
        title_align="left",
        border_style=BORDER,
    )


# -- Pricing setup wizard view -----------------------------------------------


_PRICING_STEP_LABELS = [
    "Azure region",
    "Discount rate (%)",
    "VAT rate (%)",
    "DBU rate (per DBU)",
]
_PRICING_STEP_HINTS = [
    "e.g. westeurope, eastus, westus2",
    "Enterprise discount as percentage (e.g. 37)",
    "Value-added tax as percentage (e.g. 19)",
    "Price per DBU in USD (auto-fetched from Azure API)",
]
_PRICING_STEP_KEYS = ["region", "discount", "vat", "dbu_rate"]


def _render_pricing_setup_view(state: TuiState) -> Panel:
    """Render the multi-step pricing setup wizard."""
    lines: list[str] = [
        "",
        f"  [{ACCENT_BOLD}]Pricing Setup[/{ACCENT_BOLD}]",
        "",
    ]

    step = state.pricing_setup_step

    # Show completed steps
    for i in range(step):
        key = _PRICING_STEP_KEYS[i]
        value = state.pricing_setup_values.get(key, "")
        lines.append(
            f"  [{DIM}]{_PRICING_STEP_LABELS[i]}:[/{DIM}]  "
            f"[{BRIGHT}]{value}[/{BRIGHT}]  [{DIM}]OK[/{DIM}]"
        )

    # Current step
    lines.append("")
    lines.append(
        f"  [{BRIGHT}]{_PRICING_STEP_LABELS[step]}:[/{BRIGHT}]"
    )
    lines.append(
        f"  [{DIM}]{_PRICING_STEP_HINTS[step]}[/{DIM}]"
    )
    lines.append("")
    lines.append(
        f"  [{ACCENT}]>[/{ACCENT}] [{BRIGHT}]{state.input_buffer}[/{BRIGHT}]"
        f"[{ACCENT}]_[/{ACCENT}]"
    )

    # Remaining steps
    if step < 3:
        lines.append("")
        for i in range(step + 1, 4):
            key = _PRICING_STEP_KEYS[i]
            default = state.pricing_setup_values.get(key, "")
            lines.append(
                f"  [{DIM}]{_PRICING_STEP_LABELS[i]}:[/{DIM}]  "
                f"[{DIM}]default: {default}[/{DIM}]"
            )

    if state.status_message:
        lines.append("")
        lines.append(f"  [{DIM}]{state.status_message}[/{DIM}]")

    lines.append("")
    lines.append(
        f"  [{DIM}]After completing all steps, prices will be fetched[/{DIM}]"
    )
    lines.append(
        f"  [{DIM}]from the Azure Retail Prices API.[/{DIM}]"
    )

    return Panel(
        Text.from_markup("\n".join(lines)),
        title=f"[{DIM}]Pricing Setup[/{DIM}]",
        title_align="left",
        border_style=BORDER,
    )


# -- What-if view -----------------------------------------------------------


def _render_whatif_view(state: TuiState, body_height: int) -> Panel:
    """Render the dedicated what-if analysis view.

    Layout:
      - Model header line (name, params)
      - Horizontal quant selector bar
      - Horizontal context selector bar
      - Recomputed fit table for the selected quant/context
    """
    if state.model_profile is None or state.inventory is None:
        return Panel(
            Text.from_markup(f"  [{DIM}]No model loaded[/{DIM}]"),
            border_style=BORDER,
        )

    model = state.model_profile
    param_str = f"{model.parameter_count / 1e9:.1f}B" if model.parameter_count else "?"
    native_ctx = model.context_length

    # Current selections
    selected_quant = QUANTIZATION_OPTIONS[state.whatif_quant_index]
    if state.whatif_ctx_index == 0:
        selected_ctx = native_ctx or 4096
        ctx_label = "default"
    else:
        selected_ctx = CONTEXT_PRESETS[state.whatif_ctx_index - 1]
        ctx_label = f"{selected_ctx:,}"

    lines: list[str] = []

    # -- Model header
    ctx_suffix = f"  [{DIM}]native ctx: {native_ctx:,}[/{DIM}]" if native_ctx else ""
    lines.append(
        f"  [{ACCENT_BOLD}]{model.model_id}[/{ACCENT_BOLD}]  "
        f"[{DIM}]{param_str}[/{DIM}]  "
        f"[{DIM}]{model.family.value}[/{DIM}]"
        f"{ctx_suffix}"
    )
    lines.append("")

    # -- VLM note
    is_vlm = model.family == ModelFamily.VLM
    if is_vlm:
        lines.append(
            f"  [italic dim]Note: Vision encoder stays at fp16. "
            f"Quantization applies to language backbone only.[/italic dim]"
        )
        lines.append("")

    # -- Quant selector bar
    quant_label = "Lang Quant:" if is_vlm else "Quant:"
    quant_pad = "  " if is_vlm else "   "
    quant_active = state.whatif_selector_row == 0
    quant_indicator = f"[{ACCENT}]>[/{ACCENT}]" if quant_active else f"[{DIM}] [/{DIM}]"
    quant_parts: list[str] = []
    for i, q in enumerate(QUANTIZATION_OPTIONS):
        if i == state.whatif_quant_index:
            quant_parts.append(f"[{ACCENT_BOLD}]\\[{q}][/{ACCENT_BOLD}]")
        else:
            quant_parts.append(f"[{MUTED}]{q}[/{MUTED}]")
    lines.append(
        f"  {quant_indicator} [{DIM}]{quant_label}[/{DIM}]{quant_pad}"
        + "  ".join(quant_parts)
    )

    # -- Context selector bar
    ctx_active = state.whatif_selector_row == 1
    ctx_indicator = f"[{ACCENT}]>[/{ACCENT}]" if ctx_active else f"[{DIM}] [/{DIM}]"
    ctx_options: list[str] = ["default"] + [str(p) for p in CONTEXT_PRESETS]
    ctx_parts: list[str] = []
    for i, label in enumerate(ctx_options):
        if i == state.whatif_ctx_index:
            ctx_parts.append(f"[{ACCENT_BOLD}]\\[{label}][/{ACCENT_BOLD}]")
        else:
            ctx_parts.append(f"[{MUTED}]{label}[/{MUTED}]")
    lines.append(
        f"  {ctx_indicator} [{DIM}]Context:[/{DIM}] "
        + "  ".join(ctx_parts)
    )
    lines.append("")

    # -- TurboQuant KV cache toggle
    kv_quant = KvCacheQuant.TURBOQUANT if state.whatif_turboquant else KvCacheQuant.FP16
    if state.whatif_turboquant:
        tq_badge = f"[{ACCENT_BOLD}]\\[TurboQuant ON][/{ACCENT_BOLD}]"
        tq_note = f"  [{DIM}]KV cache ~{kv_quant.compression_ratio:.1f}× compressed (3b keys + 2b values)[/{DIM}]"
    else:
        tq_badge = f"[{DIM}]\\[TurboQuant off][/{DIM}]"
        tq_note = ""
    lines.append(f"  [{DIM}]KV Cache:[/{DIM}]  {tq_badge}{tq_note}")
    lines.append("")

    # -- Estimate summary for current selection
    est = estimate_model_memory_gb(
        model, selected_quant,
        context_override=selected_ctx if state.whatif_ctx_index > 0 else None,
        kv_quant=kv_quant,
    )
    if est.incomplete:
        lines.append(
            f"  [{DIM}]Selected:[/{DIM}] [{BRIGHT}]{selected_quant}[/{BRIGHT}]  "
            f"[{DIM}]ctx:[/{DIM}] [{BRIGHT}]{ctx_label}[/{BRIGHT}]  "
            f"[{DIM}]est:[/{DIM}] [{BRIGHT}]? (insufficient metadata)[/{BRIGHT}]"
        )
    else:
        lines.append(
            f"  [{DIM}]Selected:[/{DIM}] [{BRIGHT}]{selected_quant}[/{BRIGHT}]  "
            f"[{DIM}]ctx:[/{DIM}] [{BRIGHT}]{ctx_label}[/{BRIGHT}]  "
            f"[{DIM}]est:[/{DIM}] [{BRIGHT}]{est.total_gb:.1f} GB[/{BRIGHT}]  "
            f"[{DIM}](model {est.total_gb - est.kv_cache_gb - est.runtime_overhead_gb:.1f} + "
            f"kv {est.kv_cache_gb:.1f} + rt {est.runtime_overhead_gb:.1f})[/{DIM}]"
        )
    lines.append("")

    # -- Build the fit table for all GPU nodes with the selected quant/ctx
    gpu_nodes = state.whatif_gpu_nodes()

    if not gpu_nodes:
        lines.append(f"  [{DIM}]No GPU nodes in workspace[/{DIM}]")
        return Panel(
            Text.from_markup("\n".join(lines)),
            title=f"[{DIM}]What-If Analysis[/{DIM}]",
            title_align="left",
            border_style=BORDER,
        )

    table = Table(
        expand=True,
        show_edge=False,
        pad_edge=True,
        show_header=True,
        header_style=MUTED,
    )
    table.add_column("Node Type", min_width=24, style=BRIGHT)
    table.add_column("Fit", width=11)
    table.add_column("Headroom", justify="right", width=9, style=MUTED)
    table.add_column("GPU Mem", justify="right", width=8, style=DIM)
    table.add_column("GPUs", justify="center", width=5, style=DIM)
    table.add_column("Mem%", justify="right", width=5, style=MUTED)
    table.add_column("Est. tok/s", justify="right", width=10, style=DIM)
    if state.pricing_loaded:
        table.add_column("$/hr", justify="right", width=7, style=MUTED)

    # Compute visible window for the table
    available_table_rows = max(body_height - len(lines) - 6, 5)
    state.clamp_whatif_table()
    state.whatif_table_offset, start, end = state.compute_scroll_window(
        state.whatif_table_index, state.whatif_table_offset,
        len(gpu_nodes), available_table_rows + 4,  # +4 because compute_scroll_window subtracts 4
    )

    for i in range(start, end):
        node = gpu_nodes[i]
        is_selected = i == state.whatif_table_index

        # Total GPU memory: per-GPU × count (tensor parallelism across all GPUs)
        available = (node.gpu_memory_gb or 0.0) * max(node.gpu_count, 1)
        headroom = available - est.total_gb

        if node.gpu_count <= 0 or available <= 0:
            fit_level = FitLevel.UNLIKELY
        elif headroom >= max(available * 0.15, 2.0):
            fit_level = FitLevel.SAFE
        elif headroom >= 0:
            fit_level = FitLevel.BORDERLINE
        else:
            fit_level = FitLevel.UNLIKELY

        fit_color = FIT_COLORS.get(fit_level, MUTED)
        fit_text = f"[{fit_color}]{fit_level.value}[/{fit_color}]"
        headroom_str = "?" if est.incomplete else f"{headroom:+.1f} GB"
        total_gpu_mem = (node.gpu_memory_gb or 0.0) * max(node.gpu_count, 1) if node.gpu_memory_gb else 0.0
        gpu_mem = f"{total_gpu_mem:.0f} GB" if total_gpu_mem else "-"

        # Memory usage %
        if total_gpu_mem > 0 and not est.incomplete:
            mem_pct = (est.total_gb / total_gpu_mem) * 100
            if mem_pct < 70:
                pct_color = "green"
            elif mem_pct <= 85:
                pct_color = "yellow"
            else:
                pct_color = "red"
            mem_pct_str = f"[{pct_color}]{mem_pct:.0f}%[/{pct_color}]"
        else:
            mem_pct_str = "-"

        # Estimated tok/s
        if not est.incomplete and model.parameter_count and node.gpu_memory_bandwidth_gb_s:
            from ..engines.fit import estimate_tokens_per_second
            tps = estimate_tokens_per_second(model, node, selected_quant)
            tok_str = f"~{tps:,.0f}" if tps is not None else "-"
        else:
            tok_str = "-"

        style = SELECTED_STYLE if is_selected else ""
        prefix = ">" if is_selected else " "

        row = [
            f"{prefix} {node.node_type_id}",
            fit_text,
            headroom_str,
            gpu_mem,
            str(node.gpu_count),
            mem_pct_str,
            tok_str,
        ]
        if state.pricing_loaded:
            total = _node_total_hourly(
                node.node_type_id, state.vm_pricing,
                node.dbu_per_hour, state.dbu_rate_per_unit,
                state.pricing_discount, state.pricing_vat,
            )
            row.append(f"{total:.2f}" if total is not None else "-")

        table.add_row(*row, style=style)

    # Position indicator
    total = len(gpu_nodes)
    pos = f"{start + 1}-{end} of {total}" if total > 0 else "0"

    # Count fit levels
    safe = 0
    border = 0
    unlikely = 0
    for node in gpu_nodes:
        avail = (node.gpu_memory_gb or 0.0) * max(node.gpu_count, 1)
        h = avail - est.total_gb
        if node.gpu_count <= 0 or avail <= 0:
            unlikely += 1
        elif h >= max(avail * 0.15, 2.0):
            safe += 1
        elif h >= 0:
            border += 1
        else:
            unlikely += 1

    # Build the panel combining text header and table
    header_text = Text.from_markup("\n".join(lines))

    # We need to compose the header text + table into a single renderable
    from rich.console import Group
    content = Group(header_text, table)

    subtitle_parts = [
        f"[green]{safe}ok[/green]",
        f"[yellow]{border}maybe[/yellow]",
        f"[red]{unlikely}no[/red]",
        f"[{DIM}]{pos}[/{DIM}]",
    ]

    return Panel(
        content,
        title=f"[{DIM}]What-If Analysis[/{DIM}]",
        subtitle="  ".join(subtitle_parts),
        title_align="left",
        subtitle_align="right",
        border_style=BORDER,
    )
