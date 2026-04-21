"""Main TUI application loop."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from rich.console import Console
from rich.live import Live

from ..adapters.huggingface import (
    GatedRepoError,
    HuggingFaceAPIError,
    fetch_huggingface_metadata,
    normalize_huggingface_repo_metadata,
)
from ..adapters.huggingface.catalog import CURATED_MODELS, discover_trending_models, get_full_catalog
from ..adapters.azure.dbu_rates import (
    build_dbu_rate_cache,
    fetch_dbu_rates,
    fetch_dbu_unit_prices,
    load_dbu_cache,
    save_dbu_cache,
)
from ..adapters.azure.price_cache import (
    load_price_cache,
    refresh_price_cache,
)
from ..auth import (
    DatabricksCredentials,
    HuggingFaceCredentials,
    KeyringNotAvailableError,
    credential_exists,
    load_stored_credentials,
    run_auth_wizard,
)
from ..collectors.databricks import DatabricksAPIError, DatabricksInventoryCollector
from ..collectors.databricks.inventory import enrich_dbu_rates
from ..config import AppConfig, WorkloadType, WORKLOAD_DBU_PRESETS, WORKLOAD_LABELS, _WORKLOAD_CYCLE, save_pricing_config
from ..domain import ModelFamily, WorkloadProfile
from ..planners import recommend_compute_for_model
from .keys import (
    KEY_BACKSPACE,
    KEY_DOWN,
    KEY_END,
    KEY_ENTER,
    KEY_ESCAPE,
    KEY_HOME,
    KEY_LEFT,
    KEY_PAGE_DOWN,
    KEY_PAGE_UP,
    KEY_RIGHT,
    KEY_TAB,
    KEY_UP,
    read_key,
    read_key_nonblocking,
)
from .state import FitFilter, InputMode, TuiState, View
from .views import build_layout


# -- History file -----------------------------------------------------------

_HISTORY_PATH = Path(
    os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
) / "dbx-model-planner" / "model_history.json"


def _load_model_history() -> list[str]:
    """Load model history from disk."""
    try:
        if _HISTORY_PATH.exists():
            data = json.loads(_HISTORY_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(x) for x in data[:10]]
    except Exception:
        pass
    return []


def _save_model_history(history: list[str]) -> None:
    """Persist model history to disk."""
    try:
        _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        _HISTORY_PATH.write_text(json.dumps(history[:10]), encoding="utf-8")
    except Exception:
        pass


# -- Entry point ------------------------------------------------------------


def _wait_for_thread(
    thread: threading.Thread,
    state: TuiState,
    live: Live,
    console: Console,
) -> bool:
    """Wait for a background thread while keeping the UI responsive.

    Polls for keypresses so the user can press ``q`` to quit even during
    long network operations.  Returns ``True`` if the thread completed
    normally, ``False`` if the user requested quit.
    """
    while thread.is_alive():
        key = read_key_nonblocking(0.15)
        if key in ("q", "Q"):
            state.should_quit = True
            return False
        live.update(build_layout(state, console.height))
    return True


def _prompt_initial_pricing_setup(console: Console, config: AppConfig) -> None:
    """Prompt the user for pricing config when no Azure region is set."""
    console.print()
    console.print("[bold yellow]Pricing Setup Required[/bold yellow]")
    console.print("No Azure region is configured. Please provide pricing information.")
    console.print()

    # Azure region (required)
    while True:
        region = console.input("[bold]Azure region[/bold] (e.g. eastus, westeurope): ").strip()
        if region:
            break
        console.print("[red]Azure region is required.[/red]")

    # Discount rate (optional)
    discount_input = console.input("[bold]Discount rate %[/bold] [dim](default 0)[/dim]: ").strip()
    try:
        discount_rate = max(0.0, min(1.0, float(discount_input) / 100.0)) if discount_input else 0.0
    except ValueError:
        discount_rate = 0.0
        console.print("[yellow]Invalid discount rate, using 0%.[/yellow]")

    # VAT rate (optional)
    vat_input = console.input("[bold]VAT rate %[/bold] [dim](default 0)[/dim]: ").strip()
    try:
        vat_rate = max(0.0, min(1.0, float(vat_input) / 100.0)) if vat_input else 0.0
    except ValueError:
        vat_rate = 0.0
        console.print("[yellow]Invalid VAT rate, using 0%.[/yellow]")

    # Apply to config
    config.pricing.azure_region = region
    config.pricing.discount_rate = discount_rate
    config.pricing.vat_rate = vat_rate

    # Persist to config file
    try:
        save_pricing_config(
            azure_region=region,
            discount_rate=discount_rate,
            vat_rate=vat_rate,
        )
        console.print("[green]Pricing configuration saved.[/green]")
    except Exception:
        console.print("[yellow]Could not save pricing config to file (values applied in memory).[/yellow]")


def run_tui(*, config: AppConfig) -> int:
    """Run the interactive TUI application.

    Flow:
      1. Load or prompt for credentials (in normal terminal mode)
      2. Sync live inventory from Databricks
      3. Enter full-screen TUI
    """
    console = Console()

    # -- Step 1: Credentials (pre-TUI, normal terminal) ----------------
    console.print()
    console.print("[bold]dbx-model-planner[/bold]")
    console.print()

    dbx_creds, hf_creds, azure_region = _load_credentials(console)
    if dbx_creds is None:
        console.print("[red]Cannot continue without Databricks credentials.[/red]")
        return 1

    # Apply stored region to config (credential region takes precedence
    # over config file, but env var / config file can still override).
    if azure_region and not config.pricing.azure_region:
        config.pricing.azure_region = azure_region

    # -- Step 1b: Pricing setup (pre-TUI, if no region configured) -----
    if not config.pricing.azure_region:
        _prompt_initial_pricing_setup(console, config)

    # -- Step 2: Live inventory sync (pre-TUI, normal terminal) --------
    console.print()
    console.print("Syncing workspace inventory...")
    try:
        collector = DatabricksInventoryCollector(credentials=dbx_creds)
        collection = collector.collect(
            progress_fn=lambda msg: console.print(f"  {msg}"),
        )
        for note in collection.notes:
            console.print(f"  {note}")
    except DatabricksAPIError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1

    # -- Step 3: Build state and enter TUI -----------------------------
    state = TuiState(
        inventory=collection.snapshot,
        workspace_notes=collection.notes,
        model_history=_load_model_history(),
        pricing_discount=config.pricing.discount_rate,
        pricing_vat=config.pricing.vat_rate,
        currency_code=config.pricing.currency_code,
        dbu_rate_per_unit=config.databricks.dbu_rate_per_unit,
        workload_type=config.databricks.workload_type,
    )
    state.rebuild_node_lists()

    # -- Step 3a: Load DBU rates (from cache or live fetch) ------------
    _load_dbu_rates(state, config, console)

    # -- Step 3b: Try to load cached prices ----------------------------
    if config.pricing.auto_fetch_pricing and config.pricing.azure_region:
        _try_load_cached_prices(state, config)

    console.print()
    dbu_count = sum(1 for n in state.inventory.compute if n.dbu_per_hour is not None)
    dbu_msg = f", {dbu_count} with DBU rates" if dbu_count else ""
    console.print(f"Loaded {len(state.gpu_nodes)} GPU nodes, {len(state.cpu_nodes)} CPU nodes{dbu_msg}.")
    if state.pricing_loaded:
        console.print(f"Loaded {state.pricing_node_count} cached prices ({state.pricing_region}).")
    if not config.pricing.azure_region:
        console.print("[yellow]No Azure region configured. Press $ in TUI to set up pricing.[/yellow]")
    console.print("Launching TUI... (press [bold]q[/bold] to quit)")
    console.print()

    # Enter full-screen TUI
    try:
        _run_tui_loop(state, config, hf_creds, console)
    except KeyboardInterrupt:
        pass

    # Persist model history
    _save_model_history(state.model_history)

    return 0


def _load_credentials(console: Console) -> tuple[DatabricksCredentials | None, HuggingFaceCredentials | None, str | None]:
    """Load stored credentials or run the wizard."""
    if credential_exists("databricks"):
        try:
            dbx_creds, hf_creds, azure_region = load_stored_credentials()
            if dbx_creds:
                console.print(f"Loaded credentials: {dbx_creds.host}")
                if azure_region:
                    console.print(f"Azure region: {azure_region}")
                return dbx_creds, hf_creds, azure_region
        except KeyringNotAvailableError:
            console.print("[yellow]System keyring unavailable.[/yellow]")

    console.print("No stored credentials. Starting setup...")
    try:
        return run_auth_wizard(input_fn=input, output_fn=console.print)
    except KeyringNotAvailableError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return None, None, None


def _build_dbu_pricing(state: TuiState, config: AppConfig) -> dict[str, float] | None:
    """Build DBU hourly cost dict from inventory dbu_per_hour and state rate.

    Formula: DBU_cost = DBU_Count × per_DBU_unit_rate.

    Uses ``state.dbu_rate_per_unit`` which reflects API-derived prices
    (updated by workload type toggle and pricing wizard), falling back
    to the config value only when state hasn't been updated.
    """
    if state.inventory is None:
        return None
    rate = state.dbu_rate_per_unit
    if rate <= 0:
        return None
    result: dict[str, float] = {}
    for node in state.inventory.compute:
        if node.dbu_per_hour is not None and node.dbu_per_hour > 0:
            result[node.node_type_id] = round(node.dbu_per_hour * rate, 4)
    return result or None


def _sync_dbu_unit_price(state: TuiState, config: AppConfig | None = None) -> None:
    """Update ``state.dbu_rate_per_unit`` from API-derived per-DBU prices.

    If *config* is provided, its ``databricks.dbu_rate_per_unit`` is also updated
    so that cost calculations use the correct per-DBU price in USD.
    """
    wt = state.workload_type
    if state.dbu_unit_prices and state.dbu_unit_price_currency == state.currency_code:
        price = state.dbu_unit_prices.get(wt)
        if price is not None:
            state.dbu_rate_per_unit = price
            if config is not None:
                config.databricks.dbu_rate_per_unit = price
            return
    # Fallback: leave state.dbu_rate_per_unit unchanged (preset or manual)


def _try_load_cached_prices(state: TuiState, config: AppConfig) -> None:
    """Try to load prices from the file cache. Non-blocking."""
    try:
        cache = load_price_cache(ttl_seconds=config.pricing.price_cache_ttl_seconds)
        if cache.is_populated and cache.region == config.pricing.azure_region:
            state.vm_pricing = cache.as_vm_pricing_dict()
            state.pricing_loaded = True
            state.pricing_region = cache.region
            state.pricing_node_count = len(state.vm_pricing)
    except Exception:
        pass


def _load_dbu_rates(state: TuiState, config: AppConfig, console: Console) -> None:
    """Load DBU rates from cache. If cache is missing, schedule a background fetch.

    The Azure pricing page is ~22 MB, so we never block startup on it.
    On first run, DBU rates will appear after the TUI launches.
    """
    if state.inventory is None:
        return

    # Try loading from cache first (instant)
    cache = load_dbu_cache()
    if cache is not None and cache.is_populated:
        count = enrich_dbu_rates(state.inventory.compute, cache.as_dict())
        if count > 0:
            console.print(f"  Loaded {count} DBU rates from cache.")

        # Apply cached per-DBU unit prices if available and currency matches
        if cache.dbu_unit_prices and cache.unit_price_currency == state.currency_code:
            state.dbu_unit_prices = dict(cache.dbu_unit_prices)
            state.dbu_unit_price_currency = cache.unit_price_currency
            _sync_dbu_unit_price(state, config)
            console.print(
                f"  Loaded per-DBU prices from cache ({cache.unit_price_currency})."
            )

        if count > 0:
            return

    # No cache -- will fetch in background after TUI launches
    console.print("  DBU rates not cached. Will fetch in background.")


def _fetch_dbu_rates_background(
    state: TuiState,
    config: AppConfig,
    live: Live,
    console: Console,
) -> None:
    """Fetch DBU rates from Azure pricing page in a background thread (non-blocking).

    Also fetches per-DBU unit prices from the Azure Retail Prices API
    so that DBU costs are in USD.
    """
    if state.inventory is None:
        return

    # Don't start a second fetch if one is already in progress
    if state.active_dbu_thread and state.active_dbu_thread.is_alive():
        state.status_message = "DBU rate fetch already in progress..."
        return

    state.status_message = "Fetching DBU rates from Azure pricing page..."
    live.update(build_layout(state, console.height))

    result: dict[str, object] = {}

    def _fetch() -> None:
        try:
            entries = fetch_dbu_rates(timeout=90.0)
            if entries:
                result["entries"] = entries
            else:
                result["error"] = "No entries parsed"
        except Exception as exc:
            result["error"] = str(exc)

        # Also fetch per-DBU unit prices from the Azure Retail Prices API
        try:
            unit_prices = fetch_dbu_unit_prices(
                region=config.pricing.azure_region,
                currency_code=config.pricing.currency_code,
            )
            if unit_prices:
                result["unit_prices"] = unit_prices
        except Exception:
            pass  # Non-fatal; we'll fall back to presets

    def _finalize() -> None:
        if "entries" in result:
            entries = result["entries"]
            cache = build_dbu_rate_cache(entries)
            count = enrich_dbu_rates(state.inventory.compute, cache.as_dict())

            # Attach per-DBU unit prices to cache if fetched
            unit_prices = result.get("unit_prices")
            if unit_prices:
                cache.dbu_unit_prices = unit_prices
                cache.unit_price_currency = config.pricing.currency_code
                state.dbu_unit_prices = dict(unit_prices)
                state.dbu_unit_price_currency = config.pricing.currency_code
                _sync_dbu_unit_price(state, config)

            state.status_message = f"Loaded {count} DBU rates from Azure pricing page."
            try:
                save_dbu_cache(cache)
            except Exception:
                pass
        elif "error" in result:
            state.status_message = f"DBU rate fetch failed: {result['error']}"
        else:
            state.status_message = ""

    thread = threading.Thread(target=_fetch, daemon=True)
    thread.start()
    state.active_dbu_thread = thread
    state.active_dbu_finalizer = _finalize


def _fetch_prices_background(
    state: TuiState,
    config: AppConfig,
    live: Live,
    console: Console,
) -> None:
    """Fetch Azure VM prices in a background thread (non-blocking)."""
    if state.inventory is None:
        return

    # Don't start a second fetch if one is already in progress
    if state.active_price_thread and state.active_price_thread.is_alive():
        state.status_message = "Price fetch already in progress..."
        return

    state.pricing_loading = True
    state.status_message = f"Fetching Azure prices ({config.pricing.azure_region})..."
    live.update(build_layout(state, console.height))

    node_ids = [n.node_type_id for n in state.inventory.compute]
    result: dict = {}

    def _fetch() -> None:
        try:
            cache = refresh_price_cache(
                node_ids,
                config.pricing.azure_region,
                currency_code=config.pricing.currency_code,
                ttl_seconds=config.pricing.price_cache_ttl_seconds,
            )
            result["cache"] = cache
        except Exception as exc:
            result["error"] = str(exc)

    def _finalize() -> None:
        state.pricing_loading = False
        if "error" in result:
            state.pricing_error = result["error"]
            state.status_message = f"Pricing error: {result['error']}"
            return
        cache = result.get("cache")
        if cache is not None:
            state.vm_pricing = cache.as_vm_pricing_dict()
            state.pricing_loaded = True
            state.pricing_region = cache.region
            state.pricing_node_count = len(state.vm_pricing)
            state.pricing_error = None
            state.status_message = f"Loaded {len(state.vm_pricing)} prices ({cache.region})"

    thread = threading.Thread(target=_fetch, daemon=True)
    thread.start()
    state.active_price_thread = thread
    state.active_price_finalizer = _finalize


def _run_tui_loop(
    state: TuiState,
    config: AppConfig,
    hf_creds: HuggingFaceCredentials | None,
    console: Console,
) -> None:
    """Full-screen TUI event loop using Rich Live."""

    with Live(
        build_layout(state, console.height),
        console=console,
        screen=True,
        refresh_per_second=10,
    ) as live:
        # Auto-fetch pricing on TUI startup if enabled, region is set, and not already loaded
        has_region = bool(config.pricing.azure_region)
        if config.pricing.auto_fetch_pricing and not state.pricing_loaded and has_region:
            _fetch_prices_background(state, config, live, console)
            live.update(build_layout(state, console.height))

        # Fetch DBU rates in background if not cached
        _has_dbu = any(n.dbu_per_hour is not None for n in (state.inventory.compute if state.inventory else []))
        if not _has_dbu:
            _fetch_dbu_rates_background(state, config, live, console)
            live.update(build_layout(state, console.height))
        elif not state.dbu_unit_prices and has_region:
            # DBU counts are cached but per-DBU unit prices are missing
            # (e.g. old cache format). Fetch prices
            # from the Azure Retail Prices API (lightweight call).
            _fetch_dbu_unit_prices_background(state, config, live, console)
            live.update(build_layout(state, console.height))

        if not has_region and not state.pricing_loaded:
            state.status_message = "No Azure region set. Press $ to configure or run 'auth login'."

        while not state.should_quit:
            # -- Check if background fetches completed -----------------
            if state.active_price_thread and not state.active_price_thread.is_alive():
                if state.active_price_finalizer:
                    state.active_price_finalizer()
                    state.active_price_finalizer = None
                state.active_price_thread = None
                live.update(build_layout(state, console.height))

            if state.active_dbu_thread and not state.active_dbu_thread.is_alive():
                if state.active_dbu_finalizer:
                    state.active_dbu_finalizer()
                    state.active_dbu_finalizer = None
                state.active_dbu_thread = None
                live.update(build_layout(state, console.height))

            # Read a keypress (non-blocking so we can poll threads)
            key = read_key_nonblocking(0.15)
            if key is None:
                continue

            # Clear stale status messages on any keypress so footer
            # keybinding hints reappear.  Handlers can still set new
            # status messages that will show until the next keypress.
            state.status_message = ""

            # Handle the key based on current input mode
            if state.input_mode == InputMode.SEARCH:
                _handle_search_input(state, key, config, hf_creds, live, console)
            elif state.input_mode == InputMode.MODEL_ID:
                _handle_model_id_input(state, key, config, hf_creds, live, console)
            elif state.input_mode == InputMode.PRICING:
                _handle_pricing_input(state, key, config, live, console)
            else:
                _handle_normal_input(state, key, config, hf_creds, live, console)

            # Re-render
            live.update(build_layout(state, console.height))


# -- Input handlers ----------------------------------------------------------


def _handle_normal_input(
    state: TuiState,
    key: str,
    config: AppConfig,
    hf_creds: HuggingFaceCredentials | None,
    live: Live,
    console: Console,
) -> None:
    """Handle keypresses in normal navigation mode."""

    if key in ("q", "Q"):
        state.should_quit = True
        return

    if key == KEY_ESCAPE:
        if state.view in (View.MODEL_FIT, View.MODEL_INPUT, View.MODEL_BROWSE, View.WHAT_IF, View.PRICING_SETUP):
            _go_back(state)
        return

    # -- Navigation: j/k/arrows -----------------------------------------
    if key in ("j", KEY_DOWN):
        if state.view == View.INVENTORY:
            state.selected_index += 1
            state.clamp_selection()
        elif state.view == View.MODEL_FIT:
            state.fit_selected_index += 1
            state.clamp_fit_selection()
        elif state.view == View.MODEL_BROWSE:
            state.browse_selected_index += 1
            state.clamp_browse_selection()
        elif state.view == View.WHAT_IF:
            state.whatif_table_index += 1
            state.clamp_whatif_table()
        return

    if key in ("k", KEY_UP):
        if state.view == View.INVENTORY:
            state.selected_index -= 1
            state.clamp_selection()
        elif state.view == View.MODEL_FIT:
            state.fit_selected_index -= 1
            state.clamp_fit_selection()
        elif state.view == View.MODEL_BROWSE:
            state.browse_selected_index -= 1
            state.clamp_browse_selection()
        elif state.view == View.WHAT_IF:
            state.whatif_table_index -= 1
            state.clamp_whatif_table()
        return

    # -- Page navigation ------------------------------------------------
    if key == KEY_PAGE_DOWN:
        state.page_down()
        return

    if key == KEY_PAGE_UP:
        state.page_up()
        return

    # -- Home / End / g / G ---------------------------------------------
    if key in ("g", KEY_HOME):
        if state.view == View.INVENTORY:
            state.selected_index = 0
            state.scroll_offset = 0
        elif state.view == View.MODEL_FIT:
            state.fit_selected_index = 0
            state.fit_scroll_offset = 0
        elif state.view == View.MODEL_BROWSE:
            state.browse_selected_index = 0
            state.browse_scroll_offset = 0
        elif state.view == View.WHAT_IF:
            state.whatif_table_index = 0
            state.whatif_table_offset = 0
        return

    if key in ("G", KEY_END):
        if state.view == View.INVENTORY:
            state.selected_index = max(0, len(state.displayed_nodes) - 1)
        elif state.view == View.MODEL_FIT:
            if state.fit_displayed_candidates:
                state.fit_selected_index = max(0, len(state.fit_displayed_candidates) - 1)
        elif state.view == View.MODEL_BROWSE:
            state.browse_selected_index = max(0, len(state.browse_displayed) - 1)
        elif state.view == View.WHAT_IF:
            count = state.whatif_candidate_count()
            state.whatif_table_index = max(0, count - 1)
        return

    # -- Search ---------------------------------------------------------
    if key == "/":
        if state.view == View.MODEL_BROWSE:
            state.input_mode = InputMode.SEARCH
            state.browse_search = ""
        else:
            state.input_mode = InputMode.SEARCH
            state.search_query = ""
        return

    # -- CPU toggle --------------------------------------------------------
    if key in ("c", "C"):
        if state.view == View.INVENTORY:
            state.toggle_cpu_nodes()
        return

    # -- Model input (from inventory or model fit only) -------------------
    if key in ("m", "M"):
        if state.view in (View.INVENTORY, View.MODEL_FIT):
            state.previous_view = state.view
            state.view = View.MODEL_INPUT
            state.input_mode = InputMode.MODEL_ID
            state.input_buffer = ""
            state.status_message = ""
        return

    # -- Browse model catalog -------------------------------------------
    if key in ("b", "B"):
        if state.view == View.INVENTORY:
            _open_browse(state)
        return

    # -- Fit filter in model fit view ----------------------------------------
    if key in ("f", "F"):
        if state.view == View.MODEL_FIT:
            state.cycle_fit_filter()
        return

    # -- Discover in browse -------------------------------------------------
    if key in ("d", "D"):
        if state.view == View.MODEL_BROWSE:
            _discover_trending(state, hf_creds, live, console)
        return

    # -- What-if view from model fit view --------------------------------
    if key in ("w", "W"):
        if state.view == View.MODEL_FIT and state.model_profile is not None:
            if state.model_profile.family in (ModelFamily.EMBEDDING, ModelFamily.RERANKER):
                state.status_message = "What-if analysis is not available for embedding/reranker models"
            else:
                _enter_whatif(state)
        return

    # -- What-if TurboQuant KV cache toggle (K) ---------------------------
    if key == "K":
        if state.view == View.WHAT_IF:
            state.whatif_turboquant = not state.whatif_turboquant
            tq = "ON" if state.whatif_turboquant else "OFF"
            state.status_message = f"TurboQuant KV cache compression: {tq}"
        return

    # -- What-if selector navigation (left/right/h/l/Tab) -----------------
    if key in (KEY_LEFT, "h"):
        if state.view == View.WHAT_IF:
            _whatif_selector_left(state)
        return

    if key in (KEY_RIGHT, "l"):
        if state.view == View.WHAT_IF:
            _whatif_selector_right(state)
        return

    # -- Price refresh ---------------------------------------------------
    if key == "$":
        _enter_pricing_setup(state, config)
        return

    # -- Workload type toggle -------------------------------------------
    if key in ("t", "T"):
        _toggle_workload_type(state, config)
        return

    # -- Tab: cycle category in browse / toggle what-if selector
    if key == KEY_TAB:
        if state.view == View.MODEL_BROWSE:
            _cycle_browse_category(state)
        elif state.view == View.WHAT_IF:
            state.whatif_selector_row = 1 - state.whatif_selector_row
        return

    # -- Enter: fit from browse -------------------------------------------
    if key == KEY_ENTER:
        if state.view == View.MODEL_BROWSE:
            entry = state.selected_browse_entry()
            if entry:
                # Fit the selected model against workspace
                state.previous_view = View.MODEL_BROWSE
                state.model_gated = entry.gated
                _fetch_model_threaded(state, entry.model_id, config, hf_creds, live, console)
        return


def _handle_search_input(
    state: TuiState,
    key: str,
    config: AppConfig,
    hf_creds: HuggingFaceCredentials | None,
    live: Live,
    console: Console,
) -> None:
    """Handle keypresses in search mode."""
    is_browse = state.view == View.MODEL_BROWSE

    if key == KEY_ESCAPE:
        state.input_mode = InputMode.NORMAL
        if is_browse:
            state.browse_search = ""
            state.rebuild_browse_list()
            state.browse_selected_index = 0
            state.browse_scroll_offset = 0
        else:
            state.search_query = ""
            state.rebuild_node_lists()
            state.selected_index = 0
            state.scroll_offset = 0
        return

    if key == KEY_ENTER:
        state.input_mode = InputMode.NORMAL
        if is_browse:
            search_text = state.browse_search.strip()
            parts = search_text.split("/")
            if len(parts) == 2 and all(p.strip() for p in parts) and " " not in search_text:
                # Looks like a HuggingFace model ID (e.g. meta-llama/Llama-3-8B)
                # Fetch directly from HuggingFace
                state.browse_search = ""
                state.previous_view = View.MODEL_BROWSE
                _fetch_model_threaded(state, search_text, config, hf_creds, live, console)
            else:
                state.rebuild_browse_list()
                state.browse_selected_index = 0
                state.browse_scroll_offset = 0
        else:
            state.rebuild_node_lists()
            state.selected_index = 0
            state.scroll_offset = 0
        return

    if key == KEY_BACKSPACE:
        if is_browse:
            state.browse_search = state.browse_search[:-1]
            state.rebuild_browse_list()
            state.browse_selected_index = 0
            state.browse_scroll_offset = 0
        else:
            state.search_query = state.search_query[:-1]
            state.rebuild_node_lists()
            state.selected_index = 0
            state.scroll_offset = 0
        return

    if len(key) == 1 and key.isprintable():
        if is_browse:
            state.browse_search += key
            state.rebuild_browse_list()
            state.browse_selected_index = 0
            state.browse_scroll_offset = 0
        else:
            state.search_query += key
            state.rebuild_node_lists()
            state.selected_index = 0
            state.scroll_offset = 0


def _handle_model_id_input(
    state: TuiState,
    key: str,
    config: AppConfig,
    hf_creds: HuggingFaceCredentials | None,
    live: Live,
    console: Console,
) -> None:
    """Handle keypresses when entering a HuggingFace model ID."""

    if key == KEY_ESCAPE:
        state.input_mode = InputMode.NORMAL
        state.input_buffer = ""
        _go_back(state)
        return

    if key == KEY_ENTER:
        model_id = state.input_buffer.strip()
        if not model_id:
            state.status_message = "Please enter a model ID"
            return

        state.input_mode = InputMode.NORMAL

        # Threaded fetch with loading spinner
        _fetch_model_threaded(state, model_id, config, hf_creds, live, console)
        return

    if key == KEY_BACKSPACE:
        state.input_buffer = state.input_buffer[:-1]
        return

    if len(key) == 1 and key.isprintable():
        state.input_buffer += key


def _fetch_model_threaded(
    state: TuiState,
    model_id: str,
    config: AppConfig,
    hf_creds: HuggingFaceCredentials | None,
    live: Live,
    console: Console,
) -> None:
    """Fetch model from HuggingFace in a background thread with loading spinner."""
    state.loading = True
    state.status_message = f"Fetching {model_id}..."
    live.update(build_layout(state, console.height))

    result: dict = {}

    def _fetch() -> None:
        try:
            raw_metadata = fetch_huggingface_metadata(model_id, credentials=hf_creds)
            normalized = normalize_huggingface_repo_metadata(raw_metadata)
            result["model"] = normalized.model_profile
        except GatedRepoError as exc:
            result["error"] = f"Gated repo: {exc}. Add HF token with 'auth login'."
        except HuggingFaceAPIError as exc:
            result["error"] = f"HuggingFace error: {exc}"
        except Exception as exc:
            result["error"] = f"Error: {exc}"

    thread = threading.Thread(target=_fetch, daemon=True)
    thread.start()

    # Spin while thread is alive, allowing q to quit
    if not _wait_for_thread(thread, state, live, console):
        state.loading = False
        return

    state.loading = False

    if "error" in result:
        state.status_message = result["error"]
        state.view = View.MODEL_INPUT
        state.input_mode = InputMode.MODEL_ID
        return

    model = result.get("model")
    if model is None or state.inventory is None:
        state.status_message = "No model or inventory"
        return

    recommendation = recommend_compute_for_model(
        config=config,
        inventory=state.inventory,
        model=model,
        workload=WorkloadProfile(workload_name="tui-fit", online=True),
        vm_pricing=state.vm_pricing or None,
        dbu_pricing=_build_dbu_pricing(state, config),
    )

    state.model_profile = model
    state.model_recommendation = recommendation
    state.fit_filter = FitFilter.ALL
    state.fit_selected_index = 0
    state.fit_scroll_offset = 0
    state.rebuild_fit_list()
    state.view = View.MODEL_FIT
    state.status_message = ""

    # Add to history
    state.add_model_to_history(model_id)


def _go_back(state: TuiState) -> None:
    """Navigate back to the previous view."""
    if state.previous_view is not None:
        state.view = state.previous_view
        state.previous_view = None
    else:
        state.view = View.INVENTORY
    state.status_message = ""
    state.input_mode = InputMode.NORMAL


# -- Browse helpers ---------------------------------------------------------

_BROWSE_CATEGORIES = ["", "LLM", "Embedding", "VLM", "Code"]


def _open_browse(state: TuiState) -> None:
    """Open the model browse view, initializing catalog if needed."""
    if not state.browse_catalog:
        state.browse_catalog = list(CURATED_MODELS)
        state.rebuild_browse_list()

    state.previous_view = state.view
    state.view = View.MODEL_BROWSE
    state.browse_selected_index = 0
    state.browse_scroll_offset = 0
    state.browse_search = ""
    state.status_message = ""


def _cycle_browse_category(state: TuiState) -> None:
    """Cycle through category filters in browse view."""
    current = state.browse_category_filter
    try:
        idx = _BROWSE_CATEGORIES.index(current)
    except ValueError:
        idx = 0
    next_idx = (idx + 1) % len(_BROWSE_CATEGORIES)
    state.browse_category_filter = _BROWSE_CATEGORIES[next_idx]
    state.rebuild_browse_list()
    state.browse_selected_index = 0
    state.browse_scroll_offset = 0


def _discover_trending(
    state: TuiState,
    hf_creds: HuggingFaceCredentials | None,
    live: Live,
    console: Console,
) -> None:
    """Fetch trending models from HuggingFace and merge into catalog."""
    if state.browse_discovered:
        state.status_message = "Trending models already loaded"
        return

    state.loading = True
    state.status_message = "Discovering trending models from HuggingFace..."
    live.update(build_layout(state, console.height))

    result: dict = {}

    def _fetch() -> None:
        try:
            discovered = discover_trending_models(credentials=hf_creds, limit=30)
            result["discovered"] = discovered
        except Exception as exc:
            result["error"] = f"Discovery error: {exc}"

    thread = threading.Thread(target=_fetch, daemon=True)
    thread.start()

    if not _wait_for_thread(thread, state, live, console):
        state.loading = False
        return

    state.loading = False

    if "error" in result:
        state.status_message = result["error"]
        return

    discovered = result.get("discovered", [])
    state.browse_catalog = get_full_catalog(discovered)
    state.browse_discovered = True
    state.rebuild_browse_list()
    state.browse_selected_index = 0
    state.browse_scroll_offset = 0
    state.status_message = f"Added {len(discovered)} trending models"


# -- Workload type toggle helpers ------------------------------------------


def _toggle_workload_type(state: TuiState, config: AppConfig) -> None:
    """Cycle between All-Purpose Compute and Jobs Compute workload types.

    The DBU count per VM is the same for all workload types; only the
    per-DBU price differs.  Toggling updates both the config and TUI state.
    Uses API-derived per-DBU prices when available; otherwise falls back
    to the USD presets.
    """
    try:
        current = WorkloadType(state.workload_type)
        idx = _WORKLOAD_CYCLE.index(current)
    except (ValueError, KeyError):
        idx = 0
    next_idx = (idx + 1) % len(_WORKLOAD_CYCLE)
    new_wt = _WORKLOAD_CYCLE[next_idx]

    # Update config and state with new workload type
    config.databricks.workload_type = new_wt.value
    state.workload_type = new_wt.value

    # Use API-derived per-DBU price if available, else fall back to preset
    if (
        state.dbu_unit_prices
        and state.dbu_unit_price_currency == state.currency_code
        and new_wt.value in state.dbu_unit_prices
    ):
        rate = state.dbu_unit_prices[new_wt.value]
    else:
        rate = WORKLOAD_DBU_PRESETS[new_wt]

    config.databricks.dbu_rate_per_unit = rate
    state.dbu_rate_per_unit = rate

    label = WORKLOAD_LABELS[new_wt]
    state.status_message = (
        f"Workload: {label} ({rate:.4f}/DBU)"
    )


# -- Pricing setup wizard helpers ------------------------------------------

_PRICING_STEP_LABELS = ["Azure region", "Discount rate (%)", "VAT rate (%)", "DBU rate (per DBU)"]
_PRICING_STEP_KEYS = ["region", "discount", "vat", "dbu_rate"]


def _enter_pricing_setup(state: TuiState, config: AppConfig) -> None:
    """Enter the interactive pricing setup wizard."""
    state.previous_view = state.view
    state.view = View.PRICING_SETUP
    state.input_mode = InputMode.PRICING
    state.pricing_setup_step = 0
    state.pricing_setup_values = {
        "region": config.pricing.azure_region,
        "discount": str(int(config.pricing.discount_rate * 100)),
        "vat": str(int(config.pricing.vat_rate * 100)),
        "dbu_rate": str(config.databricks.dbu_rate_per_unit),
    }
    state.input_buffer = state.pricing_setup_values["region"]
    state.status_message = ""


def _handle_pricing_input(
    state: TuiState,
    key: str,
    config: AppConfig,
    live: Live,
    console: Console,
) -> None:
    """Handle keypresses during the pricing setup wizard."""
    if key == KEY_ESCAPE:
        state.input_mode = InputMode.NORMAL
        state.input_buffer = ""
        _go_back(state)
        return

    if key == KEY_ENTER:
        # Save current step value
        step_key = _PRICING_STEP_KEYS[state.pricing_setup_step]
        value = state.input_buffer.strip()
        if not value:
            state.status_message = "Please enter a value"
            return

        state.pricing_setup_values[step_key] = value

        if state.pricing_setup_step < 3:
            # Advance to next step
            state.pricing_setup_step += 1
            next_key = _PRICING_STEP_KEYS[state.pricing_setup_step]
            state.input_buffer = state.pricing_setup_values[next_key]
            state.status_message = ""
        else:
            # All steps done — apply values and fetch prices
            _apply_pricing_setup(state, config, live, console)
        return

    if key == KEY_BACKSPACE:
        state.input_buffer = state.input_buffer[:-1]
        return

    if len(key) == 1 and key.isprintable():
        state.input_buffer += key


def _apply_pricing_setup(
    state: TuiState,
    config: AppConfig,
    live: Live,
    console: Console,
) -> None:
    """Apply wizard values to config and trigger price fetch."""
    vals = state.pricing_setup_values

    # Apply region
    config.pricing.azure_region = vals["region"]

    # Apply discount rate (user enters percentage, e.g. "37" → 0.37)
    try:
        config.pricing.discount_rate = float(vals["discount"]) / 100.0
    except ValueError:
        config.pricing.discount_rate = 0.0

    # Apply VAT rate (user enters percentage, e.g. "19" → 0.19)
    try:
        config.pricing.vat_rate = float(vals["vat"]) / 100.0
    except ValueError:
        config.pricing.vat_rate = 0.0

    # Apply DBU rate per unit (user enters direct value, e.g. "0.55")
    try:
        manual_dbu_rate = float(vals["dbu_rate"])
    except ValueError:
        manual_dbu_rate = 0.55
    config.databricks.dbu_rate_per_unit = manual_dbu_rate

    # Exit wizard mode
    state.input_mode = InputMode.NORMAL
    state.view = state.previous_view or View.INVENTORY
    state.previous_view = None
    state.status_message = ""

    # Sync state with updated config values
    state.pricing_discount = config.pricing.discount_rate
    state.pricing_vat = config.pricing.vat_rate
    state.currency_code = config.pricing.currency_code
    state.dbu_rate_per_unit = config.databricks.dbu_rate_per_unit
    state.workload_type = config.databricks.workload_type

    # Invalidate cached per-DBU unit prices since region may
    # have changed.  The user's manual value is honoured until the next
    # background fetch repopulates API-derived prices.
    state.dbu_unit_prices = {}
    state.dbu_unit_price_currency = None

    # Trigger background price fetch with the new settings
    _fetch_prices_background(state, config, live, console)

    # Re-fetch DBU unit prices in the background (non-blocking).
    # This runs in a separate thread and will update state/config
    # with the correct per-DBU prices for the new region.
    _fetch_dbu_unit_prices_background(state, config, live, console)

    # Persist updated pricing config to config.toml so values
    # survive across sessions.
    try:
        save_pricing_config(
            azure_region=config.pricing.azure_region,
            discount_rate=config.pricing.discount_rate,
            vat_rate=config.pricing.vat_rate,
        )
    except Exception:
        pass  # non-critical; values are already applied in memory


def _fetch_dbu_unit_prices_background(
    state: TuiState,
    config: AppConfig,
    live: Live,
    console: Console,
) -> None:
    """Fetch per-DBU unit prices from the Azure Retail Prices API (non-blocking).

    This is a lightweight call (unlike the 22 MB HTML page fetch).
    Updates the DBU rate cache and state with correct per-DBU prices
    in USD.
    """
    # Don't start if a DBU fetch is already running
    if state.active_dbu_thread and state.active_dbu_thread.is_alive():
        return

    result: dict[str, object] = {}

    def _fetch() -> None:
        try:
            unit_prices = fetch_dbu_unit_prices(
                region=config.pricing.azure_region,
                currency_code=config.pricing.currency_code,
            )
            if unit_prices:
                result["unit_prices"] = unit_prices
        except Exception as exc:
            result["error"] = str(exc)

    def _finalize() -> None:
        unit_prices = result.get("unit_prices")
        if unit_prices:
            state.dbu_unit_prices = dict(unit_prices)
            state.dbu_unit_price_currency = config.pricing.currency_code
            _sync_dbu_unit_price(state, config)

            # Persist to the existing DBU rate cache
            try:
                cache = load_dbu_cache()
                if cache is not None:
                    cache.dbu_unit_prices = unit_prices
                    cache.unit_price_currency = config.pricing.currency_code
                    save_dbu_cache(cache)
            except Exception:
                pass

            wt_label = state.workload_type.replace("_", " ").title()
            state.status_message = (
                f"DBU price updated: {config.pricing.currency_code} "
                f"{state.dbu_rate_per_unit:.4f}/DBU ({wt_label})"
            )

    thread = threading.Thread(target=_fetch, daemon=True)
    thread.start()
    state.active_dbu_thread = thread
    state.active_dbu_finalizer = _finalize


# -- What-if view helpers --------------------------------------------------


def _enter_whatif(state: TuiState) -> None:
    """Enter the dedicated what-if view from the model fit view."""
    state.previous_view = View.MODEL_FIT
    state.view = View.WHAT_IF

    # Always start at fp16 (index 0) and default context (index 0)
    state.whatif_quant_index = 0
    state.whatif_ctx_index = 0
    state.whatif_selector_row = 0  # Start on quant selector
    state.whatif_table_index = 0
    state.whatif_table_offset = 0
    state.whatif_turboquant = False  # Start with TurboQuant off
    state.status_message = ""


def _whatif_selector_left(state: TuiState) -> None:
    """Move the active what-if selector to the left."""
    from ..engines.plan import QUANTIZATION_OPTIONS, CONTEXT_PRESETS

    if state.whatif_selector_row == 0:
        # Quant selector
        state.whatif_quant_index = max(0, state.whatif_quant_index - 1)
    else:
        # Context selector
        state.whatif_ctx_index = max(0, state.whatif_ctx_index - 1)


def _whatif_selector_right(state: TuiState) -> None:
    """Move the active what-if selector to the right."""
    from ..engines.plan import QUANTIZATION_OPTIONS, CONTEXT_PRESETS

    if state.whatif_selector_row == 0:
        # Quant selector
        max_idx = len(QUANTIZATION_OPTIONS) - 1
        state.whatif_quant_index = min(max_idx, state.whatif_quant_index + 1)
    else:
        # Context selector (0 = default + N presets)
        max_idx = len(CONTEXT_PRESETS)  # 0..len = len+1 options
        state.whatif_ctx_index = min(max_idx, state.whatif_ctx_index + 1)
