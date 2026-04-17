"""TUI application state management."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from ..adapters.huggingface.catalog import CatalogEntry
from ..domain import (
    CandidateCompute,
    FitLevel,
    HostingRecommendation,
    ModelProfile,
    WorkspaceComputeProfile,
    WorkspaceInventorySnapshot,
)


class View(StrEnum):
    """Active TUI view."""

    INVENTORY = "inventory"
    MODEL_BROWSE = "model_browse"
    MODEL_INPUT = "model_input"
    MODEL_FIT = "model_fit"
    WHAT_IF = "what_if"
    PRICING_SETUP = "pricing_setup"


class InputMode(StrEnum):
    """Whether the user is navigating or typing."""

    NORMAL = "normal"
    SEARCH = "search"
    MODEL_ID = "model_id"
    PRICING = "pricing"


class FitFilter(StrEnum):
    """Filter for model fit candidates by fit level."""

    ALL = "all"
    SAFE = "safe"
    BORDERLINE = "borderline"
    UNLIKELY = "unlikely"


FIT_FILTER_CYCLE = [FitFilter.ALL, FitFilter.SAFE, FitFilter.BORDERLINE, FitFilter.UNLIKELY]

FIT_FILTER_LABELS = {
    FitFilter.ALL: "All",
    FitFilter.SAFE: "Safe",
    FitFilter.BORDERLINE: "Borderline",
    FitFilter.UNLIKELY: "Unlikely",
}


@dataclass
class TuiState:
    """Mutable state for the TUI application."""

    # -- Workspace data -------------------------------------------------
    inventory: WorkspaceInventorySnapshot | None = None
    workspace_notes: list[str] = field(default_factory=list)

    # -- View state -----------------------------------------------------
    view: View = View.INVENTORY
    previous_view: View | None = None

    # -- Navigation -----------------------------------------------------
    selected_index: int = 0
    scroll_offset: int = 0
    visible_rows: int = 20

    # -- Input mode & text ---------------------------------------------
    input_mode: InputMode = InputMode.NORMAL
    input_buffer: str = ""
    search_query: str = ""

    # -- Filtered/sorted inventory items --------------------------------
    gpu_nodes: list[WorkspaceComputeProfile] = field(default_factory=list)
    cpu_nodes: list[WorkspaceComputeProfile] = field(default_factory=list)
    displayed_nodes: list[WorkspaceComputeProfile] = field(default_factory=list)

    # -- CPU toggle -----------------------------------------------------
    show_cpu_nodes: bool = False

    # -- Model fit state ------------------------------------------------
    model_profile: ModelProfile | None = None
    model_recommendation: HostingRecommendation | None = None
    fit_selected_index: int = 0
    fit_scroll_offset: int = 0
    fit_filter: FitFilter = FitFilter.ALL
    fit_displayed_candidates: list[CandidateCompute] = field(default_factory=list)

    # -- Model history --------------------------------------------------
    model_history: list[str] = field(default_factory=list)
    history_max: int = 10

    # -- Model browse ---------------------------------------------------
    browse_catalog: list[CatalogEntry] = field(default_factory=list)
    browse_displayed: list[CatalogEntry] = field(default_factory=list)
    browse_selected_index: int = 0
    browse_scroll_offset: int = 0
    browse_search: str = ""
    browse_category_filter: str = ""  # "" = all, "LLM", "Embedding", "VLM", "Code"
    browse_discovered: bool = False  # Whether trending models have been fetched

    # -- Pricing state ---------------------------------------------------
    vm_pricing: dict[str, float] = field(default_factory=dict)
    pricing_loaded: bool = False
    pricing_loading: bool = False
    pricing_region: str | None = None
    pricing_error: str | None = None
    pricing_node_count: int = 0  # How many nodes have prices
    pricing_discount: float = 0.0  # from config.pricing.discount_rate
    pricing_vat: float = 0.0  # from config.pricing.vat_rate
    currency_code: str = "USD"  # from config.pricing.currency_code
    dbu_rate_per_unit: float = 0.55  # from config.databricks.dbu_rate_per_unit
    workload_type: str = "all_purpose"  # from config.databricks.workload_type

    # Per-DBU unit prices fetched from Azure Retail Prices API
    # (workload_type key → per-DBU price in USD).
    # When populated, these override the static WORKLOAD_DBU_PRESETS.
    dbu_unit_prices: dict[str, float] = field(default_factory=dict)
    dbu_unit_price_currency: str | None = None

    # -- Pricing setup wizard -------------------------------------------
    pricing_setup_step: int = 0  # 0=region, 1=discount, 2=vat, 3=dbu_rate
    pricing_setup_values: dict[str, str] = field(default_factory=dict)

    # -- What-if view state ---------------------------------------------
    whatif_quant_index: int = 0     # index into QUANTIZATION_OPTIONS list
    whatif_ctx_index: int = 0       # 0=model default, 1..N=CONTEXT_PRESETS
    whatif_selector_row: int = 0    # 0=quant selector, 1=context selector
    whatif_table_index: int = 0     # selected row in what-if results table
    whatif_table_offset: int = 0    # scroll offset for what-if table

    # -- Status / messages ----------------------------------------------
    status_message: str = ""
    loading: bool = False

    # -- Quit signal ----------------------------------------------------
    should_quit: bool = False

    def compute_scroll_window(
        self,
        selected: int,
        scroll_offset: int,
        total: int,
        max_rows: int,
    ) -> tuple[int, int, int]:
        """Compute visible window start/end and updated scroll_offset.

        Returns (scroll_offset, start, end).
        """
        visible = max(max_rows - 4, 5)
        if selected < scroll_offset:
            scroll_offset = selected
        elif selected >= scroll_offset + visible:
            scroll_offset = selected - visible + 1
        start = scroll_offset
        end = min(start + visible, total)
        return scroll_offset, start, end

    def rebuild_node_lists(self) -> None:
        """Rebuild GPU/CPU/displayed node lists from current inventory and search."""
        if self.inventory is None:
            self.gpu_nodes = []
            self.cpu_nodes = []
            self.displayed_nodes = []
            return

        self.gpu_nodes = [c for c in self.inventory.compute if c.gpu_count > 0]
        self.cpu_nodes = [c for c in self.inventory.compute if c.gpu_count == 0]

        # Show GPU nodes, optionally include CPU nodes
        if self.show_cpu_nodes:
            nodes = list(self.inventory.compute)
        else:
            nodes = list(self.gpu_nodes)

        if self.search_query:
            q = self.search_query.lower()
            nodes = [
                n for n in nodes
                if q in n.node_type_id.lower()
                or q in (n.gpu_family or "").lower()
            ]

        self.displayed_nodes = self._sort_nodes(nodes)

    def _sort_nodes(self, nodes: list[WorkspaceComputeProfile]) -> list[WorkspaceComputeProfile]:
        """Sort nodes by GPU memory descending (highest VRAM first)."""
        return sorted(nodes, key=lambda n: (n.gpu_memory_gb or 0), reverse=True)

    def toggle_cpu_nodes(self) -> None:
        """Toggle CPU node visibility."""
        self.show_cpu_nodes = not self.show_cpu_nodes
        self.rebuild_node_lists()
        self.selected_index = 0
        self.scroll_offset = 0

    def rebuild_fit_list(self) -> None:
        """Rebuild the displayed fit candidates from the current filter."""
        if self.model_recommendation is None:
            self.fit_displayed_candidates = []
            return

        candidates = self.model_recommendation.candidates

        if self.fit_filter == FitFilter.ALL:
            self.fit_displayed_candidates = list(candidates)
        elif self.fit_filter == FitFilter.SAFE:
            self.fit_displayed_candidates = [c for c in candidates if c.fit_level == FitLevel.SAFE]
        elif self.fit_filter == FitFilter.BORDERLINE:
            self.fit_displayed_candidates = [c for c in candidates if c.fit_level == FitLevel.BORDERLINE]
        elif self.fit_filter == FitFilter.UNLIKELY:
            self.fit_displayed_candidates = [c for c in candidates if c.fit_level == FitLevel.UNLIKELY]

    def cycle_fit_filter(self) -> None:
        """Cycle to the next fit filter level."""
        idx = FIT_FILTER_CYCLE.index(self.fit_filter)
        next_idx = (idx + 1) % len(FIT_FILTER_CYCLE)
        self.fit_filter = FIT_FILTER_CYCLE[next_idx]
        self.rebuild_fit_list()
        self.fit_selected_index = 0
        self.fit_scroll_offset = 0

    def add_model_to_history(self, model_id: str) -> None:
        """Add a model ID to the history, keeping most recent first."""
        # Remove if already present
        self.model_history = [m for m in self.model_history if m != model_id]
        self.model_history.insert(0, model_id)
        # Trim to max
        self.model_history = self.model_history[: self.history_max]

    def clamp_selection(self) -> None:
        """Keep selected_index within bounds."""
        max_index = max(0, len(self.displayed_nodes) - 1)
        self.selected_index = max(0, min(self.selected_index, max_index))

    def clamp_fit_selection(self) -> None:
        """Keep fit_selected_index within bounds."""
        if not self.fit_displayed_candidates:
            self.fit_selected_index = 0
            return
        max_index = max(0, len(self.fit_displayed_candidates) - 1)
        self.fit_selected_index = max(0, min(self.fit_selected_index, max_index))

    def page_down(self) -> None:
        """Move selection down by one page."""
        if self.view == View.INVENTORY:
            self.selected_index = min(
                self.selected_index + self.visible_rows,
                max(0, len(self.displayed_nodes) - 1),
            )
            self.clamp_selection()
        elif self.view == View.WHAT_IF:
            count = self.whatif_candidate_count()
            if count > 0:
                self.whatif_table_index = min(
                    self.whatif_table_index + self.visible_rows,
                    count - 1,
                )
        elif self.view == View.MODEL_FIT and self.fit_displayed_candidates:
            self.fit_selected_index = min(
                self.fit_selected_index + self.visible_rows,
                max(0, len(self.fit_displayed_candidates) - 1),
            )
            self.clamp_fit_selection()
        elif self.view == View.MODEL_BROWSE:
            self.browse_page_down()

    def page_up(self) -> None:
        """Move selection up by one page."""
        if self.view == View.INVENTORY:
            self.selected_index = max(0, self.selected_index - self.visible_rows)
        elif self.view == View.WHAT_IF:
            self.whatif_table_index = max(0, self.whatif_table_index - self.visible_rows)
        elif self.view == View.MODEL_FIT:
            self.fit_selected_index = max(0, self.fit_selected_index - self.visible_rows)
        elif self.view == View.MODEL_BROWSE:
            self.browse_page_up()

    def selected_node(self) -> WorkspaceComputeProfile | None:
        """Return the currently selected node, or None."""
        if not self.displayed_nodes:
            return None
        if self.selected_index >= len(self.displayed_nodes):
            self.selected_index = len(self.displayed_nodes) - 1
        return self.displayed_nodes[self.selected_index]

    def selected_candidate(self) -> CandidateCompute | None:
        """Return the currently selected fit candidate, or None."""
        if not self.fit_displayed_candidates:
            return None
        if self.fit_selected_index >= len(self.fit_displayed_candidates):
            self.fit_selected_index = len(self.fit_displayed_candidates) - 1
        return self.fit_displayed_candidates[self.fit_selected_index]

    # -- Browse helpers -------------------------------------------------

    def rebuild_browse_list(self) -> None:
        """Filter and rebuild the browse display list."""
        entries = list(self.browse_catalog)

        if self.browse_category_filter:
            entries = [e for e in entries if e.category == self.browse_category_filter]

        if self.browse_search:
            q = self.browse_search.lower()
            entries = [
                e for e in entries
                if q in e.model_id.lower()
                or q in e.provider.lower()
                or q in e.use_case.lower()
                or q in e.category.lower()
            ]

        # Sort: curated first, then by params descending
        entries.sort(key=lambda e: (e.discovered, -e.params_raw))
        self.browse_displayed = entries

    def clamp_browse_selection(self) -> None:
        """Keep browse_selected_index within bounds."""
        max_index = max(0, len(self.browse_displayed) - 1)
        self.browse_selected_index = max(0, min(self.browse_selected_index, max_index))

    def selected_browse_entry(self) -> CatalogEntry | None:
        """Return the currently selected browse entry, or None."""
        if not self.browse_displayed:
            return None
        if self.browse_selected_index >= len(self.browse_displayed):
            self.browse_selected_index = len(self.browse_displayed) - 1
        return self.browse_displayed[self.browse_selected_index]

    def browse_page_down(self) -> None:
        """Move browse selection down by one page."""
        self.browse_selected_index = min(
            self.browse_selected_index + self.visible_rows,
            max(0, len(self.browse_displayed) - 1),
        )
        self.clamp_browse_selection()

    def browse_page_up(self) -> None:
        """Move browse selection up by one page."""
        self.browse_selected_index = max(0, self.browse_selected_index - self.visible_rows)

    # -- What-if view helpers -------------------------------------------

    def whatif_gpu_nodes(self) -> list[WorkspaceComputeProfile]:
        """Return GPU nodes for the what-if view, filtered and sorted.

        This is the single source of truth — used by both
        ``whatif_candidate_count`` (for scroll clamping) and
        ``_render_whatif_view`` (for display).
        """
        if self.inventory is None:
            return []
        nodes = [n for n in self.inventory.compute if n.gpu_count > 0]
        nodes.sort(key=lambda n: (n.gpu_memory_gb or 0), reverse=True)
        return nodes

    def clamp_whatif_table(self) -> None:
        """Keep whatif_table_index within bounds."""
        count = self.whatif_candidate_count()
        if count == 0:
            self.whatif_table_index = 0
            return
        self.whatif_table_index = max(0, min(self.whatif_table_index, count - 1))

    def whatif_candidate_count(self) -> int:
        """Return the number of candidates in the what-if results table."""
        return len(self.whatif_gpu_nodes())
