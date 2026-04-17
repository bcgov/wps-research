"""Per-fire state for the parameter analyzer workflow.

Separate from the user-facing FireInfo/AppState so the analyzer is
self-contained and trivially removable. Persisted independently to
``analyzing_parameters/`` under the output root.
"""

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AnalyzerStatus(str, Enum):
    PENDING = "pending"        # never analyzed
    ANALYZING = "analyzing"    # worker is running
    ANALYZED = "analyzed"      # all N*M runs finished
    PARTIAL = "partial"        # cancelled or crashed partway
    ERROR = "error"            # fatal error


@dataclass
class AnalyzerRun:
    """One entry in a fire's analyzer run grid."""
    set_idx: int                       # 0..N-1 (parameter-set index)
    run_idx: int                       # 0..M-1 (HDBSCAN run within set)
    params: dict = field(default_factory=dict)
    padding_used: float = 0.0
    agreement_pct: float = -1.0
    ml_area_ha: float = -1.0
    status: str = "pending"            # pending|running|done|error|skipped
    error_msg: str = ""

    # Files in the fire's analyzer cache dir (relative names)
    thumb_rel: str = ""                # thumbnail PNG (always retained)
    comparison_rel: str = ""           # full comparison figure (retained)
    classified_rel: str = ""           # classification .bin (may be dropped
                                       # after thumbnail generation for disk)

    # Acceptance
    accepted: bool = False
    accept_id: str = ""                # e.g., 'run_0001' under canonical dir
    accepted_at: str = ""

    # When the run completed on disk (sidecar timestamp).
    timestamp: str = ""


@dataclass
class AnalyzerFireInfo:
    """Per-fire state in the analyzer."""
    fire_numbe: str
    status: AnalyzerStatus = AnalyzerStatus.PENDING
    error_msg: str = ""

    # Cache / canonical paths (set when the fire is first touched)
    cache_dir: str = ""                # .analyzer_cache/<FIRE>
    canonical_dir: str = ""            # analyzing_parameters/<FIRE>

    runs: list = field(default_factory=list)  # list[AnalyzerRun]

    # Live console buffer for UI
    console_log: list = field(default_factory=list)

    # Backdrop: the biggest-padding crop we have saved (for overlaying
    # all accepted perimeters on a common canvas).
    saved_max_padding: float = 0.0
    saved_max_crop_rel: str = ""       # filename under canonical_dir

    # Last update — helps UI decide when to refresh
    last_update: str = ""


@dataclass
class AnalyzerConfig:
    """Admin-defined configuration for the current analysis run.

    Persisted to analyzing_parameters/analyzer_config.yaml.
    """
    # List of param dicts, one per set. Each dict contains the full
    # set of pipeline parameters (padding, sample_rate, embed_bands,
    # tsne_*, rf_*, controlled_ratio, hdbscan_min_samples, ...).
    param_sets: list = field(default_factory=list)

    # How many HDBSCAN runs per parameter set (M).
    m_runs_per_set: int = 3

    # HDBSCAN variation across the M replicates:
    # Each run r in (0..M-1) uses
    #   hdbscan_min_samples = base + sign*level*m_run_jitter
    # with sign/level derived from r so replicates fan out around the
    # base (e.g. base, +1, -1, +2, -2 for step=1).
    # Set to 0 to disable jitter (all M runs use the base value — may
    # yield identical results if cuML HDBSCAN is deterministic on this
    # GPU for the given input).
    m_run_jitter: int = 1

    # Fires selected for the next run (fire_numbe list).
    selected_fires: list = field(default_factory=list)

    # Free-text description of this config.
    description: str = ""


class AnalyzerState:
    """Global analyzer state — parallel to AppState but strictly scoped."""

    def __init__(self):
        self.lock = threading.RLock()
        self.fires: dict[str, AnalyzerFireInfo] = {}
        self.config = AnalyzerConfig()

        # Paths (populated at init)
        self.analyzer_root: str = ""   # <out_dir>/analyzing_parameters
        self.cache_root: str = ""      # <analyzer_root>/.analyzer_cache
        self.config_file: str = ""     # <analyzer_root>/analyzer_config.yaml
        self.csv_file: str = ""        # <analyzer_root>/analyzer_accepted.csv

        # Worker control
        self.running: bool = False
        self.cancel_event = threading.Event()
        self.batch_status: Optional[dict] = None  # {total, completed, current_fire, errors}
        self.worker_thread: Optional[threading.Thread] = None


# Fixed CSV fieldnames for analyzer_accepted.csv.
# Superset of the user-facing accepted_params.csv columns so the two
# can be unioned later for ranking. Extra columns track which grid cell
# the row came from and enough fire characteristics for offline analysis.
ANALYZER_CSV_FIELDNAMES = [
    # Identity & acceptance
    'fire_numbe', 'accept_id', 'accepted_at',
    # Fire characteristics (for offline filtering/grouping)
    'fire_year', 'fire_size_ha', 'fire_region', 'fire_zone',
    'size_bucket_lo', 'size_bucket_hi',
    'perimeter_type', 'crop_w', 'crop_h',
    # Grid cell
    'set_idx', 'run_idx',
    # Run outcomes
    'agreement_pct', 'ml_area_ha',
    # Parameters (same names as accepted_params.csv where overlapping)
    'padding', 'sample_rate', 'min_samples', 'max_samples', 'seed',
    'embed_bands',
    'tsne_perplexity', 'tsne_learning_rate',
    'tsne_max_iter', 'tsne_init', 'tsne_n_components',
    'tsne_random_state',
    'controlled_ratio', 'hdbscan_min_samples',
    'rf_n_estimators', 'rf_max_depth', 'rf_max_features',
    'rf_random_state', 'contour_width',
]


# Parameters that, if changed, INVALIDATE a cached t-SNE+RF .npz state.
# Runs sharing the same values for all of these can reuse one cache file;
# differing values for controlled_ratio / hdbscan_min_samples / contour_width
# do not invalidate the cache (HDBSCAN-only re-run is valid).
TSNE_RF_CACHE_KEYS = (
    'padding',
    'sample_rate', 'min_samples', 'max_samples', 'seed',
    'embed_bands',
    'tsne_perplexity', 'tsne_learning_rate', 'tsne_max_iter',
    'tsne_init', 'tsne_n_components', 'tsne_random_state',
    'rf_n_estimators', 'rf_max_depth', 'rf_max_features',
    'rf_random_state',
)
