"""Per-fire state management for the VIIRS-web interface.

Differs from the polygon-driven sibling: fires are user-created from a
drawn bbox + date range. There is no polygon shapefile; fire identity
is the user-supplied name."""

import math
import os
import re
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

CONSOLE_LOG_MAX_LINES = 2000

# Regex for user-supplied fire names. Reused for path-traversal validation.
# Must start with alnum; remaining chars from a small safe set; max 64.
FIRE_NAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_. -]{0,63}$')


def _is_valid_fire_name(name: str) -> bool:
    """Return True iff *name* is a safe user-supplied fire identifier.

    Rejects empty, oversized, traversal substring (..), path separators,
    and anything outside the FIRE_NAME_RE alphabet."""
    if not isinstance(name, str) or not name:
        return False
    if '..' in name or '/' in name or '\\' in name:
        return False
    return bool(FIRE_NAME_RE.fullmatch(name))


class FireStatus(str, Enum):
    PENDING = "pending"
    PREPARING = "preparing"
    READY = "ready"
    MAPPING = "mapping"
    MAPPED = "mapped"
    ACCEPTED = "accepted"
    ERROR = "error"


@dataclass
class FireInfo:
    """Tracks per-fire state through the web workflow."""
    fire_numbe: str
    fire_date: str
    fire_year: int
    fire_size_ha: float

    status: FireStatus = FireStatus.PENDING
    error_msg: str = ""

    # Paths set during prepare
    cache_dir: str = ""
    crop_bin: str = ""
    hint_bin: str = ""
    perim_bin: str = ""
    viirs_bin: str = ""

    # Crop info
    crop_w: int = 0
    crop_h: int = 0
    padding_used: float = 0.0

    # Accumulation dates
    acc_start: str = ""
    acc_end: str = ""
    perimeter_type: str = ""

    # Sampling (computed from crop dims)
    sample_size: int = 0

    # Preview images available
    available_views: list = field(default_factory=list)

    # Mapping results
    last_comparison: str = ""
    last_params: dict = field(default_factory=dict)
    ml_area_ha: float = -1.0  # -1 = not computed

    # Tracking
    previously_accepted: bool = False
    hidden: bool = False
    notes: str = ""
    agreement_pct: float = -1.0   # -1 = not computed
    previously_accepted_agreement_pct: float = -1.0
    console_log: deque = field(
        default_factory=lambda: deque(maxlen=CONSOLE_LOG_MAX_LINES))
    serial_results: list = field(default_factory=list)
    serial_settings: list = field(default_factory=list)
    recommended_override: Optional[list] = None

    serial_canceled: bool = False
    serial_prev_status: Optional[FireStatus] = None
    serial_accept_promoted: bool = False
    serial_accept_event: Optional[threading.Event] = None

    # Live progress snapshot. For VIIRS prepare the stage values are
    # 'downloading_viirs', 'shapifying', 'accumulating', 'rasterizing',
    # 'cropping' (in order). For mapping/serial they are the fire_mapping
    # CLI's own stage names.
    progress: dict = field(default_factory=dict)

    last_cancel_reason: str = ""
    rebrush_dirty: bool = False

    # ---------- VIIRS-web additions ----------

    # User-drawn bbox in the raster's native CRS — VIIRS download AOI.
    bbox_native: Optional[tuple] = None       # (x_min, y_min, x_max, y_max)
    # Same bbox in WGS84 — for LAADS DAAC URL.
    bbox_wgs84: Optional[tuple] = None        # (W, S, E, N)
    # User-entered date range (YYYY-MM-DD strings).
    viirs_start_date: str = ""
    viirs_end_date: str = ""
    # Flips True the moment the fire reaches READY for the first time.
    # Flips False when a logged-in user opens its detail page. Drives the
    # "new" badge in the fire list.
    is_new: bool = False
    # Cancel signal for the VIIRS prepare worker. Set by the cancel
    # endpoint, polled by the worker between stages.
    cancel_event: Optional[threading.Event] = None


class AppState:
    """Global application state shared across all routes."""

    def __init__(self):
        self.lock = threading.RLock()
        self.fires: dict[str, FireInfo] = {}

        # Raster info (active year)
        self.raster_path: str = ""
        self.raster_crs: str = ""
        self.raster_gt = None
        self.raster_W: int = 0
        self.raster_H: int = 0

        # Paths
        self.output_root: str = ""
        self.project_root: str = ""
        self.cli_script: str = ""

        # Multi-year registry
        self.active_year: int = 0
        self.shared_root: str = ""
        self.rasters_by_year: dict = {}     # {year: abs raster path}
        self.outdirs_by_year: dict = {}     # {year: abs per-year out dir}

        # Overview PNG + sidecar JSON paths per year
        self.overview_png_by_year: dict = {}   # {year: abs png path}
        self.overview_meta_by_year: dict = {}  # {year: abs json path}

        # Year-wide VIIRS shp dirs (downloaded + shapified once at boot;
        # per-fire prepare just runs ``accumulate`` against this dir).
        self.viirs_shp_dirs_by_year: dict = {}  # {year: abs dir}

        self.settings_file: str = ""

        # Defaults
        self.padding: float = 0.1
        self.sample_rate: float = 0.05
        self.min_samples: int = 500
        self.max_samples: int = 30000

        # LAADS DAAC token (read once at startup)
        self.laads_token: str = ""
        self.viirs_download_workers: int = 16
        self.viirs_shapify_workers: int = 8
        self.viirs_concurrent_jobs: int = 1

        # Registry of running VIIRS prepare workers, keyed by fire name.
        # Mirrors _serial_procs / _rebrush_procs shape so cancel handlers
        # can use the same pattern. Each entry is the worker ``Thread``;
        # in-flight subprocesses (e.g. shapify) are tracked separately
        # via Popen handles in viirs_worker.
        self.viirs_jobs: dict = {}        # {fire_name: threading.Thread}
        self.viirs_subprocs: dict = {}    # {fire_name: subprocess.Popen}

        # Authentication — two roles
        self.admin_password: Optional[str] = None
        self.user_password: Optional[str] = None

        # Sessions
        self.sessions: dict = {}
        self.session_file: str = ""

        # IP access control
        self.approved_ips: dict = {}
        self.blocked_ips: dict = {}
        self.pending_ips: dict = {}
        self.ip_file: str = ""

        # Recommended settings
        self.recommended_settings: list = []
        self.k_runs_per_setting: int = 3
        self.k_jitter: int = 1

        # Queue tracking
        self.current_job: Optional[dict] = None
        self.waiting_jobs: list = []

        self.stage_timings: dict = {}
        self.notifications: dict = {}
        self.broadcast_cursor: dict = {}
        self.broadcast_counter: int = 0
        self.notification_counter: int = 0

        self.cache_retention: dict = {
            'max_gb': 20.0,
            'max_age_days': 30,
            'sweep_interval_hours': 6,
            'enabled': True,
        }
        self.cache_last_sweep: float = 0.0
        self.batch_status: Optional[dict] = None

        self.trust_proxy: bool = False
        self.insecure_no_auth: bool = False
        self.allowed_origins: set = set()

        self.login_attempts: dict = {}
        self.fire_state_load_failed: bool = False

        # ------------------------------------------------------------
        # Startup gate. The server starts listening immediately;
        # everything that depends on the year-wide VIIRS bootstrap
        # being complete is gated behind startup_complete. While
        # False, every request gets a plain "still starting up" page
        # regardless of auth/routing (see handlers/base.py).
        # startup_progress is a free-form dict the background
        # bootstrap thread updates so the placeholder page (and
        # /api/startup_status) can show real progress rather than a
        # blank wait.
        # ------------------------------------------------------------
        self.startup_complete: bool = False
        self.startup_error: str = ""
        self.startup_progress: dict = {
            'stage': 'starting',
            'detail': '',
        }

    # ------------------------------------------------------------------
    # Fire registry rebuild from disk (no polygon source)
    # ------------------------------------------------------------------

    def init_fires_from_disk(self):
        """Rebuild the fire registry from the per-year output dir.

        Sources:
          1. ``<output_root>/<NAME>/<NAME>_params.yaml`` — accepted fires.
          2. ``<output_root>/.web_cache/<NAME>/`` — in-flight or stale
             fires (no live worker on cold start, so these flip to ERROR).
        """
        import yaml

        if not self.output_root or not os.path.isdir(self.output_root):
            return

        # Pass 1: accepted fires (canonical dir)
        for name in sorted(os.listdir(self.output_root)):
            if not _is_valid_fire_name(name):
                continue
            fire_dir = os.path.join(self.output_root, name)
            if not os.path.isdir(fire_dir):
                continue
            comp = os.path.join(fire_dir, f'{name}_comparison.png')
            if not os.path.isfile(comp):
                continue
            fire = FireInfo(
                fire_numbe=name,
                fire_date='',
                fire_year=int(self.active_year or 0),
                fire_size_ha=0.0,
                status=FireStatus.ACCEPTED,
                perimeter_type='viirs',
            )
            params_path = os.path.join(fire_dir, f'{name}_params.yaml')
            if os.path.isfile(params_path):
                try:
                    with open(params_path, encoding='utf-8') as pf:
                        pdoc = yaml.safe_load(pf) or {}
                    fi = pdoc.get('fire') or {}
                    fire.fire_date = str(fi.get('fire_date', '') or '')
                    try:
                        fire.fire_size_ha = float(
                            fi.get('fire_size_ha', 0) or 0)
                    except (TypeError, ValueError):
                        fire.fire_size_ha = 0.0
                    if not math.isfinite(fire.fire_size_ha):
                        fire.fire_size_ha = 0.0
                    try:
                        fire.agreement_pct = float(
                            fi.get('agreement_pct', -1) or -1)
                    except (TypeError, ValueError):
                        fire.agreement_pct = -1.0
                    try:
                        fire.ml_area_ha = float(
                            fi.get('ml_area_ha', -1) or -1)
                    except (TypeError, ValueError):
                        fire.ml_area_ha = -1.0
                    fire.notes = str(fi.get('notes', '') or '')
                    inputs = pdoc.get('inputs') or {}
                    pt = str(inputs.get('perimeter_type', 'viirs')
                             or 'viirs')
                    fire.perimeter_type = pt or 'viirs'
                    acc = pdoc.get('accumulation') or {}
                    fire.viirs_start_date = str(
                        acc.get('start_date', '') or '')
                    fire.viirs_end_date = str(
                        acc.get('end_date', '') or '')
                    fire.acc_start = fire.viirs_start_date
                    fire.acc_end = fire.viirs_end_date
                    bbox = pdoc.get('bbox') or {}
                    if isinstance(bbox.get('native'), list) \
                            and len(bbox['native']) == 4:
                        fire.bbox_native = tuple(
                            float(v) for v in bbox['native'])
                    if isinstance(bbox.get('wgs84'), list) \
                            and len(bbox['wgs84']) == 4:
                        fire.bbox_wgs84 = tuple(
                            float(v) for v in bbox['wgs84'])
                    last_params = {}
                    _context_sections = {
                        'fire', 'run', 'inputs', 'crop', 'sampling',
                        'accumulation', 'bbox',
                    }
                    for section, payload in pdoc.items():
                        if section in _context_sections:
                            continue
                        if isinstance(payload, dict):
                            for k, v in payload.items():
                                last_params[k] = v
                    fire.last_params = last_params
                except Exception:
                    pass
            self.fires[name] = fire

        # Pass 2: in-flight / stale fires from .web_cache
        cache_root = os.path.join(self.output_root, '.web_cache')
        if os.path.isdir(cache_root):
            for name in sorted(os.listdir(cache_root)):
                if not _is_valid_fire_name(name):
                    continue
                if name in self.fires:
                    continue
                cache_dir = os.path.join(cache_root, name)
                if not os.path.isdir(cache_dir):
                    continue
                fire = FireInfo(
                    fire_numbe=name,
                    fire_date='',
                    fire_year=int(self.active_year or 0),
                    fire_size_ha=0.0,
                    status=FireStatus.ERROR,
                    error_msg='interrupted; retry create',
                    cache_dir=cache_dir,
                    perimeter_type='viirs',
                )
                self.fires[name] = fire

        # Restore notes overlay if present
        notes_path = os.path.join(self.output_root, 'notes.yaml')
        if os.path.isfile(notes_path):
            try:
                with open(notes_path, encoding='utf-8') as nf:
                    notes_data = yaml.safe_load(nf) or {}
                for fn, note_text in notes_data.items():
                    if fn in self.fires and note_text:
                        self.fires[fn].notes = str(note_text)
            except Exception:
                pass
