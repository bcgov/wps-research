"""Per-fire state management for the web interface."""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import geopandas as gpd
import pandas as pd


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

    # Tracking
    previously_accepted: bool = False
    hidden: bool = False
    notes: str = ""
    agreement_pct: float = -1.0   # -1 = not computed
    console_log: list = field(default_factory=list)  # buffered output lines


class AppState:
    """Global application state shared across all routes."""

    def __init__(self):
        self.fires: dict[str, FireInfo] = {}

        # Loaded data
        self.gdf: Optional[gpd.GeoDataFrame] = None
        self.viirs_gdf: Optional[gpd.GeoDataFrame] = None

        # Raster info
        self.raster_path: str = ""
        self.polygon_file: str = ""
        self.raster_crs: str = ""
        self.raster_gt = None
        self.raster_W: int = 0
        self.raster_H: int = 0

        # Paths
        self.viirs_shp_dir: str = ""
        self.output_root: str = ""
        self.project_root: str = ""
        self.cli_script: str = ""

        # Defaults
        self.padding: float = 0.2
        self.sample_rate: float = 0.05
        self.min_samples: int = 500
        self.max_samples: int = 30000
        self.perimeter_mode: str = "viirs"

        # Authentication — two roles
        self.admin_password: Optional[str] = None
        self.user_password: Optional[str] = None

        # Sessions (cookie-based)
        self.sessions: dict = {}        # token → {role, username, ip, created_at}
        self.session_file: str = ""     # path for persistence

        # IP access control
        self.approved_ips: dict = {}    # {ip: {approved_by, timestamp}}
        self.blocked_ips: dict = {}     # {ip: {blocked_by, timestamp}}
        self.pending_ips: dict = {}     # {ip: {first_seen, last_seen}}
        self.ip_file: str = ""          # path to persistent YAML

        # Recommended settings (list of {min_ha, max_ha, params})
        self.recommended_settings: list = []

        # Queue tracking
        self.current_job: Optional[dict] = None   # {fire_numbe, client_ip, started_at}
        self.waiting_jobs: list = []               # [{fire_numbe, client_ip, queued_at}]

        # Batch mapping
        self.batch_status: Optional[dict] = None   # {running, total, completed, current_fire, errors}

    def init_fires_from_gdf(self):
        """Populate fires dict from the loaded GeoDataFrame."""
        import re as _re
        for _, row in self.gdf.iterrows():
            fn = str(row.get('FIRE_NUMBE', ''))
            if not fn:
                continue
            # Reject fire numbers that could cause path traversal
            if '..' in fn or '/' in fn or '\\' in fn or not _re.fullmatch(
                    r'[A-Za-z0-9_. -]+', fn):
                continue

            # Parse fire date
            raw = row.get('FIRE_DATE', '')
            try:
                if hasattr(raw, 'strftime') and pd.notna(raw):
                    fire_date = raw.strftime('%Y-%m-%d')
                elif raw and pd.notna(raw):
                    fire_date = str(raw).split()[0]
                else:
                    fire_date = ''
            except (ValueError, TypeError):
                fire_date = ''

            # Parse fire year
            try:
                fire_year = int(row.get('FIRE_YEAR', 0) or 0)
            except (ValueError, TypeError):
                fire_year = 0

            # Parse fire size
            try:
                fire_size = float(row.get('FIRE_SIZE_', 0) or 0)
            except (ValueError, TypeError):
                fire_size = 0.0

            # Check if already accepted (canonical dir has comparison PNG)
            status = FireStatus.PENDING
            notes = ''
            agreement = -1.0
            last_params = {}
            canon_dir = os.path.join(self.output_root, fn)
            if os.path.exists(os.path.join(canon_dir, f'{fn}_comparison.png')):
                status = FireStatus.ACCEPTED
                # Restore notes, agreement, params from saved YAML
                params_path = os.path.join(canon_dir, f'{fn}_params.yaml')
                if os.path.isfile(params_path):
                    try:
                        import yaml
                        with open(params_path) as _pf:
                            _pd = yaml.safe_load(_pf) or {}
                        fire_info = _pd.get('fire', {})
                        notes = fire_info.get('notes', '')
                        agreement = float(
                            fire_info.get('agreement_pct', -1) or -1)
                        # Restore mapping params
                        for section in ('tsne', 'hdbscan', 'random_forest',
                                        'sampling', 'crop'):
                            if section in _pd:
                                last_params[section] = _pd[section]
                    except Exception:
                        pass

            self.fires[fn] = FireInfo(
                fire_numbe=fn,
                fire_date=fire_date,
                fire_year=fire_year,
                fire_size_ha=round(fire_size, 1),
                status=status,
                notes=notes,
                agreement_pct=agreement,
                last_params=last_params,
            )
