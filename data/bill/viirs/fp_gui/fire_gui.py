"""
viirs/fp_gui/fire_gui.py

RENDERING RESOURCE KNOBS
=========================
1. config.py -- all tuneable constants (edit via Config dialog)
2. fire_map_canvas.py -- figsize / dpi / blitting
3. fire_data_manager.py -- precompute_frames / numba cache
"""

import os
import re as _re
import sys
import threading
import datetime
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import date, datetime as _datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import config as cfg
from fire_data_manager import FireDataManager
from raster_loader import RasterLoader
from fire_map_canvas import FireMapCanvas, NAV_ZOOM_IN, NAV_ZOOM_OUT, NAV_PAN
from fire_animation_controller import FireAnimationController
from config_dialog import ConfigDialog
from file_browser import browse_directory, browse_file


def _auto_select_bands(band_names: list) -> list:
    """Choose up to 3 bands to display for a new raster load.

    Strategy (evaluated in order):
    1. If bands share an underscore-based suffix and any group uses a known
       "post" keyword (post, after, later, new), return the first 3 of that group.
    2. If multiple suffix groups exist, return the first 3 of the first group
       that is NOT a known "pre" keyword (pre, before, prior, old, prev).
    3. Fallback: first min(n, 3) bands (1-based).

    This handles names like B12_pre / B12_post / B12_diff as well as
    Band12_before / Band12_after, etc.  When no consistent grouping is
    detected the function degrades to the plain default.
    """
    n = len(band_names)
    if n == 0:
        return []

    _POST = {'post', 'after', 'later', 'new'}
    _PRE  = {'pre', 'before', 'prior', 'old', 'prev', 'previous'}

    # Build an ordered list of unique suffixes and their 1-based band indices.
    suffix_order: list = []
    suffix_indices: dict = {}
    has_suffix = False

    for i, name in enumerate(band_names):
        if '_' in name:
            suffix = name.rsplit('_', 1)[1].lower()
            has_suffix = True
            if suffix not in suffix_indices:
                suffix_order.append(suffix)
                suffix_indices[suffix] = []
            suffix_indices[suffix].append(i + 1)

    # Only use suffix logic when there are at least two distinct groups,
    # meaning the bands really are "paired" (pre/post/diff/…).
    if has_suffix and len(suffix_indices) > 1:
        for kw in _POST:
            if kw in suffix_indices:
                return suffix_indices[kw][:3]
        for suffix in suffix_order:
            if suffix not in _PRE:
                return suffix_indices[suffix][:3]

    return list(range(1, min(n, 3) + 1))


class FireAccumulationGUI:
    """
    Top-level GUI for the VIIRS fire pixel accumulation viewer.

    Layout (vertical, top-to-bottom):
        1. Data Loader    -- raster load + shapefile status
        2. Date & Download -- date range, apply filter, download
        3. Ref & Output   -- reference raster, auto-computed output
        4. Navigation     -- pan/zoom, layer toggles
    """

    def __init__(self, raster_path: Optional[str] = None):
        self._root = tk.Tk()
        self._root.title("VIIRS Fire Pixel Accumulation Viewer")
        self._root.geometry("1600x1000")
        try:
            self._root.state("zoomed")
        except tk.TclError:
            try:
                self._root.attributes("-zoomed", True)
            except tk.TclError:
                pass
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Components
        self._data_mgr = FireDataManager()
        self._raster_loader = RasterLoader()
        self._canvas = None
        self._animator = None

        # State
        self._raster_loaded = False
        self._crs_string = ""
        self._acc_thread = None
        self._download_thread: Optional[threading.Thread] = None
        self._download_cancel = threading.Event()
        self._download_executor: Optional[ThreadPoolExecutor] = None
        self._shapify_proc = None
        self._post_download_callback = None  # callable run after download finishes

        # Working directory (set when raster is loaded)
        self._working_dir: Optional[str] = None

        # Full paths (UI shows basenames only)
        self._raster_full_path: str = ""
        self._base_raster_full_path: str = ""
        self._viirs_dir: str = ""
        self._acc_output_full_path: str = ""

        # Token
        self._token: str = ""
        self._token_loaded = False
        try:
            with open("/data/.tokens/laads", "r") as fh:
                self._token = fh.read().strip()
                self._token_loaded = bool(self._token)
        except Exception:
            pass

        # Raster CRS info for download
        self._ref_wkt: Optional[str] = None
        self._ref_epsg: Optional[int] = None

        # tk variables
        self._shapefile_dir_var = tk.StringVar()
        self._raster_path_var = tk.StringVar()
        self._start_date_var = tk.StringVar()
        self._end_date_var = tk.StringVar()
        self._interval_var = tk.IntVar(value=cfg.DEFAULT_ANIMATION_INTERVAL_MS)
        self._status_var = tk.StringVar(value="Ready. Load a raster to begin.")
        self._date_label_var = tk.StringVar(value="Date: \u2014")
        self._frame_label_var = tk.StringVar(value="Frame: 0 / 0")
        self._pixel_count_var = tk.StringVar(value="Pixels: 0")
        self._viewport_pixel_var = tk.StringVar(value="In view: 0")
        self._crs_var = tk.StringVar(value="EPSG: \u2014")
        self._updating_slider = False

        # Accumulate + Rasterize variables
        self._base_raster_var = tk.StringVar()
        self._acc_output_dir_var = tk.StringVar()

        # Slider throttle
        self._slider_throttle_id = None
        self._slider_release_id = None
        self._slider_dragging = False
        self._slider_pending_val = None
        self._SLIDER_THROTTLE_MS = 30
        self._SLIDER_RELEASE_MS = 150
        self._SLIDER_MAX_POINTS = 5000

        # Nav buttons
        self._nav_buttons = {}
        self._active_nav = NAV_PAN

        # Band selection (1-based indices, in user-chosen order)
        self._selected_bands: list = []

        # Preserved slider position (survives reload)
        self._preserved_slider_index: Optional[int] = None
        self._preserved_slider_date: Optional[date] = None

        self._build_ui()

        self._animator = FireAnimationController(
            self._root,
            on_frame=self._on_animation_frame,
            on_finished=self._on_animation_finished,
        )

        self._canvas.set_on_viewport_changed(self._on_viewport_changed)

        if raster_path:
            self._root.after(0, lambda p=raster_path: self._load_startup_raster(p))

    # ==================================================================
    # UI construction — compact 3-row layout for maximum canvas space
    #
    #   Row 1: [Config] Raster:[name][Load] | Shapefiles: [status]
    #   Row 2: Start[__] End[__] [Apply] [Download] | Playback controls
    #   Row 3: LEFT: Ref:[name][Browse]  |  RIGHT: Pan Zoom+ Zoom- Home Fire Bg
    # ==================================================================

    def _build_ui(self):
        _font = ("TkDefaultFont", 9)
        _bg = "#d9d9d9"
        try:
            _bg = self._root.cget("background")
        except Exception:
            pass
        _active_bg = "#b0c4ff"

        ctrl = ttk.Frame(self._root, padding=(4, 2))
        ctrl.pack(side=tk.TOP, fill=tk.X)

        # ==============================================================
        # ROW 1 — Data Loader: Raster + Shapefiles on ONE line
        # ==============================================================
        row1 = ttk.Frame(ctrl)
        row1.pack(fill=tk.X, pady=(0, 1))

        ttk.Button(row1, text="\u2699", command=self._open_config,
                   width=3).pack(side=tk.LEFT, padx=(0, 4))

        ttk.Label(row1, text="Raster:", font=_font).pack(side=tk.LEFT)
        raster_entry = ttk.Entry(row1, textvariable=self._raster_path_var,
                                 width=28)
        raster_entry.pack(side=tk.LEFT, padx=2)
        raster_entry.bind("<Return>", lambda _: self._load_raster_from_entry())
        ttk.Button(row1, text="Browse",
                   command=self._browse_raster).pack(side=tk.LEFT)

        ttk.Separator(row1, orient=tk.VERTICAL).pack(
            side=tk.LEFT, padx=6, fill=tk.Y, pady=1)

        ttk.Label(row1, text="Shapefiles:", font=_font).pack(side=tk.LEFT)
        self._shp_status_label = tk.Label(
            row1, text="not found", fg="gray", font=_font)
        self._shp_status_label.pack(side=tk.LEFT, padx=4)

        # -- Sentinel-2 Fire Mapping (right side of row 1) --
        self._fire_mapping_btn = tk.Button(
            row1, text="Sentinel-2 Fire Mapping", bg="#E53935", fg="white",
            font=_font + ("bold",), activebackground="#B71C1C",
            cursor="hand2", command=self._launch_fire_mapping,
        )
        self._fire_mapping_btn.pack(side=tk.RIGHT, padx=(8, 0))

        # ==============================================================
        # ROW 2 — Date / Download  |  Playback (all one line)
        # ==============================================================
        row2 = ttk.Frame(ctrl)
        row2.pack(fill=tk.X, pady=1)

        # -- Date / Download cluster --
        ttk.Label(row2, text="Start:", font=_font).pack(side=tk.LEFT)
        _start_entry = ttk.Entry(row2, textvariable=self._start_date_var, width=11)
        _start_entry.pack(side=tk.LEFT, padx=2)
        _start_entry.bind("<FocusOut>", self._apply_date_filter_silent)
        _start_entry.bind("<Return>", lambda _: self._apply_date_filter())
        ttk.Label(row2, text="End:", font=_font).pack(side=tk.LEFT, padx=(4, 0))
        _end_entry = ttk.Entry(row2, textvariable=self._end_date_var, width=11)
        _end_entry.pack(side=tk.LEFT, padx=2)
        _end_entry.bind("<FocusOut>", self._apply_date_filter_silent)
        _end_entry.bind("<Return>", lambda _: self._apply_date_filter())

        self._download_btn = tk.Button(
            row2, text="\u2b07 Download", bg="#2196F3", fg="white",
            font=_font + ("bold",), activebackground="#1565C0",
            command=self._start_download,
        )
        self._download_btn.pack(side=tk.LEFT, padx=(2, 0))

        # Wide gap to visually separate Download from Playback
        ttk.Frame(row2, width=16).pack(side=tk.LEFT)
        ttk.Separator(row2, orient=tk.VERTICAL).pack(
            side=tk.LEFT, padx=2, fill=tk.Y, pady=1)
        ttk.Frame(row2, width=16).pack(side=tk.LEFT)

        # -- Playback cluster --
        self._play_btn = ttk.Button(row2, text="\u25b6 Play",
                                    command=self._toggle_play, width=7)
        self._play_btn.pack(side=tk.LEFT, padx=1)
        ttk.Button(row2, text="\u23ee", command=self._reset_animation,
                   width=2).pack(side=tk.LEFT, padx=1)

        ttk.Separator(row2, orient=tk.VERTICAL).pack(
            side=tk.LEFT, padx=3, fill=tk.Y, pady=1)

        ttk.Button(row2, text="\u2190",
                   command=self._skip_back_n, width=2).pack(side=tk.LEFT, padx=1)
        self._skip_n_var = tk.IntVar(value=1)
        ttk.Spinbox(row2, from_=1, to=999, increment=1,
                     textvariable=self._skip_n_var, width=3).pack(
            side=tk.LEFT, padx=1)
        ttk.Button(row2, text="\u2192",
                   command=self._skip_fwd_n, width=2).pack(side=tk.LEFT, padx=1)

        ttk.Separator(row2, orient=tk.VERTICAL).pack(
            side=tk.LEFT, padx=3, fill=tk.Y, pady=1)

        ttk.Label(row2, text="ms:", font=_font).pack(side=tk.LEFT)
        ttk.Spinbox(row2, from_=50, to=5000, increment=50,
                     textvariable=self._interval_var, width=5,
                     command=self._update_speed).pack(side=tk.LEFT, padx=2)

        ttk.Separator(row2, orient=tk.VERTICAL).pack(
            side=tk.LEFT, padx=3, fill=tk.Y, pady=1)

        self._frame_slider = ttk.Scale(
            row2, from_=0, to=1, orient=tk.HORIZONTAL,
            command=self._on_slider)
        self._frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self._frame_slider.bind("<Button-1>", self._slider_snap_click)

        # ==============================================================
        # ROW 3 — LEFT: Accumulate & Rasterize  |  RIGHT: Navigation
        # ==============================================================
        row3 = ttk.Frame(ctrl)
        row3.pack(fill=tk.X, pady=(1, 0))
        row3.columnconfigure(0, weight=1)
        row3.columnconfigure(1, weight=0)

        # -- LEFT: Accumulate & Rasterize --
        acc_frame = ttk.Frame(row3)
        acc_frame.grid(row=0, column=0, sticky="ew")

        ttk.Label(acc_frame, text="Ref:", font=_font).pack(side=tk.LEFT)
        ref_entry = ttk.Entry(acc_frame, textvariable=self._base_raster_var,
                              width=20)
        ref_entry.pack(side=tk.LEFT, padx=2)
        ref_entry.bind("<Return>", lambda _: self._load_ref_from_entry())
        ttk.Button(acc_frame, text="Browse",
                   command=self._browse_base_raster).pack(
            side=tk.LEFT, padx=(0, 4))

        ttk.Separator(acc_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, padx=6, fill=tk.Y, pady=1)

        # -- RIGHT: Navigation + layer toggles --
        nav_frame = ttk.Frame(row3)
        nav_frame.grid(row=0, column=1, sticky="e")

        def _nav_btn(text, mode):
            btn = tk.Button(
                nav_frame, text=text, relief=tk.RAISED, bd=1,
                padx=5, pady=0, font=_font, bg=_bg,
                activebackground=_active_bg, cursor="hand2",
                command=lambda m=mode: self._set_nav_mode(m),
            )
            btn.pack(side=tk.LEFT, padx=1)
            self._nav_buttons[mode] = btn

        self._band_btn = tk.Button(
            nav_frame, text="Band", relief=tk.RAISED, bd=1,
            padx=5, pady=0, font=_font, bg=_bg,
            activebackground=_active_bg, cursor="hand2",
            command=self._show_band_selector,
        )
        self._band_btn.pack(side=tk.LEFT, padx=1)

        ttk.Separator(nav_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, padx=4, fill=tk.Y, pady=1)

        _nav_btn("Pan",    NAV_PAN)

        ttk.Separator(nav_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, padx=4, fill=tk.Y, pady=1)

        _nav_btn("Zoom+",  NAV_ZOOM_IN)
        _nav_btn("Zoom\u2212", NAV_ZOOM_OUT)

        tk.Button(
            nav_frame, text="\u2302", relief=tk.RAISED, bd=1,
            padx=5, pady=0, font=_font, bg=_bg, cursor="hand2",
            command=self._nav_home,
        ).pack(side=tk.LEFT, padx=(4, 1))

        ttk.Separator(nav_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, padx=4, fill=tk.Y, pady=1)

        self._show_fire_var = tk.BooleanVar(value=True)
        self._fire_chk = tk.Checkbutton(
            nav_frame, text="Fire Pixels",
            variable=self._show_fire_var,
            command=self._toggle_fire,
            indicatoron=True, selectcolor="#22cc22",
            bg=_bg, activebackground=_bg, font=_font,
        )
        self._fire_chk.pack(side=tk.LEFT, padx=1)

        self._show_raster_var = tk.BooleanVar(value=True)
        self._raster_chk = tk.Checkbutton(
            nav_frame, text="Background Image",
            variable=self._show_raster_var,
            command=self._toggle_raster,
            indicatoron=True, selectcolor="#22cc22",
            bg=_bg, activebackground=_bg, font=_font,
        )
        self._raster_chk.pack(side=tk.LEFT, padx=1)

        def _rc(*_):
            c = "#22cc22" if self._show_raster_var.get() else "#111111"
            self._raster_chk.configure(selectcolor=c)
        def _fc(*_):
            c = "#22cc22" if self._show_fire_var.get() else "#111111"
            self._fire_chk.configure(selectcolor=c)
        self._show_raster_var.trace_add("write", _rc)
        self._show_fire_var.trace_add("write", _fc)

        self._highlight_nav_button(NAV_PAN)

        # ==============================================================
        # STATUS BAR (bottom)
        # ==============================================================
        status = ttk.Frame(self._root, padding=(4, 2))
        status.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(status, textvariable=self._status_var).pack(side=tk.LEFT)
        ttk.Label(status, textvariable=self._viewport_pixel_var).pack(
            side=tk.RIGHT, padx=12)
        ttk.Label(status, textvariable=self._pixel_count_var).pack(
            side=tk.RIGHT, padx=12)
        ttk.Label(status, textvariable=self._frame_label_var).pack(
            side=tk.RIGHT, padx=12)
        ttk.Label(status, textvariable=self._date_label_var).pack(
            side=tk.RIGHT, padx=12)
        ttk.Label(status, textvariable=self._crs_var,
                  font=("TkDefaultFont", 9, "bold")).pack(
            side=tk.RIGHT, padx=12)

        # ==============================================================
        # CANVAS (fills all remaining space)
        # ==============================================================
        canvas_frame = ttk.Frame(self._root)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        self._canvas = FireMapCanvas(canvas_frame, figsize=(12, 7), dpi=100)

    def _slider_snap_click(self, event):
        slider = self._frame_slider
        width = slider.winfo_width()
        if width <= 0:
            return
        x = max(0, min(event.x, width))
        from_val = float(slider.cget("from"))
        to_val = float(slider.cget("to"))
        fraction = x / width
        value = from_val + fraction * (to_val - from_val)
        slider.set(value)
        return "break"

    # ==================================================================
    # Token management
    # ==================================================================

    def _prompt_token(self) -> bool:
        """Show popup to get LAADS DAAC token. Returns True if token was set."""
        popup = tk.Toplevel(self._root)
        popup.title("LAADS DAAC Token Required")
        popup.transient(self._root)
        popup.grab_set()

        w, h = 500, 200
        sx = self._root.winfo_screenwidth()
        sy = self._root.winfo_screenheight()
        popup.geometry(f"{w}x{h}+{(sx-w)//2}+{(sy-h)//2}")
        popup.resizable(False, False)

        result = [False]

        ttk.Label(popup, text="Token not loaded",
                  font=("TkDefaultFont", 11, "bold")).pack(pady=(12, 4))
        ttk.Label(popup, text=(
            "Paste your LAADS DAAC token below, or place it at:\n"
            "/data/.tokens/laads\n\n"
            "To get a token, see README.md."
        ), justify=tk.CENTER).pack(padx=12, pady=4)

        f = ttk.Frame(popup)
        f.pack(fill=tk.X, padx=12, pady=4)
        ttk.Label(f, text="Token:").pack(side=tk.LEFT)
        paste_var = tk.StringVar()
        ttk.Entry(f, textvariable=paste_var, width=40, show="*").pack(
            side=tk.LEFT, padx=4, fill=tk.X, expand=True)

        def _apply():
            val = paste_var.get().strip()
            if val:
                self._token = val
                self._token_loaded = True
                result[0] = True
            popup.destroy()

        btn_f = ttk.Frame(popup)
        btn_f.pack(pady=8)
        tk.Button(
            btn_f, text="  Set Token  ", bg="#4CAF50", fg="white",
            font=("TkDefaultFont", 9, "bold"), activebackground="#388E3C",
            command=_apply,
        ).pack(side=tk.LEFT, padx=8)
        tk.Button(
            btn_f, text="  Cancel  ", bg="#F44336", fg="white",
            font=("TkDefaultFont", 9, "bold"), activebackground="#C62828",
            command=popup.destroy,
        ).pack(side=tk.LEFT, padx=8)

        popup.wait_window()
        return result[0]

    # ==================================================================
    # Config dialog
    # ==================================================================

    def _open_config(self):
        dialog = ConfigDialog(self._root)
        if not dialog.applied:
            return
        self._interval_var.set(cfg.DEFAULT_ANIMATION_INTERVAL_MS)
        if self._animator:
            self._animator.interval_ms = cfg.DEFAULT_ANIMATION_INTERVAL_MS
        if self._canvas:
            self._canvas.scatter_size = cfg.DEFAULT_SCATTER_SIZE
        if self._animator and self._animator.current_date:
            self._on_animation_frame(self._animator.current_date)

    # ==================================================================
    # Sentinel-2 Fire Mapping
    # ==================================================================

    def _ask_download_and_launch(self, message: str) -> bool:
        """Show a dialog with the message and a 'Download and Launch Again'
        button.  Returns True if the user clicks the button, False on Cancel."""
        result = [False]
        popup = tk.Toplevel(self._root)
        popup.title("Fire Mapping")
        popup.transient(self._root)
        popup.grab_set()

        w, h = 460, 200
        sx = self._root.winfo_screenwidth()
        sy = self._root.winfo_screenheight()
        popup.geometry(f"{w}x{h}+{(sx - w) // 2}+{(sy - h) // 2}")
        popup.resizable(False, False)

        ttk.Label(popup, text=message, wraplength=420,
                  justify=tk.LEFT).pack(padx=16, pady=(16, 10))

        btn_f = ttk.Frame(popup)
        btn_f.pack(pady=10)

        def _go():
            result[0] = True
            popup.destroy()

        tk.Button(
            btn_f, text="  Download and Launch Again  ",
            bg="#4CAF50", fg="white",
            font=("TkDefaultFont", 10, "bold"),
            activebackground="#388E3C", command=_go,
        ).pack(side=tk.LEFT, padx=8)
        tk.Button(
            btn_f, text="  Cancel  ",
            bg="#F44336", fg="white",
            font=("TkDefaultFont", 9, "bold"),
            activebackground="#C62828", command=popup.destroy,
        ).pack(side=tk.LEFT, padx=8)

        popup.wait_window()
        return result[0]

    def _launch_fire_mapping(self):
        """Launch fire_mapping with the main raster and the best-match .bin."""
        # 1. Check raster is loaded
        if not self._raster_loaded or not self._raster_full_path:
            messagebox.showerror(
                "Fire Mapping", "Please load a raster image first.")
            return

        # 2. Check shapefiles are loaded (green status)
        if self._shp_status_label.cget("text") != "loaded":
            messagebox.showerror(
                "Fire Mapping",
                "Shapefiles must be detected and loaded (green status) "
                "before launching fire mapping.")
            return

        # 3. Read start/end dates from the boxes
        start_str = self._start_date_var.get().strip()
        end_str = self._end_date_var.get().strip()
        if not start_str or not end_str:
            messagebox.showerror(
                "Fire Mapping",
                "Start and End dates must be set before launching fire mapping.")
            return

        try:
            box_start = date.fromisoformat(start_str)
            box_end = date.fromisoformat(end_str)
        except ValueError:
            messagebox.showerror(
                "Fire Mapping",
                "Invalid date format. Use YYYY-MM-DD.")
            return

        # 4. Search for _ACCUMULATED folder in the main raster's directory
        raster_dir = os.path.dirname(self._raster_full_path)
        raster_base = os.path.splitext(
            os.path.basename(self._raster_full_path))[0]

        start_compact = start_str.replace("-", "")
        pattern = _re.compile(
            rf'^{_re.escape(raster_base)}_(\d{{8}})_(\d{{8}})_ACCUMULATED$')

        candidates = []
        if os.path.isdir(raster_dir):
            for name in os.listdir(raster_dir):
                full = os.path.join(raster_dir, name)
                if os.path.isdir(full):
                    m = pattern.match(name)
                    if m:
                        folder_start = m.group(1)
                        folder_end = m.group(2)
                        # Start must match, end must be >= box end
                        if folder_start == start_compact:
                            try:
                                f_end = date(
                                    int(folder_end[:4]),
                                    int(folder_end[4:6]),
                                    int(folder_end[6:8]))
                                if f_end >= box_end:
                                    candidates.append((name, f_end, full))
                            except ValueError:
                                continue

        if not candidates:
            if self._ask_download_and_launch(
                f"Directory not found: no ACCUMULATION with "
                f"start = {start_str} and end >= {end_str}\n\n"
                f"Raster directory: {raster_dir}"):
                self._post_download_callback = self._launch_fire_mapping
                self._start_download()
            return

        # Pick the folder with the smallest end date that still covers box_end
        candidates.sort(key=lambda c: c[1])
        acc_folder_path = candidates[0][2]

        # 5. Find best .bin file: end date closest to box_end but not exceeding
        #    File pattern: VIIRS_VNP14IMG_{YYYYMMDDTHHMM}_{YYYYMMDDTHHMM}.bin
        bin_pattern = _re.compile(
            r'^VIIRS_VNP14IMG_(\d{8}T\d{4})_(\d{8}T\d{4})\.bin$')

        bin_candidates = []
        for fname in os.listdir(acc_folder_path):
            bm = bin_pattern.match(fname)
            if bm:
                end_tag = bm.group(2)  # e.g. "20250902T1430"
                try:
                    bin_end_dt = _datetime.strptime(end_tag, "%Y%m%dT%H%M")
                    bin_end_date = bin_end_dt.date()
                    if bin_end_date <= box_end:
                        bin_candidates.append(
                            (fname, bin_end_dt,
                             os.path.join(acc_folder_path, fname)))
                except ValueError:
                    continue

        if not bin_candidates:
            if self._ask_download_and_launch(
                f"No rasterized .bin file found with end date <= {end_str} "
                f"in:\n{acc_folder_path}"):
                self._post_download_callback = self._launch_fire_mapping
                self._start_download()
            return

        # Pick the .bin with end datetime closest to box_end
        bin_candidates.sort(key=lambda c: c[1], reverse=True)
        best_bin_path = bin_candidates[0][2]

        # 6. Launch fire_mapping
        #    Derive repo root from this file: fp_gui/ -> viirs/ -> bill/ -> data/ -> wps-research/
        repo_root = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.pardir, os.pardir, os.pardir, os.pardir))
        fire_mapping_script = os.path.join(
            repo_root, "py", "fire_mapping", "fire_mapping.py")

        if not os.path.isfile(fire_mapping_script):
            messagebox.showerror(
                "Fire Mapping",
                f"fire_mapping.py not found at:\n{fire_mapping_script}")
            return

        try:
            subprocess.Popen(
                ["python3", fire_mapping_script,
                 self._raster_full_path, best_bin_path])
            self._status_var.set(
                f"Launched fire_mapping with {os.path.basename(best_bin_path)}")
        except Exception as e:
            messagebox.showerror("Fire Mapping", f"Failed to launch: {e}")

    # ==================================================================
    # Nav tool switching
    # ==================================================================

    def _set_nav_mode(self, mode):
        self._active_nav = mode
        self._highlight_nav_button(mode)
        if self._canvas:
            self._canvas.set_nav_mode(mode)

    def _highlight_nav_button(self, active):
        _bg = "#d9d9d9"
        try:
            _bg = self._root.cget("background")
        except Exception:
            pass
        for mode, btn in self._nav_buttons.items():
            if mode == active:
                btn.configure(relief=tk.SUNKEN, bg="#b0c4ff")
            else:
                btn.configure(relief=tk.RAISED, bg=_bg)

    def _nav_home(self):
        if self._canvas:
            self._canvas.view_home()

    # ==================================================================
    # Viewport callback
    # ==================================================================

    def _on_viewport_changed(self):
        self._update_viewport_pixel_count()

    def _update_viewport_pixel_count(self):
        if self._canvas:
            n = self._canvas.count_visible_pixels()
            self._viewport_pixel_var.set(f"In view: {n}")

    # ==================================================================
    # Date validation helper (specific error messages)
    # ==================================================================

    def _validate_dates(self) -> tuple:
        """
        Validate start/end date fields.
        Returns (start_str, end_str, start_date_or_dt, end_date_or_dt) on success.
        Returns None on failure (after showing a messagebox).
        """
        start_str = self._start_date_var.get().strip()
        end_str = self._end_date_var.get().strip()

        if not start_str and not end_str:
            messagebox.showerror("Missing Dates",
                                 "Both Start and End dates are empty.\n"
                                 "Please enter dates in YYYY-MM-DD format.")
            return None
        if not start_str:
            messagebox.showerror("Missing Start Date",
                                 "Start date is empty.\n"
                                 "Please enter a start date (YYYY-MM-DD).")
            return None
        if not end_str:
            messagebox.showerror("Missing End Date",
                                 "End date is empty.\n"
                                 "Please enter an end date (YYYY-MM-DD).")
            return None

        try:
            start = date.fromisoformat(start_str)
        except ValueError:
            messagebox.showerror("Invalid Start Date",
                                 f"Start date '{start_str}' is not valid.\n"
                                 "Use YYYY-MM-DD format (e.g. 2025-09-01).")
            return None

        try:
            end = date.fromisoformat(end_str)
        except ValueError:
            messagebox.showerror("Invalid End Date",
                                 f"End date '{end_str}' is not valid.\n"
                                 "Use YYYY-MM-DD format (e.g. 2025-09-30).")
            return None

        if start > end:
            messagebox.showerror("Invalid Date Range",
                                 f"Start date ({start_str}) is after "
                                 f"end date ({end_str}).")
            return None

        return start_str, end_str, start, end

    # ==================================================================
    # Working directory & base raster sync
    # ==================================================================

    def _set_working_dir(self, raster_path: str):
        """Set the working directory to the parent of the raster file."""
        self._working_dir = os.path.dirname(os.path.abspath(raster_path))

    def _sync_base_raster(self):
        """One-way: mirror the visualization raster to Reference Raster."""
        if self._raster_full_path:
            self._base_raster_full_path = self._raster_full_path
            self._base_raster_var.set(os.path.basename(self._raster_full_path))

    def _get_initial_browse_dir(self) -> str:
        """Return working dir for browse dialogs, or home."""
        if self._working_dir and os.path.isdir(self._working_dir):
            return self._working_dir
        return os.path.expanduser("~")

    def _compute_viirs_dir(self):
        """Compute the _VIIRS download folder path from the raster path."""
        if not self._raster_full_path:
            self._viirs_dir = ""
            return
        raster_dir = os.path.dirname(self._raster_full_path)
        raster_base = os.path.splitext(os.path.basename(
            self._raster_full_path))[0]
        self._viirs_dir = os.path.join(raster_dir, f"{raster_base}_VIIRS")

    def _check_viirs_folder(self):
        """Check for _VIIRS folder and auto-load shapefiles if found."""
        if not self._viirs_dir:
            self._update_shp_status(False)
            return

        if os.path.isdir(self._viirs_dir):
            self._shapefile_dir_var.set(self._viirs_dir)
            self._load_shapefiles()
            # load_shapefiles updates status on success; if it failed
            # (empty dir), the status will be set there
        else:
            self._update_shp_status(False)

    def _update_shp_status(self, loaded: bool):
        """Update the shapefile status indicator and clear fire-pixel info
        when shapefiles are not loaded."""
        if loaded:
            self._shp_status_label.configure(text="loaded", fg="#22cc22")
        else:
            self._shp_status_label.configure(text="not found", fg="gray")
            # Clear all fire-pixel-related display info
            self._start_date_var.set("")
            self._end_date_var.set("")
            self._date_label_var.set("Date: \u2014")
            self._frame_label_var.set("Frame: 0 / 0")
            self._pixel_count_var.set("Pixels: 0")
            self._viewport_pixel_var.set("In view: 0")
            self._frame_slider.set(0)
            self._frame_slider.configure(to=0)
            if self._animator:
                self._animator.stop()
                self._play_btn.config(text="\u25b6  Play")

    def _update_acc_output_name(self):
        """Auto-compute accumulation output directory name from ref + dates."""
        ref_path = self._base_raster_full_path
        start = self._start_date_var.get().strip()
        end = self._end_date_var.get().strip()

        if not ref_path or not start or not end:
            self._acc_output_dir_var.set("")
            self._acc_output_full_path = ""
            return

        try:
            # Validate dates before computing
            date.fromisoformat(start)
            date.fromisoformat(end)
        except ValueError:
            return

        ref_base = os.path.splitext(os.path.basename(ref_path))[0]
        ref_dir = os.path.dirname(ref_path)
        start_compact = start.replace("-", "")
        end_compact = end.replace("-", "")
        folder_name = f"{ref_base}_{start_compact}_{end_compact}_ACCUMULATED"
        self._acc_output_full_path = os.path.join(ref_dir, folder_name)
        self._acc_output_dir_var.set(folder_name)

    def _find_accumulated_folders(self, ref_dir, ref_base):
        """
        Find existing _ACCUMULATED folders for a given reference basename.
        Returns list of (folder_name, start_compact, end_compact).
        """
        import re as _re
        results = []
        pattern = _re.compile(
            rf'^{_re.escape(ref_base)}_(\d{{8}})_(\d{{8}})_ACCUMULATED$')
        if not os.path.isdir(ref_dir):
            return results
        for name in os.listdir(ref_dir):
            full = os.path.join(ref_dir, name)
            if os.path.isdir(full):
                m = pattern.match(name)
                if m:
                    results.append((name, m.group(1), m.group(2)))
        return results

    # ==================================================================
    # Browse helpers
    # ==================================================================

    def _browse_raster(self):
        """Open file browser (with 'Load' button) and load on selection."""
        f = browse_file(
            self._root,
            title="Load raster file",
            initial_dir=(os.path.dirname(self._raster_full_path)
                         if self._raster_full_path
                         else self._get_initial_browse_dir()),
            filetypes=[
                ("ENVI binary", "*.bin"),
                ("ENVI header", "*.hdr"),
                ("ENVI data", "*.dat *.img *.bsq *.bil *.bip"),
                ("GeoTIFF", "*.tif *.tiff"),
                ("All files", "*.*"),
            ],
            current_value=self._raster_full_path,
            select_label="Load",
        )
        if f:
            self._raster_full_path = f
            self._raster_path_var.set(os.path.basename(f))
            self._load_raster()

    def _load_raster_from_entry(self):
        """Load raster from a path typed/pasted into the entry box."""
        val = self._raster_path_var.get().strip()
        if not val:
            return

        # If it's already just the basename of the loaded file, nothing to do
        if (self._raster_full_path
                and val == os.path.basename(self._raster_full_path)):
            return

        # Try as an absolute path
        if os.path.isfile(val):
            self._raster_full_path = os.path.abspath(val)
            self._raster_path_var.set(os.path.basename(val))
            self._load_raster()
            return

        # Try relative to working directory
        if self._working_dir:
            candidate = os.path.join(self._working_dir, val)
            if os.path.isfile(candidate):
                self._raster_full_path = os.path.abspath(candidate)
                self._raster_path_var.set(os.path.basename(candidate))
                self._load_raster()
                return

        messagebox.showwarning("Warning",
                               f"File not found: {val}")

    def _browse_base_raster(self):
        f = browse_file(
            self._root,
            title="Select reference raster for rasterization",
            initial_dir=(os.path.dirname(self._base_raster_full_path)
                         if self._base_raster_full_path
                         else self._get_initial_browse_dir()),
            filetypes=[
                ("ENVI binary", "*.bin"),
                ("ENVI header", "*.hdr"),
                ("GeoTIFF", "*.tif *.tiff"),
                ("All files", "*.*"),
            ],
            current_value=self._base_raster_full_path,
        )
        if f:
            self._base_raster_full_path = f
            self._base_raster_var.set(os.path.basename(f))
            self._update_acc_output_name()

    def _load_ref_from_entry(self):
        """Resolve a path typed/pasted into the Ref entry box."""
        val = self._base_raster_var.get().strip()
        if not val:
            return

        # If it's just the basename of the already-loaded ref, nothing to do
        if (self._base_raster_full_path
                and val == os.path.basename(self._base_raster_full_path)):
            return

        # Try as an absolute path
        if os.path.isfile(val):
            self._base_raster_full_path = os.path.abspath(val)
            self._base_raster_var.set(os.path.basename(val))
            self._update_acc_output_name()
            return

        # Try relative to working directory
        if self._working_dir:
            candidate = os.path.join(self._working_dir, val)
            if os.path.isfile(candidate):
                self._base_raster_full_path = os.path.abspath(candidate)
                self._base_raster_var.set(os.path.basename(candidate))
                self._update_acc_output_name()
                return

        messagebox.showwarning("Warning", f"File not found: {val}")

    # ==================================================================
    # Startup raster loading (CLI argument)
    # ==================================================================

    def _parse_s2_date_from_path(self, path: str) -> Optional[date]:
        """If the filename starts with S2, parse the date from the 3rd field.

        E.g. S2B_MSIL1C_20251009T192229_... -> 2025-10-09
        Returns None if not an S2 file or the timestamp field is missing/invalid.
        """
        fname = os.path.basename(path)
        if not fname.upper().startswith("S2"):
            return None
        parts = fname.split("_")
        if len(parts) < 3:
            return None
        ts = parts[2]  # e.g. "20251009T192229"
        for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
            try:
                return _datetime.strptime(ts, fmt).date()
            except ValueError:
                continue
        print(f"[WARN] S2 filename: could not parse timestamp from 3rd field: {ts!r}",
              file=sys.stderr)
        return None

    def _load_startup_raster(self, path: str):
        """Load a raster supplied on the command line."""
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            print(f"[ERROR] Raster file not found: {path}", file=sys.stderr)
            return

        self._raster_full_path = os.path.abspath(path)
        self._raster_path_var.set(os.path.basename(path))
        try:
            self._load_raster()
        except Exception as exc:
            print(f"[ERROR] Failed to load raster: {exc}", file=sys.stderr)

    def _check_s2_dates_and_download(self):
        """Called after every new raster load.

        If no shapefiles were found (no existing data) and the loaded file is
        a Sentinel-2 image, parse the acquisition date from the filename,
        populate the date boxes, and — when both boxes are filled —
        automatically show the Download confirmation dialog.
        """
        # Shapefiles already loaded means data exists; nothing to do.
        if self._shp_status_label.cget("text") == "loaded":
            return

        s2_date = self._parse_s2_date_from_path(self._raster_full_path)
        if s2_date is None:
            return

        self._end_date_var.set(str(s2_date))

        march_1 = date(s2_date.year, 3, 1)
        if s2_date >= march_1:
            self._start_date_var.set(str(march_1))
        else:
            print(f"[INFO] S2 date {s2_date} is before March 1; "
                  "only end date populated.", file=sys.stderr)

        if self._start_date_var.get().strip() and self._end_date_var.get().strip():
            self._root.after(100, self._start_download)

    # ==================================================================
    # Loading (raster must be loaded first)
    # ==================================================================

    def _load_raster(self, bands=None):
        """Load (or reload) the raster.

        Parameters
        ----------
        bands : list[int] | None
            1-based band indices to display.  When *None* (default, used for
            new-file loads) the band selection is reset and all bands (up to 3)
            are shown.  Pass an explicit list when reloading for a band change
            (skips shapefile reload / precompute since fire data is unchanged).
        """
        band_change_only = bands is not None

        raster_path = self._raster_full_path
        if not raster_path or not os.path.exists(raster_path):
            messagebox.showwarning("Warning",
                                   "Could not read the selected raster file.")
            return

        # Preserve current animation position
        self._save_slider_position()

        self._status_var.set("Loading raster\u2026")
        self._root.update_idletasks()

        try:
            if band_change_only:
                # Explicit band selection (e.g. from band selector popup)
                self._selected_bands = list(bands)
            else:
                # New raster — peek at band names via cheap gdal metadata read
                # so we can auto-select the best 3 bands in a single load pass.
                self._selected_bands = []
                try:
                    from osgeo import gdal as _gdal
                    _ds = _gdal.Open(raster_path, _gdal.GA_ReadOnly)
                    if _ds is not None:
                        _names = [_ds.GetRasterBand(i).GetDescription()
                                  for i in range(1, _ds.RasterCount + 1)]
                        _ds = None
                        self._selected_bands = _auto_select_bands(_names)
                except Exception:
                    pass

            load_bands = self._selected_bands if self._selected_bands else None
            img = self._raster_loader.load(raster_path, bands=load_bands)
            ext = self._raster_loader.extent
            self._canvas.display_raster(img, ext)
            self._raster_loaded = True

            # If auto-selection produced nothing, fall back to first 3.
            if not self._selected_bands:
                n = self._raster_loader.raster._n_band
                self._selected_bands = list(range(1, min(n, 3) + 1))

            # Set working directory
            self._set_working_dir(raster_path)

            # Compute scatter size from raster resolution (VNP14IMG 375m / pixel_size)
            scatter_sz = self._raster_loader.compute_scatter_size()
            cfg.DEFAULT_SCATTER_SIZE = scatter_sz
            self._canvas.scatter_size = scatter_sz

            self._status_var.set(
                f"Raster loaded: {os.path.basename(raster_path)}  "
                f"({self._raster_loader._raster._xSize}"
                f"\u00d7{self._raster_loader._raster._ySize})  "
                f"Fire pixel scatter size: {scatter_sz}"
            )
        except Exception as exc:
            self._raster_loaded = False
            messagebox.showwarning("Warning",
                                   f"Could not read raster image:\n{exc}")
            self._status_var.set("Raster load failed.")
            return

        # Restore slider position
        self._restore_slider_position()

        # Band-only change: image updated, fire data unchanged — done.
        if band_change_only:
            return

        # --- Below only runs for a genuinely new raster file ---

        # Store CRS info for download
        self._store_raster_crs_info(raster_path)

        self._update_crs_display()

        # Sync ref raster and compute VIIRS dir
        self._sync_base_raster()
        self._compute_viirs_dir()

        if (self._data_mgr.master_gdf is not None
                and not self._data_mgr.master_gdf.empty):
            self._reclip_and_refresh()

        # Auto-load shapefiles from _VIIRS folder if it exists
        self._check_viirs_folder()

        # After shapefile check: if none found, try S2 date auto-fill + download
        self._check_s2_dates_and_download()

    # ==================================================================
    # Band Selector Popup
    # ==================================================================

    def _show_band_selector(self):
        """Open a popup to choose up to 3 bands for RGB display."""
        raster = self._raster_loader.raster
        if raster is None:
            messagebox.showinfo("Band Selector",
                                "Load a raster first.")
            return

        band_names = raster.band_info_list          # list of str, len == n_bands
        n_bands = len(band_names)

        # ----------------------------------------------------------
        # Popup window
        # ----------------------------------------------------------
        popup = tk.Toplevel(self._root)
        popup.title("Select Bands (max 3)")
        popup.resizable(False, False)
        popup.grab_set()                             # modal

        CELL_W = 260
        CELL_H = 32
        MAX_VISIBLE = 10                             # rows before scrolling
        visible_rows = min(n_bands, MAX_VISIBLE)
        list_h = visible_rows * CELL_H

        # ----------------------------------------------------------
        # Scrollable list area
        # ----------------------------------------------------------
        outer = tk.Frame(popup, bd=1, relief=tk.SUNKEN)
        outer.pack(padx=8, pady=(8, 4))

        canvas = tk.Canvas(outer, width=CELL_W, height=list_h,
                           highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient=tk.VERTICAL,
                                  command=canvas.yview)
        inner = tk.Frame(canvas)

        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Mouse-wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)       # Windows / macOS
        canvas.bind_all("<Button-4>",                          # Linux scroll up
                        lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>",                          # Linux scroll down
                        lambda e: canvas.yview_scroll(1, "units"))

        # ----------------------------------------------------------
        # State: selection_order[i] = 0 means unselected,
        #        otherwise the 1-based pick order.
        # ----------------------------------------------------------
        selection_order = [0] * n_bands
        labels: list[tk.Label] = []

        # Pre-fill with current selection (if any)
        for order, band_idx in enumerate(self._selected_bands, start=1):
            if 1 <= band_idx <= n_bands:
                selection_order[band_idx - 1] = order

        def _next_order():
            return max(selection_order) + 1 if any(selection_order) else 1

        def _compact_order():
            """Re-number so orders are consecutive 1, 2, 3 …"""
            ordered = sorted(
                [(sel, i) for i, sel in enumerate(selection_order) if sel > 0]
            )
            for new_ord, (_, idx) in enumerate(ordered, start=1):
                selection_order[idx] = new_ord

        def _refresh_cells():
            for i, lbl in enumerate(labels):
                if selection_order[i] > 0:
                    lbl.configure(bg="#ADD8E6",                  # light blue
                                  text=f" {selection_order[i]}  {band_names[i]}")
                else:
                    lbl.configure(bg="white",
                                  text=f"      {band_names[i]}")

        def _toggle(idx):
            if selection_order[idx] > 0:
                # Deselect
                selection_order[idx] = 0
                _compact_order()
            else:
                # If already at 3, evict the one with order == 1 (oldest pick)
                chosen = sum(1 for s in selection_order if s > 0)
                if chosen >= 3:
                    for j in range(n_bands):
                        if selection_order[j] == 1:
                            selection_order[j] = 0
                            break
                    _compact_order()
                selection_order[idx] = _next_order()
            _refresh_cells()

        # Build cells
        for i in range(n_bands):
            lbl = tk.Label(
                inner, text="", anchor="w", width=CELL_W // 8,
                height=1, relief=tk.RIDGE, bd=1, bg="white",
                font=("TkDefaultFont", 10), padx=4, pady=2,
            )
            lbl.pack(fill=tk.X)
            lbl.bind("<Button-1>", lambda e, idx=i: _toggle(idx))
            labels.append(lbl)

        _refresh_cells()

        # ----------------------------------------------------------
        # Buttons
        # ----------------------------------------------------------
        btn_frame = tk.Frame(popup)
        btn_frame.pack(pady=(4, 8))

        def _apply():
            ordered = sorted(
                [(sel, i) for i, sel in enumerate(selection_order) if sel > 0]
            )
            if not ordered:
                messagebox.showwarning("Band Selector",
                                       "Select at least 1 band.",
                                       parent=popup)
                return
            chosen_bands = [i + 1 for _, i in ordered]  # 1-based
            # Unbind mousewheel before closing
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")
            popup.destroy()
            self._load_raster(bands=chosen_bands)

        def _cancel():
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")
            popup.destroy()

        tk.Button(btn_frame, text="Apply", width=8,
                  command=_apply).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="Cancel", width=8,
                  command=_cancel).pack(side=tk.LEFT, padx=4)

        # Center the popup over the main window
        popup.update_idletasks()
        pw, ph = popup.winfo_width(), popup.winfo_height()
        rx = self._root.winfo_x() + (self._root.winfo_width() - pw) // 2
        ry = self._root.winfo_y() + (self._root.winfo_height() - ph) // 2
        popup.geometry(f"+{rx}+{ry}")

    def _store_raster_crs_info(self, raster_path: str):
        """Read and cache WKT/EPSG from the raster for download bbox conversion."""
        try:
            from osgeo import gdal, osr
            gdal.UseExceptions()
            ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
            if ds is None:
                return
            wkt = ds.GetProjection()
            ds = None
            if wkt:
                self._ref_wkt = wkt
                srs = osr.SpatialReference()
                srs.ImportFromWkt(wkt)
                srs.AutoIdentifyEPSG()
                code = srs.GetAuthorityCode(None)
                self._ref_epsg = int(code) if code else None
        except Exception:
            pass

    def _update_crs_display(self):
        crs_str = self._raster_loader.crs
        if crs_str:
            try:
                from osgeo import osr
                srs = osr.SpatialReference()
                srs.ImportFromWkt(crs_str)
                srs.AutoIdentifyEPSG()
                code = srs.GetAuthorityCode(None)
                if code:
                    self._crs_string = f"EPSG: {code}"
                else:
                    self._crs_string = "EPSG: Unknown"
            except Exception:
                self._crs_string = "EPSG: Unknown"
        else:
            self._crs_string = "EPSG: \u2014"
        self._crs_var.set(self._crs_string)

    def _load_shapefiles(self):
        # Require raster first
        if not self._raster_loaded:
            messagebox.showwarning(
                "Raster Required",
                "Please load a raster image before loading shapefiles.\n"
                "The raster defines the projection and spatial extent.")
            return

        shp_dir = self._shapefile_dir_var.get().strip()
        if not shp_dir or not os.path.isdir(shp_dir):
            messagebox.showerror("Error",
                                 "Please select a valid shapefile directory.")
            return

        # Preserve current animation position
        self._save_slider_position()

        self._status_var.set("Scanning shapefiles\u2026")
        self._root.update_idletasks()

        file_dict = self._data_mgr.scan_directory(shp_dir)
        if not file_dict:
            self._update_shp_status(False)
            self._status_var.set("No shapefiles found in directory.")
            return

        def _lp(loaded, total):
            self._status_var.set(f"Loading shapefiles\u2026 {loaded}/{total}")
            self._root.update_idletasks()

        raster_crs = self._raster_loader.crs if self._raster_loaded else None
        gdf = self._data_mgr.load_all(progress_cb=_lp, target_crs=raster_crs)

        if gdf.empty:
            self._update_shp_status(False)
            self._status_var.set("No data loaded (shapefiles empty).")
            return

        n_total = len(gdf)
        n_removed = 0

        if self._raster_loaded and self._raster_loader.extent is not None:
            ext = self._raster_loader.extent
            n_removed = self._data_mgr.clip_to_extent(*ext)
            n_kept = n_total - n_removed
            if n_kept == 0:
                messagebox.showwarning("Warning",
                                       "All pixels fell outside the raster.")
                self._status_var.set("No pixels within raster extent.")
                return
        else:
            n_kept = n_total
            data_ext = self._data_mgr.get_data_extent()
            if data_ext is not None:
                self._canvas.set_black_background(data_ext)

        self._canvas.set_row_lookup(self._data_mgr.get_row_data)

        min_d, max_d = self._data_mgr.get_date_range_bounds()
        self._start_date_var.set(str(min_d))
        self._end_date_var.set(str(max_d))

        self._status_var.set("Pre-computing frames\u2026")
        self._root.update_idletasks()

        def _pp(done, total):
            self._status_var.set(f"Pre-computing frames\u2026 {done}/{total}")
            self._root.update_idletasks()

        self._data_mgr.precompute_frames(progress_cb=_pp)
        self._setup_animator(preserve_position=True)

        raster_note = (f" ({n_removed} outside raster discarded)"
                       if self._raster_loaded else " (no raster)")
        self._status_var.set(
            f"Loaded {n_kept} pixels{raster_note} "
            f"from {len(file_dict)} files.  {min_d} \u2192 {max_d}"
        )
        self._update_shp_status(True)
        self._update_acc_output_name()

    def _reclip_and_refresh(self):
        ext = self._raster_loader.extent
        if ext is None:
            return

        self._status_var.set("Clipping to raster extent\u2026")
        self._root.update_idletasks()

        raster_crs = self._raster_loader.crs

        def _lp(loaded, total):
            self._status_var.set(
                f"Reloading shapefiles\u2026 {loaded}/{total}")
            self._root.update_idletasks()

        gdf = self._data_mgr.load_all(progress_cb=_lp, target_crs=raster_crs)
        if gdf.empty:
            return

        n_total = len(gdf)
        n_removed = self._data_mgr.clip_to_extent(*ext)
        n_kept = n_total - n_removed

        if n_kept == 0:
            messagebox.showwarning("Warning",
                                   "All pixels fell outside the new raster.")
            self._status_var.set("No pixels within raster extent.")
            return

        self._canvas.set_row_lookup(self._data_mgr.get_row_data)
        min_d, max_d = self._data_mgr.get_date_range_bounds()
        self._start_date_var.set(str(min_d))
        self._end_date_var.set(str(max_d))

        self._status_var.set("Pre-computing frames\u2026")
        self._root.update_idletasks()
        self._data_mgr.precompute_frames()

        self._setup_animator(preserve_position=True)
        self._status_var.set(
            f"Re-clipped: {n_kept} pixels ({n_removed} outside raster)."
            f"  {min_d} \u2192 {max_d}"
        )

    def _apply_date_filter_silent(self, event=None):
        """Apply date filter on focus-out — no error popups for empty/invalid."""
        start_str = self._start_date_var.get().strip()
        end_str = self._end_date_var.get().strip()
        if not start_str or not end_str:
            return
        try:
            date.fromisoformat(start_str)
            date.fromisoformat(end_str)
        except ValueError:
            return
        self._apply_date_filter()

    def _apply_date_filter(self):
        """Apply date filter -- works even without shapefiles loaded."""
        result = self._validate_dates()
        if result is None:
            return
        start_str, end_str, start, end = result

        # If no shapefiles loaded yet, just store the dates (for download)
        if (self._data_mgr.master_gdf is None
                or self._data_mgr.master_gdf.empty):
            self._status_var.set(
                f"Date range set: {start} \u2192 {end}  (no shapefiles loaded)")
            return

        self._save_slider_position()

        self._status_var.set("Filtering by date\u2026")
        self._root.update_idletasks()
        gdf = self._data_mgr.load_filtered(start, end)

        if gdf.empty:
            messagebox.showinfo("Info", "No data in that date range.")
            self._status_var.set("No data in range.")
            return

        ext = self._canvas.raster_extent
        if ext is not None and self._raster_loaded:
            self._data_mgr.clip_to_extent(*ext)
        else:
            data_ext = self._data_mgr.get_data_extent()
            if data_ext is not None:
                self._canvas.set_black_background(data_ext)

        self._canvas.set_row_lookup(self._data_mgr.get_row_data)
        self._canvas.clear()

        self._status_var.set("Pre-computing frames\u2026")
        self._root.update_idletasks()
        self._data_mgr.precompute_frames()

        self._setup_animator(preserve_position=True)
        n = (len(self._data_mgr.master_gdf)
             if self._data_mgr.master_gdf is not None else 0)
        self._status_var.set(
            f"Filtered: {n} pixels,  {start} \u2192 {end}")
        self._update_acc_output_name()

    # ==================================================================
    # Slider position preservation
    # ==================================================================

    def _save_slider_position(self):
        """Save the current animation position so we can restore it."""
        if self._animator and self._animator.current_date is not None:
            self._preserved_slider_date = self._animator.current_date
            self._preserved_slider_index = self._animator.current_index

    def _restore_slider_position(self):
        """Restore the slider to the saved date/index if valid."""
        if self._animator is None:
            return
        if self._preserved_slider_date is not None:
            self._animator.jump_to_date(self._preserved_slider_date)
            if self._animator.current_date is not None:
                self._on_animation_frame(self._animator.current_date)
            self._preserved_slider_date = None
            self._preserved_slider_index = None

    def _setup_animator(self, preserve_position: bool = False):
        start_str = self._start_date_var.get().strip()
        end_str = self._end_date_var.get().strip()
        try:
            start = date.fromisoformat(start_str)
            end = date.fromisoformat(end_str)
        except ValueError:
            return

        saved_date = self._preserved_slider_date if preserve_position else None

        self._animator.set_date_range(start, end)
        self._frame_slider.configure(
            to=max(self._animator.total_frames - 1, 1))

        if saved_date is not None:
            # Restore to saved position
            self._animator.jump_to_date(saved_date)
            idx = self._animator.current_index
            current = self._animator.current_date or start
            self._frame_slider.set(idx)
            self._frame_label_var.set(
                f"Frame: {idx + 1} / {self._animator.total_frames}")
            self._date_label_var.set(f"Date: {current}")
            self._on_animation_frame(current)
            self._preserved_slider_date = None
            self._preserved_slider_index = None
        else:
            self._frame_label_var.set(
                f"Frame: 0 / {self._animator.total_frames}")
            self._date_label_var.set(f"Date: {start}")
            self._on_animation_frame(start)

    # ==================================================================
    # Playback
    # ==================================================================

    def _toggle_play(self):
        if self._animator is None:
            return
        if self._animator.is_playing:
            self._animator.pause()
            self._play_btn.config(text="\u25b6  Play")
        else:
            self._animator.interval_ms = self._interval_var.get()
            self._animator.play()
            if self._animator.is_playing:
                self._play_btn.config(text="\u23f8  Pause")

    def _skip_fwd_n(self):
        if self._animator:
            self._animator.jump_by(self._skip_n_var.get())

    def _skip_back_n(self):
        if self._animator:
            self._animator.jump_by(-self._skip_n_var.get())

    def _reset_animation(self):
        if self._animator:
            self._animator.stop()
            self._play_btn.config(text="\u25b6  Play")
            self._canvas.clear()
            self._canvas.reset_view()
            self._frame_slider.set(0)
            self._date_label_var.set("Date: \u2014")

    def _update_speed(self):
        if self._animator:
            self._animator.interval_ms = self._interval_var.get()

    def _toggle_raster(self):
        if self._canvas:
            self._canvas.set_raster_visible(self._show_raster_var.get())

    def _toggle_fire(self):
        if self._canvas:
            self._canvas.set_scatter_visible(self._show_fire_var.get())

    # -- Slider --

    def _on_slider(self, val):
        if self._updating_slider:
            return
        if not self._animator or self._animator.is_playing:
            return

        self._slider_pending_val = int(float(val))
        self._slider_dragging = True

        if self._slider_throttle_id is not None:
            self._root.after_cancel(self._slider_throttle_id)
        if self._slider_release_id is not None:
            self._root.after_cancel(self._slider_release_id)

        self._slider_throttle_id = self._root.after(
            self._SLIDER_THROTTLE_MS, self._slider_deferred_render)

    def _slider_deferred_render(self):
        self._slider_throttle_id = None
        val = self._slider_pending_val
        if val is None:
            return

        self._animator.jump_to(val)
        current_date = self._animator.current_date
        if current_date is None:
            return

        fd = self._data_mgr.get_frame(current_date)

        self._date_label_var.set(f"Date: {current_date}")
        idx = self._animator.current_index
        total = self._animator.total_frames
        self._frame_label_var.set(f"Frame: {idx + 1} / {total}")
        self._pixel_count_var.set(f"Pixels: {fd.n_pixels}")

        scatter_sz = cfg.DEFAULT_SCATTER_SIZE
        self._canvas.scatter_size = scatter_sz
        self._canvas.update_scatter(
            fd.x, fd.y, fd.ages, fd.indices,
            n_levels=cfg.N_COLOUR_LEVELS,
            max_points=self._SLIDER_MAX_POINTS,
        )
        self._update_viewport_pixel_count()

        self._slider_release_id = self._root.after(
            self._SLIDER_RELEASE_MS, self._slider_full_render)

    def _slider_full_render(self):
        self._slider_release_id = None
        self._slider_dragging = False

        if not self._animator or self._animator.current_date is None:
            return

        fd = self._data_mgr.get_frame(self._animator.current_date)
        self._canvas.scatter_size = cfg.DEFAULT_SCATTER_SIZE
        self._canvas.update_scatter(
            fd.x, fd.y, fd.ages, fd.indices,
            n_levels=cfg.N_COLOUR_LEVELS,
        )
        self._update_viewport_pixel_count()

    # -- Animation frame --

    def _on_animation_frame(self, current_date: date):
        fd = self._data_mgr.get_frame(current_date)

        self._canvas.scatter_size = cfg.DEFAULT_SCATTER_SIZE
        self._canvas.update_scatter(
            fd.x, fd.y, fd.ages, fd.indices,
            n_levels=cfg.N_COLOUR_LEVELS,
        )

        self._date_label_var.set(f"Date: {current_date}")
        idx = self._animator.current_index
        total = self._animator.total_frames
        self._frame_label_var.set(f"Frame: {idx + 1} / {total}")
        self._pixel_count_var.set(f"Pixels: {fd.n_pixels}")
        self._update_viewport_pixel_count()

        self._updating_slider = True
        self._frame_slider.set(idx)
        self._updating_slider = False

    def _on_animation_finished(self):
        self._play_btn.config(text="\u25b6  Play")
        self._status_var.set("Animation complete.")

    # ==================================================================
    # Download (integrated — no separate dialog)
    # ==================================================================

    def _start_download(self):
        """Validate and show confirmation popup for download."""
        if not self._raster_loaded:
            messagebox.showwarning(
                "Raster Required",
                "Load a raster image first.\n"
                "The raster provides the reference CRS and bounding box.")
            return

        # Handle token — prompt if not loaded
        if not self._token:
            if not self._prompt_token():
                return

        result = self._validate_dates()
        if result is None:
            return
        start_str, end_str, start_d, end_d = result
        start_dt = datetime.datetime.combine(start_d, datetime.time())
        end_dt = datetime.datetime.combine(end_d, datetime.time())

        if self._ref_wkt is None:
            messagebox.showerror("Error",
                                 "Could not read CRS from raster.")
            return

        if self._download_thread and self._download_thread.is_alive():
            messagebox.showwarning("Busy", "Download already in progress.")
            return

        # Compute bbox (always from the main raster for CRS/extent)
        raster_path = self._raster_full_path
        # Ref raster drives accumulation output dir
        ref_path = self._base_raster_full_path or self._raster_full_path
        try:
            from download_dialog import _read_raster_info, _bbox_to_4326
            wkt, epsg, x_min, x_max, y_min, y_max = _read_raster_info(
                raster_path)
            west, south, east, north = _bbox_to_4326(
                wkt, x_min, x_max, y_min, y_max)
        except Exception as exc:
            messagebox.showerror("Error",
                                 f"Could not compute bbox:\n{exc}")
            return

        # Compute VIIRS dir
        if not self._viirs_dir:
            self._compute_viirs_dir()
        viirs_dir = self._viirs_dir

        # Update acc output name
        self._update_acc_output_name()

        # Confirmation popup
        popup = tk.Toplevel(self._root)
        popup.title("Confirm Download")
        popup.transient(self._root)
        popup.grab_set()

        w, h = 480, 240
        sx = self._root.winfo_screenwidth()
        sy = self._root.winfo_screenheight()
        popup.geometry(f"{w}x{h}+{(sx-w)//2}+{(sy-h)//2}")
        popup.resizable(False, False)

        ttk.Label(popup, text="Download VNP14IMG Data",
                  font=("TkDefaultFont", 12, "bold")).pack(pady=(12, 6))

        info = ttk.Frame(popup)
        info.pack(fill=tk.X, padx=12, pady=4)

        def _row(label, value):
            f = ttk.Frame(info)
            f.pack(fill=tk.X, pady=1)
            ttk.Label(f, text=label, width=14, anchor="e",
                      font=("TkDefaultFont", 9, "bold")).pack(side=tk.LEFT)
            ttk.Label(f, text=value, anchor="w",
                      font=("TkDefaultFont", 9)).pack(side=tk.LEFT, padx=6)

        epsg_str = f"EPSG:{self._ref_epsg}" if self._ref_epsg else "Unknown"
        _row("CRS:", epsg_str)
        _row("Date range:", f"{start_str}  \u2192  {end_str}")
        _row("Save to:", os.path.basename(viirs_dir))
        if self._acc_output_full_path:
            _row("Accumulation:", os.path.basename(self._acc_output_full_path))

        btn_f = ttk.Frame(popup)
        btn_f.pack(pady=10)

        def _go():
            popup.destroy()
            self._run_download(
                self._token, start_dt, end_dt, viirs_dir, ref_path,
                west, south, east, north)

        tk.Button(
            btn_f, text="  \u2714  Confirm  ", bg="#4CAF50", fg="white",
            font=("TkDefaultFont", 10, "bold"), activebackground="#388E3C",
            command=_go,
        ).pack(side=tk.LEFT, padx=8)
        tk.Button(
            btn_f, text="  Cancel  ", bg="#F44336", fg="white",
            font=("TkDefaultFont", 9, "bold"), activebackground="#C62828",
            command=popup.destroy,
        ).pack(side=tk.LEFT, padx=8)

    def _run_download(self, token, start_dt, end_dt, viirs_dir, ref_path,
                      west, south, east, north):
        self._download_btn.configure(state=tk.DISABLED)
        self._download_cancel.clear()

        self._download_thread = threading.Thread(
            target=self._download_worker,
            args=(token, start_dt, end_dt, viirs_dir, ref_path,
                  west, south, east, north),
            daemon=True,
        )
        self._download_thread.start()

    def _download_worker(self, token, start_dt, end_dt, viirs_dir, ref_path,
                         west, south, east, north):
        product = "VNP14IMG"
        interval = datetime.timedelta(days=1)
        day = start_dt
        days = []
        while day <= end_dt:
            days.append(day)
            day += interval
        total_days = len(days)

        def _status(msg):
            try:
                self._root.after(0, lambda: self._status_var.set(msg))
            except Exception:
                pass

        # Import sync
        try:
            from viirs.utils.laads_data_download_v2 import sync
            sync_fn = sync
        except ImportError:
            _status("ERROR: Could not import laads_data_download_v2.sync")
            self._download_finish()
            return

        completed = 0
        skipped = 0
        lock = threading.Lock()

        def download_one(download_day):
            nonlocal completed, skipped
            if self._download_cancel.is_set():
                return

            jday = download_day.timetuple().tm_yday
            year = download_day.year

            download_url = (
                f"https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/details?"
                f"products={product}&"
                f"temporalRanges={year}-{jday}&"
                f"regions=%5BBBOX%5DN{north:.6f}%20S{south:.6f}"
                f"%20E{east:.6f}%20W{west:.6f}"
            )
            download_path = os.path.join(
                viirs_dir, f"{year:04d}", f"{jday:03d}")

            # Check if already downloaded (has .nc files)
            if os.path.isdir(download_path):
                nc_files = [f for f in os.listdir(download_path)
                            if f.lower().endswith('.nc')]
                if nc_files:
                    print(f"[SKIP] {download_day.strftime('%Y-%m-%d')} "
                          f"— {len(nc_files)} .nc file(s) already exist")
                    with lock:
                        skipped += 1
                        completed += 1
                        done = completed
                    _status(f"Downloaded {done}/{total_days} days  "
                            f"({download_day.strftime('%Y-%m-%d')} skipped)")
                    return

            os.makedirs(download_path, exist_ok=True)

            print(f"\n[DOWNLOAD] {download_day.strftime('%Y-%m-%d')}")
            print(f"  URL: {download_url}")
            print(f"  Dir: {download_path}")

            try:
                sync_fn(download_url, download_path, token)
            except Exception as exc:
                print(f"[WARN] Download error for {download_day}: {exc}")

            with lock:
                completed += 1
                done = completed

            _status(f"Downloaded {done}/{total_days} days  "
                    f"({download_day.strftime('%Y-%m-%d')})")

        max_workers = 16
        _status(f"Downloading: {total_days} days, {max_workers} workers\u2026")

        executor = ThreadPoolExecutor(max_workers=max_workers)
        self._download_executor = executor
        try:
            future_to_day = {
                executor.submit(download_one, d): d for d in days}
            for future in as_completed(future_to_day):
                if self._download_cancel.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                try:
                    future.result()
                except Exception as exc:
                    d = future_to_day[future]
                    print(f"[WARN] Unhandled error for {d}: {exc}")
        finally:
            self._download_executor = None
            executor.shutdown(wait=False)

        if self._download_cancel.is_set():
            _status("Download cancelled.")
            self._download_finish()
            return

        if skipped > 0:
            print(f"\n[INFO] Download finished. "
                  f"{skipped}/{total_days} days skipped (already exist).")
        _status("Download complete. Running shapify\u2026")
        print(f"\n[INFO] Download finished.")

        self._run_shapify(viirs_dir, ref_path, west, south, east, north,
                          _status)

    def _run_shapify(self, viirs_dir, ref_path, west, south, east, north,
                     _status):
        if self._download_cancel.is_set():
            self._download_finish()
            return

        _status("Running shapify (converting .nc \u2192 .shp)\u2026")
        print("\n[INFO] Starting shapify\u2026")

        cmd = [
            sys.executable, "-m", "viirs.utils.shapify",
            viirs_dir,
            "-r", ref_path,
            "-w", "16",
            "--bbox",
            f"{west:.6f}", f"{south:.6f}", f"{east:.6f}", f"{north:.6f}",
        ]
        print(f"[INFO] Command: {' '.join(cmd)}")

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
            self._shapify_proc = proc
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    print(f"  [shapify] {line}")
                if self._download_cancel.is_set():
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    _status("Shapify cancelled.")
                    self._download_finish()
                    return

            proc.wait()
            self._shapify_proc = None
            if proc.returncode == 0:
                # Count created shapefiles
                n_shp = 0
                if os.path.isdir(viirs_dir):
                    for root, dirs, files in os.walk(viirs_dir):
                        for fn in files:
                            if fn.lower().endswith(".shp"):
                                n_shp += 1

                _status(f"Shapify complete! {n_shp} shapefiles. "
                        f"Starting accumulation\u2026")
                print("[INFO] Shapify finished successfully.")

                # Auto-run accumulation + rasterize
                self._run_accumulation_after_download(
                    viirs_dir, ref_path, _status)
                return  # _run_accumulation handles finish
            else:
                _status(f"Shapify exited with code {proc.returncode}.")
                print(f"[WARN] Shapify exit code: {proc.returncode}")
        except FileNotFoundError:
            _status("ERROR: shapify command not found.")
        except Exception as exc:
            _status(f"Shapify error: {exc}")

        self._download_finish()

    def _download_finish(self):
        def _on_main_thread():
            self._download_btn.configure(state=tk.NORMAL)
            # Reload shapefiles so new data appears in the viewer
            if (self._viirs_dir and os.path.isdir(self._viirs_dir)
                    and self._raster_loaded):
                self._shapefile_dir_var.set(self._viirs_dir)
                self._load_shapefiles()
                self._update_shp_status(True)
            # Run post-download callback if set (e.g. re-launch fire mapping)
            cb = self._post_download_callback
            self._post_download_callback = None
            if cb is not None:
                cb()
        try:
            self._root.after(0, _on_main_thread)
        except Exception:
            pass

    # ==================================================================
    # Accumulate + Rasterize (integrated into download pipeline)
    # ==================================================================

    def _run_accumulation_after_download(self, viirs_dir, ref_path, _status):
        """
        Run accumulation + rasterize after download completes.
        Handles smart accumulation: checks for existing folders,
        extends if same start date, skips if exact match.
        """
        start_str = self._start_date_var.get().strip()
        end_str = self._end_date_var.get().strip()

        if not start_str or not end_str:
            _status("Download complete (no dates set for accumulation).")
            self._download_finish()
            return

        ref_base = os.path.splitext(os.path.basename(ref_path))[0]
        ref_dir = os.path.dirname(ref_path)
        start_compact = start_str.replace("-", "")
        end_compact = end_str.replace("-", "")

        out_folder_name = f"{ref_base}_{start_compact}_{end_compact}_ACCUMULATED"
        out_dir = os.path.join(ref_dir, out_folder_name)

        # Smart accumulation: check for existing folders
        existing = self._find_accumulated_folders(ref_dir, ref_base)

        # Check for exact match
        for folder_name, f_start, f_end in existing:
            if f_start == start_compact and f_end == end_compact:
                _status(f"Accumulation already exists: {folder_name}")
                self._show_done_popup(
                    "Already Exists",
                    f"Accumulation already exists:\n{folder_name}\n\n"
                    f"No new accumulation needed.")
                # Update the output dir display
                self._acc_output_full_path = os.path.join(
                    ref_dir, folder_name)
                try:
                    self._root.after(
                        0, lambda fn=folder_name:
                        self._acc_output_dir_var.set(fn))
                except Exception:
                    pass
                self._download_finish()
                return

        # Check for same start, different end
        for folder_name, f_start, f_end in existing:
            if f_start == start_compact:
                old_folder_path = os.path.join(ref_dir, folder_name)

                if end_compact > f_end:
                    # New end is further out — extend by renaming
                    _status(f"Extending accumulation from {f_end} to "
                            f"{end_compact}\u2026")
                    try:
                        os.rename(old_folder_path, out_dir)
                        print(f"[INFO] Renamed {folder_name} -> "
                              f"{out_folder_name}")
                    except Exception as exc:
                        print(f"[WARN] Could not rename old folder: {exc}")
                    break

                else:
                    # New end is shorter — ask user whether to create new
                    import queue as _queue
                    q = _queue.Queue()

                    def _ask():
                        ans = messagebox.askyesno(
                            "Accumulation Exists",
                            f"An accumulation with the same start date and "
                            f"a further end date already exists:\n\n"
                            f"  {folder_name}\n\n"
                            f"Do you want to create a new folder for end "
                            f"date {end_compact}?",
                            parent=self._root,
                        )
                        q.put(ans)

                    self._root.after(0, _ask)
                    # Block worker thread until user answers
                    user_ok = q.get()
                    if not user_ok:
                        _status("Accumulation cancelled.")
                        self._download_finish()
                        return
                    # User confirmed — create a brand new folder
                    break

        # Run accumulation
        self._do_accumulate_rasterize(
            viirs_dir, ref_path, out_dir, start_str, end_str, _status)

    def _do_accumulate_rasterize(self, shp_dir, base_raster, out_dir,
                                 start_str, end_str, _status):
        """Run the actual accumulation and rasterization pipeline."""
        # Import accumulate
        try:
            from viirs.utils.accumulate import accumulate
            accumulate_fn = accumulate
        except ImportError:
            _status("ERROR: Could not import accumulate.accumulate")
            self._download_finish()
            return

        # Import rasterize
        try:
            from viirs.utils.rasterize import rasterize_shapefile
            rasterize_fn = rasterize_shapefile
        except ImportError:
            _status("ERROR: Could not import rasterize.rasterize_shapefile")
            self._download_finish()
            return

        # Step 1: Accumulate
        start_compact = start_str.replace("-", "")
        end_compact = end_str.replace("-", "")

        _status("Accumulating shapefiles\u2026")

        try:
            shp_paths = accumulate_fn(
                shp_dir=shp_dir,
                start_str=start_compact,
                end_str=end_compact,
                reference_raster=base_raster,
                output_dir=out_dir,
                progress_cb=_status,
            )
        except Exception as exc:
            _status(f"Accumulation error: {exc}")
            self._download_finish()
            return

        if not shp_paths:
            _status("Accumulation produced no files.")
            self._download_finish()
            return

        n_shp = len(shp_paths)
        _status(f"Accumulated {n_shp} shapefiles. Rasterizing\u2026")

        # Step 2: Rasterize in parallel
        rasterized = 0
        errors = 0

        def _rast_one(shp_path):
            try:
                return rasterize_fn(shp_path, base_raster, out_dir)
            except Exception as exc:
                print(f"[WARN] Rasterize error {shp_path}: {exc}")
                return None

        max_w = min(16, n_shp)
        with ThreadPoolExecutor(max_workers=max_w) as pool:
            futs = {pool.submit(_rast_one, p): p for p in shp_paths}
            for fut in as_completed(futs):
                result = fut.result()
                if result:
                    rasterized += 1
                else:
                    errors += 1
                _status(f"Rasterizing\u2026 {rasterized + errors}/{n_shp}")

        out_name = os.path.basename(out_dir)
        _status(
            f"Done: {n_shp} shapefiles + {rasterized} rasters "
            f"saved to {out_name}"
        )
        print(f"[ACCUM & RASTERIZE] DONE")

        # Update output dir display
        self._acc_output_full_path = out_dir
        try:
            self._root.after(
                0, lambda n=out_name: self._acc_output_dir_var.set(n))
        except Exception:
            pass

        self._show_done_popup(
            "Pipeline Complete  \u2714",
            f"Download, shapify, accumulate & rasterize finished.\n\n"
            f"Shapefiles: {n_shp}\n"
            f"Rasters: {rasterized}\n"
            f"Date range: {start_str}  \u2192  {end_str}\n\n"
            f"Output directory:\n{out_dir}")
        self._download_finish()

    # ==================================================================
    # Completion popup (thread-safe)
    # ==================================================================

    def _show_done_popup(self, title: str, message: str):
        """Show a completion info popup — safe to call from background threads."""
        def _popup():
            messagebox.showinfo(title, message, parent=self._root)
        try:
            self._root.after(0, _popup)
        except Exception:
            pass

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def _on_close(self):
        if self._animator:
            self._animator.pause()

        # Cancel any running download
        self._download_cancel.set()
        if self._download_executor is not None:
            try:
                self._download_executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                self._download_executor.shutdown(wait=False)
        if self._shapify_proc is not None:
            try:
                self._shapify_proc.terminate()
                self._shapify_proc.kill()
            except Exception:
                pass

        self._root.destroy()

    def run(self):
        self._root.mainloop()