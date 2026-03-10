"""
viirs/fp_gui/fire_gui.py

RENDERING RESOURCE KNOBS
=========================
1. config.py -- all tuneable constants (edit via Config dialog)
2. fire_map_canvas.py -- figsize / dpi / blitting
3. fire_data_manager.py -- precompute_frames / numba cache
"""

import os
import sys
import threading
import datetime
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import date
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import config as cfg
from fire_data_manager import FireDataManager
from raster_loader import RasterLoader
from fire_map_canvas import FireMapCanvas, NAV_ZOOM_IN, NAV_ZOOM_OUT, NAV_PAN
from fire_animation_controller import FireAnimationController
from config_dialog import ConfigDialog
from file_browser import browse_directory, browse_file


class FireAccumulationGUI:
    """
    Top-level GUI for the VIIRS fire pixel accumulation viewer.

    Layout (vertical, top-to-bottom):
        1. Data Loader    -- raster + shapefile paths
        2. Date & Download -- date range, apply filter, token, download
        3. Playback       -- play/pause, skip, speed, frame slider
        4. Tools          -- accumulate/rasterize, nav, layer toggles

    Navigation:
        Pan        -- drag to move the map
        Zoom +     -- draw green rectangle to zoom in
        Zoom -     -- draw red rectangle to zoom out
        Scroll     -- zoom in / out at cursor (any mode)
        Left-click -- pixel detail popup (any mode, if not a drag)
        Right-click-- pixel detail popup (any mode)
    """

    def __init__(self):
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

        # Working directory (set when raster is loaded)
        self._working_dir: Optional[str] = None

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

        # Sync: when raster_path changes, default base_raster if empty
        self._raster_path_var.trace_add("write", self._sync_base_raster)

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

    # ==================================================================
    # UI construction — compact 3-row layout for maximum canvas space
    #
    #   Row 1: [Config] Raster:[___][Browse][Load] | Shapefiles:[___][Browse][Load]
    #   Row 2: Start[__] End[__] [Apply] 🔑 [⬇Download] | ▶⏮ ←Skip[n]Skip→ Speed[__] Frame[===slider===]
    #   Row 3: LEFT: Ref:[___][Browse] Out:[___][Browse] [Accum&Rast]  |  RIGHT: Pan Zoom+ Zoom- ⌂ ☑Fire ☑Bg
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
        ttk.Entry(row1, textvariable=self._raster_path_var, width=28).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(row1, text="Browse",
                   command=self._browse_raster).pack(side=tk.LEFT)
        ttk.Button(row1, text="Load",
                   command=self._load_raster).pack(side=tk.LEFT, padx=(2, 0))
        ttk.Button(row1, text="\u2715", width=2,
                   command=lambda: self._raster_path_var.set("")).pack(
            side=tk.LEFT, padx=(0, 2))

        ttk.Separator(row1, orient=tk.VERTICAL).pack(
            side=tk.LEFT, padx=6, fill=tk.Y, pady=1)

        ttk.Label(row1, text="Shapefiles:", font=_font).pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=self._shapefile_dir_var, width=28).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(row1, text="Browse",
                   command=self._browse_shapefile_dir).pack(side=tk.LEFT)
        ttk.Button(row1, text="Load",
                   command=self._load_shapefiles).pack(side=tk.LEFT, padx=(2, 0))
        ttk.Button(row1, text="\u2715", width=2,
                   command=lambda: self._shapefile_dir_var.set("")).pack(
            side=tk.LEFT, padx=(0, 2))

        # ==============================================================
        # ROW 2 — Date / Download  |  Playback (all one line)
        # ==============================================================
        row2 = ttk.Frame(ctrl)
        row2.pack(fill=tk.X, pady=1)

        # -- Date / Download cluster --
        ttk.Label(row2, text="Start:", font=_font).pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self._start_date_var, width=11).pack(
            side=tk.LEFT, padx=2)
        ttk.Label(row2, text="End:", font=_font).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Entry(row2, textvariable=self._end_date_var, width=11).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(row2, text="Apply",
                   command=self._apply_date_filter).pack(side=tk.LEFT, padx=2)

        self._token_btn = tk.Button(
            row2, text="\U0001f511", font=("TkDefaultFont", 10),
            width=2, relief=tk.RAISED, cursor="hand2",
            command=self._manage_token,
        )
        self._token_btn.pack(side=tk.LEFT, padx=(4, 0))
        self._update_token_indicator()

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
        ttk.Entry(acc_frame, textvariable=self._base_raster_var,
                  width=20).pack(side=tk.LEFT, padx=2)
        ttk.Button(acc_frame, text="Browse",
                   command=self._browse_base_raster).pack(side=tk.LEFT)
        ttk.Button(acc_frame, text="\u2715", width=2,
                   command=lambda: self._base_raster_var.set("")).pack(
            side=tk.LEFT, padx=(0, 4))

        ttk.Label(acc_frame, text="Out:", font=_font).pack(side=tk.LEFT)
        ttk.Entry(acc_frame, textvariable=self._acc_output_dir_var,
                  width=20).pack(side=tk.LEFT, padx=2)
        ttk.Button(acc_frame, text="Browse",
                   command=self._browse_acc_output_dir).pack(side=tk.LEFT)
        ttk.Button(acc_frame, text="\u2715", width=2,
                   command=lambda: self._acc_output_dir_var.set("")).pack(
            side=tk.LEFT, padx=(0, 4))

        self._acc_btn = tk.Button(
            acc_frame, text="Accumulate & Rasterize",
            bg="#4CAF50", fg="white",
            font=_font + ("bold",),
            activebackground="#388E3C",
            command=self._confirm_accumulate_rasterize,
        )
        self._acc_btn.pack(side=tk.LEFT, padx=4)

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

        _nav_btn("Pan",    NAV_PAN)
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
            nav_frame, text="Fire",
            variable=self._show_fire_var,
            command=self._toggle_fire,
            indicatoron=True, selectcolor="#22cc22",
            bg=_bg, activebackground=_bg, font=_font,
        )
        self._fire_chk.pack(side=tk.LEFT, padx=1)

        self._show_raster_var = tk.BooleanVar(value=True)
        self._raster_chk = tk.Checkbutton(
            nav_frame, text="Bg",
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

    def _manage_token(self):
        """Popup to browse for / set the LAADS DAAC token."""
        popup = tk.Toplevel(self._root)
        popup.title("LAADS DAAC Token")
        popup.transient(self._root)
        popup.grab_set()

        w, h = 500, 180
        sx = self._root.winfo_screenwidth()
        sy = self._root.winfo_screenheight()
        popup.geometry(f"{w}x{h}+{(sx-w)//2}+{(sy-h)//2}")
        popup.resizable(False, False)

        ttk.Label(popup, text="LAADS DAAC Authentication Token",
                  font=("TkDefaultFont", 11, "bold")).pack(pady=(12, 6))

        f1 = ttk.Frame(popup)
        f1.pack(fill=tk.X, padx=12, pady=4)
        ttk.Label(f1, text="Token file:").pack(side=tk.LEFT)
        token_path_var = tk.StringVar()
        ttk.Entry(f1, textvariable=token_path_var, width=36).pack(
            side=tk.LEFT, padx=4)

        def _browse_token():
            f = filedialog.askopenfilename(
                parent=popup, title="Select token file",
                filetypes=[("All files", "*.*")])
            if f:
                token_path_var.set(f)
                try:
                    with open(f, "r") as fh:
                        self._token = fh.read().strip()
                        self._token_loaded = bool(self._token)
                except Exception:
                    self._token_loaded = False

        ttk.Button(f1, text="Browse", command=_browse_token).pack(
            side=tk.LEFT, padx=4)

        f2 = ttk.Frame(popup)
        f2.pack(fill=tk.X, padx=12, pady=4)
        ttk.Label(f2, text="Or paste:").pack(side=tk.LEFT)
        paste_var = tk.StringVar(value=self._token if self._token else "")
        ttk.Entry(f2, textvariable=paste_var, width=40, show="*").pack(
            side=tk.LEFT, padx=4, fill=tk.X, expand=True)

        def _apply_token():
            pasted = paste_var.get().strip()
            if pasted:
                self._token = pasted
                self._token_loaded = True
            self._update_token_indicator()
            popup.destroy()

        btn_f = ttk.Frame(popup)
        btn_f.pack(pady=8)
        tk.Button(
            btn_f, text="  \u2714 Set Token  ", bg="#4CAF50", fg="white",
            font=("TkDefaultFont", 9, "bold"), activebackground="#388E3C",
            command=_apply_token,
        ).pack(side=tk.LEFT, padx=8)
        tk.Button(
            btn_f, text="  Cancel  ", bg="#F44336", fg="white",
            font=("TkDefaultFont", 9, "bold"), activebackground="#C62828",
            command=popup.destroy,
        ).pack(side=tk.LEFT, padx=8)

    def _update_token_indicator(self):
        if self._token_loaded and self._token:
            self._token_btn.configure(bg="#4CAF50", fg="white",
                                      activebackground="#388E3C")
        else:
            self._token_btn.configure(bg="#F44336", fg="white",
                                      activebackground="#C62828")

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
            # Re-compute scatter size if raster is loaded
            if self._raster_loaded:
                scatter_sz = self._raster_loader.compute_scatter_size()
                self._canvas.scatter_size = scatter_sz
            else:
                self._canvas.scatter_size = cfg.DEFAULT_SCATTER_SIZE
        if self._animator and self._animator.current_date:
            self._on_animation_frame(self._animator.current_date)

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
    # Working directory & base raster sync
    # ==================================================================

    def _set_working_dir(self, raster_path: str):
        """Set the working directory to the parent of the raster file."""
        self._working_dir = os.path.dirname(os.path.abspath(raster_path))

        # Auto-fill shapefile dir to working_dir/VNP14IMG if it exists
        vnp_dir = os.path.join(self._working_dir, "VNP14IMG")
        if os.path.isdir(vnp_dir) and not self._shapefile_dir_var.get().strip():
            self._shapefile_dir_var.set(vnp_dir)

    def _sync_base_raster(self, *_):
        """One-way: mirror the visualization raster to Reference Raster."""
        rp = self._raster_path_var.get().strip()
        if rp:
            self._base_raster_var.set(rp)

    def _get_initial_browse_dir(self) -> str:
        """Return working dir for browse dialogs, or home."""
        if self._working_dir and os.path.isdir(self._working_dir):
            return self._working_dir
        return os.path.expanduser("~")

    # ==================================================================
    # Browse helpers (all use custom browser, pass current_value)
    # ==================================================================

    def _browse_raster(self):
        f = browse_file(
            self._root,
            title="Select raster file",
            initial_dir=self._get_initial_browse_dir(),
            filetypes=[
                ("ENVI binary", "*.bin"),
                ("ENVI header", "*.hdr"),
                ("ENVI data", "*.dat *.img *.bsq *.bil *.bip"),
                ("GeoTIFF", "*.tif *.tiff"),
                ("All files", "*.*"),
            ],
            current_value=self._raster_path_var.get().strip(),
        )
        if f:
            self._raster_path_var.set(f)

    def _browse_shapefile_dir(self):
        d = browse_directory(
            self._root,
            title="Select shapefile directory",
            initial_dir=self._get_initial_browse_dir(),
            current_value=self._shapefile_dir_var.get().strip(),
        )
        if d:
            self._shapefile_dir_var.set(d)

    def _browse_base_raster(self):
        f = browse_file(
            self._root,
            title="Select reference raster for rasterization",
            initial_dir=self._get_initial_browse_dir(),
            filetypes=[
                ("ENVI binary", "*.bin"),
                ("ENVI header", "*.hdr"),
                ("GeoTIFF", "*.tif *.tiff"),
                ("All files", "*.*"),
            ],
            allow_create_folder=True,
            current_value=self._base_raster_var.get().strip(),
        )
        if f:
            self._base_raster_var.set(f)

    def _browse_acc_output_dir(self):
        d = browse_directory(
            self._root,
            title="Select output directory",
            initial_dir=self._get_initial_browse_dir(),
            allow_create_folder=True,
            current_value=self._acc_output_dir_var.get().strip(),
        )
        if d:
            self._acc_output_dir_var.set(d)

    # ==================================================================
    # Loading (raster must be loaded first)
    # ==================================================================

    def _load_raster(self):
        raster_path = self._raster_path_var.get().strip()
        if not raster_path or not os.path.exists(raster_path):
            messagebox.showerror("Error", "Please select a valid raster file.")
            return

        # Preserve current animation position
        self._save_slider_position()

        self._status_var.set("Loading raster\u2026")
        self._root.update_idletasks()

        try:
            img = self._raster_loader.load(raster_path)
            ext = self._raster_loader.extent
            self._canvas.display_raster(img, ext)
            self._raster_loaded = True

            # Set working directory
            self._set_working_dir(raster_path)

            # Compute scatter size from raster resolution
            scatter_sz = self._raster_loader.compute_scatter_size()
            cfg.DEFAULT_SCATTER_SIZE = scatter_sz
            self._canvas.scatter_size = scatter_sz

            pixel_m = self._raster_loader.pixel_size_m
            self._status_var.set(
                f"Raster loaded: {os.path.basename(raster_path)}  "
                f"({self._raster_loader._raster._xSize}"
                f"\u00d7{self._raster_loader._raster._ySize})  "
                f"Pixel: {pixel_m:.1f}m  Scatter: {scatter_sz}"
            )
        except Exception as exc:
            self._raster_loaded = False
            messagebox.showerror("Raster Error",
                                 f"Could not load raster:\n{exc}")
            self._status_var.set("Raster load failed.")
            return

        # Store CRS info for download
        self._store_raster_crs_info(raster_path)

        self._update_crs_display()

        if (self._data_mgr.master_gdf is not None
                and not self._data_mgr.master_gdf.empty):
            self._reclip_and_refresh()

        # Restore slider position
        self._restore_slider_position()

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
            messagebox.showwarning("Warning", "No matching shapefiles found.")
            self._status_var.set("No shapefiles found.")
            return

        def _lp(loaded, total):
            self._status_var.set(f"Loading shapefiles\u2026 {loaded}/{total}")
            self._root.update_idletasks()

        raster_crs = self._raster_loader.crs if self._raster_loaded else None
        gdf = self._data_mgr.load_all(progress_cb=_lp, target_crs=raster_crs)

        if gdf.empty:
            messagebox.showwarning("Warning", "All shapefiles were empty.")
            self._status_var.set("No data loaded.")
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

    def _apply_date_filter(self):
        """Apply date filter -- works even without shapefiles loaded."""
        start_str = self._start_date_var.get().strip()
        end_str = self._end_date_var.get().strip()
        try:
            start = date.fromisoformat(start_str)
            end = date.fromisoformat(end_str)
        except ValueError:
            messagebox.showerror("Error", "Dates must be YYYY-MM-DD format.")
            return
        if start > end:
            messagebox.showerror("Error",
                                 "Start date must be before end date.")
            return

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

        scatter_sz = (self._raster_loader.compute_scatter_size()
                      if self._raster_loaded else cfg.DEFAULT_SCATTER_SIZE)
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
        scatter_sz = (self._raster_loader.compute_scatter_size()
                      if self._raster_loaded else cfg.DEFAULT_SCATTER_SIZE)
        self._canvas.scatter_size = scatter_sz
        self._canvas.update_scatter(
            fd.x, fd.y, fd.ages, fd.indices,
            n_levels=cfg.N_COLOUR_LEVELS,
        )
        self._update_viewport_pixel_count()

    # -- Animation frame --

    def _on_animation_frame(self, current_date: date):
        fd = self._data_mgr.get_frame(current_date)

        scatter_sz = (self._raster_loader.compute_scatter_size()
                      if self._raster_loaded else cfg.DEFAULT_SCATTER_SIZE)
        self._canvas.scatter_size = scatter_sz
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

        if not self._token:
            messagebox.showerror(
                "Token Required",
                "Set a LAADS DAAC token first (click the key icon).")
            return

        start_str = self._start_date_var.get().strip()
        end_str = self._end_date_var.get().strip()
        try:
            start_dt = datetime.datetime.strptime(start_str, "%Y-%m-%d")
            end_dt = datetime.datetime.strptime(end_str, "%Y-%m-%d")
        except ValueError:
            messagebox.showerror("Error", "Dates must be YYYY-MM-DD format.")
            return
        if start_dt > end_dt:
            messagebox.showerror("Error", "Start must be before End.")
            return

        if self._ref_wkt is None:
            messagebox.showerror("Error",
                                 "Could not read CRS from raster.")
            return

        if self._download_thread and self._download_thread.is_alive():
            messagebox.showwarning("Busy", "Download already in progress.")
            return

        # Compute bbox
        raster_path = self._raster_path_var.get().strip()
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

        save_dir = self._working_dir or os.path.dirname(raster_path)

        # Confirmation popup
        popup = tk.Toplevel(self._root)
        popup.title("Confirm Download")
        popup.transient(self._root)
        popup.grab_set()

        w, h = 480, 200
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
        _row("Save to:", save_dir)

        btn_f = ttk.Frame(popup)
        btn_f.pack(pady=10)

        def _go():
            popup.destroy()
            self._run_download(
                self._token, start_dt, end_dt, save_dir, raster_path,
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

    def _run_download(self, token, start_dt, end_dt, save_dir, ref_path,
                      west, south, east, north):
        self._download_btn.configure(state=tk.DISABLED)
        self._download_cancel.clear()

        self._download_thread = threading.Thread(
            target=self._download_worker,
            args=(token, start_dt, end_dt, save_dir, ref_path,
                  west, south, east, north),
            daemon=True,
        )
        self._download_thread.start()

    def _download_worker(self, token, start_dt, end_dt, save_dir, ref_path,
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
        lock = threading.Lock()

        def download_one(download_day):
            nonlocal completed
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
                save_dir, product, f"{year:04d}", f"{jday:03d}")
            os.makedirs(download_path, exist_ok=True)

            print(f"\n[DOWNLOAD] {download_day.strftime('%Y-%m-%d')}")
            print(f"  URL: {download_url}")
            print(f"  Dir: {download_path}")

            try:
                sync_fn(download_url, download_path, token)
            except Exception as exc:
                print(f"[WARN] Download error for {download_day}: {exc}")

            with lock:
                nonlocal completed
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

        _status("Download complete. Running shapify\u2026")
        print(f"\n[INFO] Download finished.")

        self._run_shapify(save_dir, ref_path, west, south, east, north,
                          _status)

    def _run_shapify(self, save_dir, ref_path, west, south, east, north,
                     _status):
        if self._download_cancel.is_set():
            self._download_finish()
            return

        _status("Running shapify (converting .nc \u2192 .shp)\u2026")
        print("\n[INFO] Starting shapify\u2026")

        cmd = [
            sys.executable, "-m", "viirs.utils.shapify",
            save_dir,
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
                vnp_dir = os.path.join(save_dir, "VNP14IMG")
                n_shp = 0
                if os.path.isdir(vnp_dir):
                    for root, dirs, files in os.walk(vnp_dir):
                        for fn in files:
                            if fn.lower().endswith(".shp"):
                                n_shp += 1

                _status(f"Download & shapify complete! {n_shp} shapefiles created.")
                print("[INFO] Shapify finished successfully.")

                # Auto-fill shapefile dir
                if os.path.isdir(vnp_dir):
                    try:
                        self._root.after(0, lambda: self._shapefile_dir_var.set(
                            vnp_dir))
                    except Exception:
                        pass

                # Show completion popup
                start_str = self._start_date_var.get().strip()
                end_str = self._end_date_var.get().strip()
                self._show_done_popup(
                    "Download Complete  \u2714",
                    f"Data downloaded and converted successfully.\n\n"
                    f"Shapefiles created: {n_shp}\n"
                    f"Date range: {start_str}  \u2192  {end_str}\n\n"
                    f"Save directory:\n{save_dir}")
            else:
                _status(f"Shapify exited with code {proc.returncode}.")
                print(f"[WARN] Shapify exit code: {proc.returncode}")
        except FileNotFoundError:
            _status("ERROR: shapify command not found.")
        except Exception as exc:
            _status(f"Shapify error: {exc}")

        self._download_finish()

    def _download_finish(self):
        try:
            self._root.after(0, lambda: self._download_btn.configure(
                state=tk.NORMAL))
        except Exception:
            pass

    # ==================================================================
    # Accumulate + Rasterize
    # ==================================================================

    def _confirm_accumulate_rasterize(self):
        """Show a confirmation popup, then run in a background thread."""
        if not self._raster_loaded:
            messagebox.showwarning(
                "Raster Required",
                "Load a raster image first.")
            return

        shp_dir = self._shapefile_dir_var.get().strip()
        if not shp_dir or not os.path.isdir(shp_dir):
            messagebox.showerror("Error",
                                 "Load shapefiles first (set shapefile dir).")
            return

        base_raster = self._base_raster_var.get().strip()
        if not base_raster or not os.path.exists(base_raster):
            messagebox.showerror("Error",
                                 "Set a valid reference raster for rasterization.")
            return

        out_dir = self._acc_output_dir_var.get().strip()
        if not out_dir:
            messagebox.showerror("Error", "Set an output directory.")
            return

        start_str = self._start_date_var.get().strip()
        end_str = self._end_date_var.get().strip()
        try:
            date.fromisoformat(start_str)
            date.fromisoformat(end_str)
        except ValueError:
            messagebox.showerror("Error", "Dates must be YYYY-MM-DD.")
            return

        if self._acc_thread and self._acc_thread.is_alive():
            messagebox.showwarning("Busy",
                                   "Accumulate & Rasterize is already running.")
            return

        # Confirmation popup
        popup = tk.Toplevel(self._root)
        popup.title("Confirm Accumulate & Rasterize")
        popup.transient(self._root)
        popup.grab_set()

        w, h = 520, 220
        sx = self._root.winfo_screenwidth()
        sy = self._root.winfo_screenheight()
        popup.geometry(f"{w}x{h}+{(sx-w)//2}+{(sy-h)//2}")
        popup.resizable(False, False)

        pad = dict(padx=12, pady=4)
        ttk.Label(popup, text="Accumulate & Rasterize",
                  font=("TkDefaultFont", 12, "bold")).pack(pady=(12, 6))

        info = ttk.Frame(popup)
        info.pack(fill=tk.X, **pad)

        def _row(label, value):
            f = ttk.Frame(info)
            f.pack(fill=tk.X, pady=1)
            ttk.Label(f, text=label, width=18, anchor="e",
                      font=("TkDefaultFont", 9, "bold")).pack(side=tk.LEFT)
            ttk.Label(f, text=value, anchor="w",
                      font=("TkDefaultFont", 9)).pack(
                side=tk.LEFT, padx=6)

        _row("Shapefile dir:", shp_dir)
        _row("Reference raster:", os.path.basename(base_raster))
        _row("Date range:", f"{start_str}  \u2192  {end_str}")
        _row("Output dir:", out_dir)

        btn_frame = ttk.Frame(popup)
        btn_frame.pack(pady=12)

        def _go():
            popup.destroy()
            self._run_accumulate_rasterize(
                shp_dir, base_raster, out_dir, start_str, end_str)

        tk.Button(
            btn_frame, text="  \u2714  Confirm  ", bg="#4CAF50", fg="white",
            font=("TkDefaultFont", 10, "bold"), activebackground="#388E3C",
            command=_go,
        ).pack()

    def _run_accumulate_rasterize(self, shp_dir, base_raster, out_dir,
                                  start_str, end_str):
        self._acc_btn.configure(state=tk.DISABLED)
        self._acc_thread = threading.Thread(
            target=self._acc_worker,
            args=(shp_dir, base_raster, out_dir, start_str, end_str),
            daemon=True,
        )
        self._acc_thread.start()

    def _acc_worker(self, shp_dir, base_raster, out_dir, start_str, end_str):
        def _status(msg):
            try:
                self._root.after(0, lambda: self._status_var.set(msg))
            except Exception:
                pass

        # Import accumulate
        try:
            from viirs.utils.accumulate import accumulate
            accumulate_fn = accumulate
        except ImportError:
            _status("ERROR: Could not import accumulate.accumulate")
            self._acc_finish()
            return

        # Import rasterize
        try:
            from viirs.utils.rasterize import rasterize_shapefile
            rasterize_fn = rasterize_shapefile
        except ImportError:
            _status("ERROR: Could not import rasterize.rasterize_shapefile")
            self._acc_finish()
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
            self._acc_finish()
            return

        if not shp_paths:
            _status("Accumulation produced no files.")
            self._acc_finish()
            return

        n_shp = len(shp_paths)
        _status(f"Accumulated {n_shp} shapefiles. Rasterizing\u2026")

        # Step 2: Rasterize in parallel
        raster_out_dir = out_dir
        rasterized = 0
        errors = 0

        def _rast_one(shp_path):
            try:
                return rasterize_fn(shp_path, base_raster, raster_out_dir)
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

        _status(
            f"Done: {n_shp} shapefiles + {rasterized} rasters "
            f"saved to {out_dir}"
        )
        print(f"[ACCUM & RASTERIZE] DONE")
        self._show_done_popup(
            "Accumulate & Rasterize Complete  \u2714",
            f"Pipeline finished successfully.\n\n"
            f"Shapefiles: {n_shp}\n"
            f"Rasters: {rasterized}\n"
            f"Date range: {start_str}  \u2192  {end_str}\n\n"
            f"Output directory:\n{out_dir}")
        self._acc_finish()

    def _acc_finish(self):
        try:
            self._root.after(0, lambda: self._acc_btn.configure(
                state=tk.NORMAL))
        except Exception:
            pass

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