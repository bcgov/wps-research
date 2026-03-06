"""
FireAccumulationGUI: the main tkinter application window.

RENDERING RESOURCE KNOBS
=========================
1. config.py — MAX_WORKERS (parallel shapefile loading)
2. config.py — DEFAULT_ANIMATION_INTERVAL_MS
3. config.py — N_COLOUR_LEVELS
4. config.py — DEFAULT_SCATTER_SIZE
5. config.py — MAX_RASTER_DISPLAY_DIM  (raster downsampling for speed)
6. fire_map_canvas.py — figsize / dpi
7. fire_map_canvas.py — blitting
8. fire_data_manager.py — precompute_frames  (comment out to save RAM)
9. fire_data_manager.py — numba cache
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import date

from config import DEFAULT_SCATTER_SIZE, DEFAULT_ANIMATION_INTERVAL_MS, N_COLOUR_LEVELS
from fire_data_manager import FireDataManager
from raster_loader import RasterLoader
from fire_map_canvas import FireMapCanvas, NAV_ZOOM_IN, NAV_ZOOM_OUT, NAV_PAN
from fire_animation_controller import FireAnimationController


class FireAccumulationGUI:
    """
    Top-level GUI for the VIIRS fire pixel accumulation viewer.

    Navigation:
        Pan        — drag to move the map
        Zoom +     — draw green rectangle to zoom in
        Zoom −     — draw red rectangle to zoom out
        Scroll     — zoom in / out at cursor (any mode)
        Right-click — pixel detail popup (any mode)
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

        # tk variables
        self._shapefile_dir_var = tk.StringVar()
        self._raster_path_var = tk.StringVar()
        self._start_date_var = tk.StringVar()
        self._end_date_var = tk.StringVar()
        self._scatter_size_var = tk.IntVar(value=DEFAULT_SCATTER_SIZE)
        self._interval_var = tk.IntVar(value=DEFAULT_ANIMATION_INTERVAL_MS)
        self._n_levels_var = tk.IntVar(value=N_COLOUR_LEVELS)
        self._status_var = tk.StringVar(value="Ready")
        self._date_label_var = tk.StringVar(value="Date: —")
        self._frame_label_var = tk.StringVar(value="Frame: 0 / 0")
        self._pixel_count_var = tk.StringVar(value="Pixels: 0")
        self._viewport_pixel_var = tk.StringVar(value="In view: 0")
        self._updating_slider = False

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

        self._build_ui()

        self._animator = FireAnimationController(
            self._root,
            on_frame=self._on_animation_frame,
            on_finished=self._on_animation_finished,
        )

        self._canvas.set_on_viewport_changed(self._on_viewport_changed)

    # ==================================================================
    # UI construction
    # ==================================================================

    def _build_ui(self):
        # Controls frame (no border — plain ttk.Frame, not LabelFrame)
        ctrl = ttk.Frame(self._root, padding=8)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        # Collapsible file + date rows
        self._setup_frame = ttk.Frame(ctrl)
        self._setup_frame.pack(fill=tk.X)
        self._setup_visible = True

        self._build_file_controls(self._setup_frame)
        self._build_date_controls(self._setup_frame)
        self._build_playback_controls(ctrl)
        self._build_nav_and_display_controls(ctrl)

        # Collapse handle
        handle = ttk.Frame(ctrl)
        handle.pack(fill=tk.X, pady=(2, 0))
        self._collapse_btn = tk.Button(
            handle, text="▲", command=self._toggle_setup,
            relief=tk.FLAT, bd=0, padx=20, pady=0,
            font=("TkDefaultFont", 7), fg="#888888",
            activeforeground="#444444", cursor="hand2",
        )
        self._collapse_btn.pack(anchor=tk.CENTER)
        try:
            _bg = handle.winfo_toplevel().cget("background")
            self._collapse_btn.configure(bg=_bg, activebackground=_bg)
        except Exception:
            pass

        # Status bar (before canvas so it always claims space)
        status = ttk.Frame(self._root, padding=4)
        status.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(status, textvariable=self._status_var).pack(side=tk.LEFT)
        ttk.Label(status, textvariable=self._viewport_pixel_var).pack(side=tk.RIGHT, padx=15)
        ttk.Label(status, textvariable=self._pixel_count_var).pack(side=tk.RIGHT, padx=15)
        ttk.Label(status, textvariable=self._frame_label_var).pack(side=tk.RIGHT, padx=15)
        ttk.Label(status, textvariable=self._date_label_var).pack(side=tk.RIGHT, padx=15)

        # Canvas (fills everything remaining)
        canvas_frame = ttk.Frame(self._root)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        self._canvas = FireMapCanvas(canvas_frame, figsize=(12, 7), dpi=100)

    # ── File row ──

    def _build_file_controls(self, parent):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)

        ttk.Label(row, text="Raster:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self._raster_path_var, width=32).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Browse…", command=self._browse_raster).pack(side=tk.LEFT)
        ttk.Button(row, text="Load Raster", command=self._load_raster).pack(side=tk.LEFT, padx=4)

        ttk.Separator(row, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=8, fill=tk.Y)

        ttk.Label(row, text="Shapefiles:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self._shapefile_dir_var, width=32).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Browse…", command=self._browse_shapefile_dir).pack(side=tk.LEFT)
        ttk.Button(row, text="Load Shapefiles", command=self._load_shapefiles).pack(side=tk.LEFT, padx=4)

    # ── Date row ──

    def _build_date_controls(self, parent):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)

        ttk.Label(row, text="Start Date (YYYY-MM-DD)").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self._start_date_var, width=12).pack(side=tk.LEFT, padx=4)
        ttk.Label(row, text="End Date (YYYY-MM-DD)").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Entry(row, textvariable=self._end_date_var, width=12).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Apply Filter", command=self._apply_date_filter).pack(side=tk.LEFT, padx=8)

        # Right side: scatter size & colour levels
        ttk.Spinbox(row, from_=10, to=500, increment=10,
                     textvariable=self._n_levels_var, width=5).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Label(row, text="Colour Levels").pack(side=tk.RIGHT)
        ttk.Separator(row, orient=tk.VERTICAL).pack(side=tk.RIGHT, padx=8, fill=tk.Y)
        ttk.Spinbox(row, from_=1, to=200, increment=1,
                     textvariable=self._scatter_size_var, width=5,
                     command=self._update_scatter_size).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Label(row, text="Scatter Size").pack(side=tk.RIGHT)

    # ── Playback rows ──

    def _build_playback_controls(self, parent):
        self._playback_row = ttk.Frame(parent)
        self._playback_row.pack(fill=tk.X, pady=2)

        self._play_btn = ttk.Button(self._playback_row, text="▶  Play",
                                     command=self._toggle_play, width=10)
        self._play_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(self._playback_row, text="← −1 Day",
                   command=self._step_back, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(self._playback_row, text="+1 Day →",
                   command=self._step_fwd, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(self._playback_row, text="⏹  Reset",
                   command=self._reset_animation, width=9).pack(side=tk.LEFT, padx=2)

        ttk.Separator(self._playback_row, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=8, fill=tk.Y)
        ttk.Label(self._playback_row, text="Speed (ms):").pack(side=tk.LEFT)
        ttk.Spinbox(self._playback_row, from_=50, to=5000, increment=50,
                     textvariable=self._interval_var, width=6,
                     command=self._update_speed).pack(side=tk.LEFT, padx=4)

        ttk.Separator(self._playback_row, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=8, fill=tk.Y)
        ttk.Label(self._playback_row, text="Frame").pack(side=tk.LEFT)
        self._frame_slider = ttk.Scale(
            self._playback_row, from_=0, to=1, orient=tk.HORIZONTAL,
            command=self._on_slider)
        self._frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        # Row 2: skip N days
        row2 = ttk.Frame(parent)
        row2.pack(fill=tk.X, pady=2)
        ttk.Button(row2, text="← Skip N Days", command=self._skip_back_n,
                   width=14).pack(side=tk.LEFT, padx=2)
        self._skip_n_var = tk.IntVar(value=7)
        ttk.Spinbox(row2, from_=1, to=365, increment=1,
                     textvariable=self._skip_n_var, width=5).pack(side=tk.LEFT, padx=4)
        ttk.Button(row2, text="Skip N Days →", command=self._skip_fwd_n,
                   width=14).pack(side=tk.LEFT, padx=2)
        ttk.Label(row2, text="(days)", font=("TkDefaultFont", 9)).pack(side=tk.LEFT, padx=4)

    # ── Nav tools + layer toggles (single row) ──

    def _build_nav_and_display_controls(self, parent):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)

        _bg = "#d9d9d9"
        try:
            _bg = row.winfo_toplevel().cget("background")
        except Exception:
            pass

        _active_bg = "#b0c4ff"
        _font = ("TkDefaultFont", 9)

        def _nav_btn(text, mode):
            btn = tk.Button(
                row, text=text, relief=tk.RAISED, bd=1,
                padx=6, pady=1, font=_font, bg=_bg,
                activebackground=_active_bg, cursor="hand2",
                command=lambda m=mode: self._set_nav_mode(m),
            )
            btn.pack(side=tk.LEFT, padx=1)
            self._nav_buttons[mode] = btn

        _nav_btn("Pan",    NAV_PAN)
        _nav_btn("Zoom +", NAV_ZOOM_IN)
        _nav_btn("Zoom −", NAV_ZOOM_OUT)

        ttk.Separator(row, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=6, fill=tk.Y)

        # View history
        for text, cmd in [("⌂", self._nav_home),
                          ("◀", self._nav_back),
                          ("▶", self._nav_forward)]:
            tk.Button(
                row, text=text, relief=tk.RAISED, bd=1,
                padx=6, pady=1, font=_font, bg=_bg, cursor="hand2",
                command=cmd,
            ).pack(side=tk.LEFT, padx=1)

        ttk.Separator(row, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=6, fill=tk.Y)

        # Hint label
        ttk.Label(row, text="Right-click a pixel for details",
                  font=("TkDefaultFont", 8), foreground="#888888").pack(side=tk.LEFT, padx=6)

        # Layer toggles (far right)
        self._show_raster_var = tk.BooleanVar(value=True)
        self._raster_chk = tk.Checkbutton(
            row, text="Show Background",
            variable=self._show_raster_var,
            command=self._toggle_raster,
            indicatoron=True, selectcolor="#22cc22",
            bg=_bg, activebackground=_bg, font=_font,
        )
        self._raster_chk.pack(side=tk.RIGHT, padx=4)

        self._show_fire_var = tk.BooleanVar(value=True)
        self._fire_chk = tk.Checkbutton(
            row, text="Show Fire Pixels",
            variable=self._show_fire_var,
            command=self._toggle_fire,
            indicatoron=True, selectcolor="#22cc22",
            bg=_bg, activebackground=_bg, font=_font,
        )
        self._fire_chk.pack(side=tk.RIGHT, padx=4)

        # Indicator colour traces
        def _rc(*_):
            c = "#22cc22" if self._show_raster_var.get() else "#111111"
            self._raster_chk.configure(selectcolor=c)
        def _fc(*_):
            c = "#22cc22" if self._show_fire_var.get() else "#111111"
            self._fire_chk.configure(selectcolor=c)
        self._show_raster_var.trace_add("write", _rc)
        self._show_fire_var.trace_add("write", _fc)

        # Default nav = Pan (highlighted)
        self._highlight_nav_button(NAV_PAN)

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

    def _nav_back(self):
        if self._canvas:
            self._canvas.view_back()

    def _nav_forward(self):
        if self._canvas:
            self._canvas.view_forward()

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
    # Loading
    # ==================================================================

    def _browse_shapefile_dir(self):
        d = filedialog.askdirectory(title="Select shapefile directory")
        if d:
            self._shapefile_dir_var.set(d)

    def _browse_raster(self):
        f = filedialog.askopenfilename(
            title="Select raster file",
            filetypes=[
                ("ENVI binary", "*.bin"),
                ("ENVI header", "*.hdr"),
                ("ENVI data", "*.dat *.img *.bsq *.bil *.bip"),
                ("GeoTIFF", "*.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if f:
            self._raster_path_var.set(f)

    def _load_raster(self):
        raster_path = self._raster_path_var.get().strip()
        if not raster_path or not os.path.exists(raster_path):
            messagebox.showerror("Error", "Please select a valid raster file.")
            return

        self._status_var.set("Loading raster…")
        self._root.update_idletasks()

        try:
            img = self._raster_loader.load(raster_path)
            ext = self._raster_loader.extent
            self._canvas.display_raster(img, ext)
            self._raster_loaded = True
            self._status_var.set(
                f"Raster loaded: {os.path.basename(raster_path)}  "
                f"({self._raster_loader._raster._xSize}×{self._raster_loader._raster._ySize})"
            )
        except Exception as exc:
            self._raster_loaded = False
            messagebox.showerror("Raster Error", f"Could not load raster:\n{exc}")
            self._status_var.set("Raster load failed.")
            return

        print(f"[INFO] Raster CRS: {self._raster_loader.crs}")

        if self._data_mgr.master_gdf is not None and not self._data_mgr.master_gdf.empty:
            self._reclip_and_refresh()

    def _load_shapefiles(self):
        shp_dir = self._shapefile_dir_var.get().strip()
        if not shp_dir or not os.path.isdir(shp_dir):
            messagebox.showerror("Error", "Please select a valid shapefile directory.")
            return

        self._status_var.set("Scanning shapefiles…")
        self._root.update_idletasks()

        file_dict = self._data_mgr.scan_directory(shp_dir)
        if not file_dict:
            messagebox.showwarning("Warning", "No matching shapefiles found.")
            self._status_var.set("No shapefiles found.")
            return

        def _lp(loaded, total):
            self._status_var.set(f"Loading shapefiles… {loaded}/{total}")
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
                messagebox.showwarning("Warning", "All pixels fell outside the raster.")
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

        self._status_var.set("Pre-computing frames…")
        self._root.update_idletasks()

        def _pp(done, total):
            self._status_var.set(f"Pre-computing frames… {done}/{total}")
            self._root.update_idletasks()

        self._data_mgr.precompute_frames(progress_cb=_pp)
        self._setup_animator()

        raster_note = (f" ({n_removed} outside raster discarded)"
                       if self._raster_loaded else " (no raster)")
        self._status_var.set(
            f"Loaded {n_kept} pixels{raster_note} "
            f"from {len(file_dict)} files.  {min_d} → {max_d}"
        )

    def _reclip_and_refresh(self):
        ext = self._raster_loader.extent
        if ext is None:
            return

        self._status_var.set("Clipping to raster extent…")
        self._root.update_idletasks()

        raster_crs = self._raster_loader.crs

        def _lp(loaded, total):
            self._status_var.set(f"Reloading shapefiles… {loaded}/{total}")
            self._root.update_idletasks()

        gdf = self._data_mgr.load_all(progress_cb=_lp, target_crs=raster_crs)
        if gdf.empty:
            return

        n_total = len(gdf)
        n_removed = self._data_mgr.clip_to_extent(*ext)
        n_kept = n_total - n_removed

        if n_kept == 0:
            messagebox.showwarning("Warning", "All pixels fell outside the new raster.")
            self._status_var.set("No pixels within raster extent.")
            return

        self._canvas.set_row_lookup(self._data_mgr.get_row_data)
        min_d, max_d = self._data_mgr.get_date_range_bounds()
        self._start_date_var.set(str(min_d))
        self._end_date_var.set(str(max_d))

        self._status_var.set("Pre-computing frames…")
        self._root.update_idletasks()
        self._data_mgr.precompute_frames()

        self._setup_animator()
        self._status_var.set(
            f"Re-clipped: {n_kept} pixels ({n_removed} outside raster).  {min_d} → {max_d}"
        )

    def _apply_date_filter(self):
        start_str = self._start_date_var.get().strip()
        end_str = self._end_date_var.get().strip()
        try:
            start = date.fromisoformat(start_str)
            end = date.fromisoformat(end_str)
        except ValueError:
            messagebox.showerror("Error", "Dates must be YYYY-MM-DD format.")
            return
        if start > end:
            messagebox.showerror("Error", "Start date must be before end date.")
            return

        self._status_var.set("Filtering by date…")
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

        self._status_var.set("Pre-computing frames…")
        self._root.update_idletasks()
        self._data_mgr.precompute_frames()

        self._setup_animator()
        n = len(self._data_mgr.master_gdf) if self._data_mgr.master_gdf is not None else 0
        self._status_var.set(f"Filtered: {n} pixels,  {start} → {end}")

    def _setup_animator(self):
        start_str = self._start_date_var.get().strip()
        end_str = self._end_date_var.get().strip()
        try:
            start = date.fromisoformat(start_str)
            end = date.fromisoformat(end_str)
        except ValueError:
            return
        self._animator.set_date_range(start, end)
        self._frame_slider.configure(to=max(self._animator.total_frames - 1, 1))
        self._frame_label_var.set(f"Frame: 0 / {self._animator.total_frames}")
        self._date_label_var.set(f"Date: {start}")
        self._on_animation_frame(start)

    # ── Collapse ──

    def _toggle_setup(self):
        if self._setup_visible:
            self._setup_frame.pack_forget()
            self._collapse_btn.config(text="▼")
            self._setup_visible = False
        else:
            self._setup_frame.pack(fill=tk.X, before=self._playback_row)
            self._collapse_btn.config(text="▲")
            self._setup_visible = True

    # ── Playback ──

    def _toggle_play(self):
        if self._animator is None:
            return
        if self._animator.is_playing:
            self._animator.pause()
            self._play_btn.config(text="▶  Play")
        else:
            self._animator.interval_ms = self._interval_var.get()
            self._animator.play()
            self._play_btn.config(text="⏸  Pause")

    def _step_fwd(self):
        if self._animator:
            self._animator.step_forward()

    def _step_back(self):
        if self._animator:
            self._animator.step_backward()

    def _skip_fwd_n(self):
        if self._animator:
            self._animator.jump_by(self._skip_n_var.get())

    def _skip_back_n(self):
        if self._animator:
            self._animator.jump_by(-self._skip_n_var.get())

    def _reset_animation(self):
        if self._animator:
            self._animator.stop()
            self._play_btn.config(text="▶  Play")
            self._canvas.clear()
            self._canvas.reset_view()
            self._frame_slider.set(0)
            self._date_label_var.set("Date: —")

    def _update_speed(self):
        if self._animator:
            self._animator.interval_ms = self._interval_var.get()

    def _update_scatter_size(self):
        if self._canvas:
            self._canvas.scatter_size = self._scatter_size_var.get()
            if self._animator and self._animator.current_date:
                self._on_animation_frame(self._animator.current_date)

    def _toggle_raster(self):
        if self._canvas:
            self._canvas.set_raster_visible(self._show_raster_var.get())

    def _toggle_fire(self):
        if self._canvas:
            self._canvas.set_scatter_visible(self._show_fire_var.get())

    # ── Slider ──

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

        self._canvas.scatter_size = self._scatter_size_var.get()
        self._canvas.update_scatter(
            fd.x, fd.y, fd.ages, fd.indices,
            n_levels=self._n_levels_var.get(),
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
        self._canvas.scatter_size = self._scatter_size_var.get()
        self._canvas.update_scatter(
            fd.x, fd.y, fd.ages, fd.indices,
            n_levels=self._n_levels_var.get(),
        )
        self._update_viewport_pixel_count()

    # ── Animation frame ──

    def _on_animation_frame(self, current_date: date):
        fd = self._data_mgr.get_frame(current_date)

        self._canvas.scatter_size = self._scatter_size_var.get()
        self._canvas.update_scatter(
            fd.x, fd.y, fd.ages, fd.indices,
            n_levels=self._n_levels_var.get(),
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
        self._play_btn.config(text="▶  Play")
        self._status_var.set("Animation complete.")

    # ── Lifecycle ──

    def _on_close(self):
        if self._animator:
            self._animator.pause()
        self._root.destroy()

    def run(self):
        self._root.mainloop()