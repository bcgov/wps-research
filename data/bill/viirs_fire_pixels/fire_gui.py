"""
FireAccumulationGUI: the main tkinter application window.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import date

from config import DEFAULT_SCATTER_SIZE, DEFAULT_ANIMATION_INTERVAL_MS, N_COLOUR_LEVELS
from fire_data_manager import FireDataManager
from raster_loader import RasterLoader
from fire_map_canvas import FireMapCanvas
from fire_animation_controller import FireAnimationController


class FireAccumulationGUI:
    """
    Top-level GUI for the VIIRS fire pixel accumulation viewer.

    Usage:
        app = FireAccumulationGUI()
        app.run()
    """

    def __init__(self):
        self._root = tk.Tk()
        self._root.title("VIIRS Fire Pixel Accumulation Viewer")
        self._root.geometry("1800x1000")
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
        self._updating_slider = False  # guard against slider ↔ frame loop

        self._build_ui()

        self._animator = FireAnimationController(
            self._root,
            on_frame=self._on_animation_frame,
            on_finished=self._on_animation_finished,
        )

    # ==================================================================
    # UI
    # ==================================================================

    def _build_ui(self):
        ctrl = ttk.LabelFrame(self._root, text="Controls", padding=8)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        self._build_file_controls(ctrl)
        self._build_date_controls(ctrl)
        self._build_playback_controls(ctrl)
        self._build_display_controls(ctrl)

        canvas_frame = ttk.Frame(self._root)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._canvas = FireMapCanvas(canvas_frame, figsize=(11, 7), dpi=100)

        status = ttk.Frame(self._root, padding=4)
        status.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(status, textvariable=self._status_var).pack(side=tk.LEFT)
        ttk.Label(status, textvariable=self._pixel_count_var).pack(side=tk.RIGHT, padx=15)
        ttk.Label(status, textvariable=self._frame_label_var).pack(side=tk.RIGHT, padx=15)
        ttk.Label(status, textvariable=self._date_label_var).pack(side=tk.RIGHT, padx=15)

    def _build_file_controls(self, parent):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)

        ttk.Label(row, text="Shapefile Dir:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self._shapefile_dir_var, width=40).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Browse…", command=self._browse_shapefile_dir).pack(side=tk.LEFT)

        ttk.Separator(row, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=8, fill=tk.Y)

        ttk.Label(row, text="Raster File:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self._raster_path_var, width=40).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Browse…", command=self._browse_raster).pack(side=tk.LEFT)

        ttk.Separator(row, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=8, fill=tk.Y)
        ttk.Button(row, text="Load Data", command=self._load_data).pack(side=tk.LEFT, padx=4)

    def _build_date_controls(self, parent):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)

        ttk.Label(row, text="Start Date (YYYY-MM-DD):").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self._start_date_var, width=12).pack(side=tk.LEFT, padx=4)

        ttk.Label(row, text="End Date (YYYY-MM-DD):").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Entry(row, textvariable=self._end_date_var, width=12).pack(side=tk.LEFT, padx=4)

        ttk.Button(row, text="Apply Date Filter", command=self._apply_date_filter).pack(side=tk.LEFT, padx=8)

    def _build_playback_controls(self, parent):
        # Row 1: core playback
        row1 = ttk.Frame(parent)
        row1.pack(fill=tk.X, pady=2)

        self._play_btn = ttk.Button(row1, text="▶  Play", command=self._toggle_play, width=10)
        self._play_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="← −1 Day", command=self._step_back, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="+1 Day →", command=self._step_fwd, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="⏹  Reset", command=self._reset_animation, width=9).pack(side=tk.LEFT, padx=2)

        ttk.Separator(row1, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=8, fill=tk.Y)

        ttk.Label(row1, text="Speed (ms):").pack(side=tk.LEFT)
        ttk.Spinbox(row1, from_=50, to=5000, increment=50,
                     textvariable=self._interval_var, width=6,
                     command=self._update_speed).pack(side=tk.LEFT, padx=4)

        ttk.Separator(row1, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=8, fill=tk.Y)
        ttk.Label(row1, text="Frame:").pack(side=tk.LEFT)
        self._frame_slider = ttk.Scale(row1, from_=0, to=1, orient=tk.HORIZONTAL,
                                        command=self._on_slider)
        self._frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        # Row 2: skip N days
        row2 = ttk.Frame(parent)
        row2.pack(fill=tk.X, pady=2)

        ttk.Button(row2, text="← Skip N Days", command=self._skip_back_n, width=14).pack(side=tk.LEFT, padx=2)
        self._skip_n_var = tk.IntVar(value=7)
        ttk.Spinbox(row2, from_=1, to=365, increment=1,
                     textvariable=self._skip_n_var, width=5).pack(side=tk.LEFT, padx=4)
        ttk.Button(row2, text="Skip N Days →", command=self._skip_fwd_n, width=14).pack(side=tk.LEFT, padx=2)

        ttk.Label(row2, text="(days)", font=("TkDefaultFont", 9)).pack(side=tk.LEFT, padx=4)

    def _build_display_controls(self, parent):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)

        ttk.Label(row, text="Scatter Size:").pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=1, to=200, increment=1,
                     textvariable=self._scatter_size_var, width=5,
                     command=self._update_scatter_size).pack(side=tk.LEFT, padx=4)

        ttk.Separator(row, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=8, fill=tk.Y)

        ttk.Label(row, text="Colour Levels:").pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=10, to=500, increment=10,
                     textvariable=self._n_levels_var, width=5).pack(side=tk.LEFT, padx=4)

        ttk.Separator(row, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=8, fill=tk.Y)

        # Checkbox style: green when on, black when off
        _chk_bg = "#d9d9d9"
        try:
            _chk_bg = row.winfo_toplevel().cget("background")
        except Exception:
            pass

        self._show_raster_var = tk.BooleanVar(value=True)
        self._raster_chk = tk.Checkbutton(
            row, text="Show Background Image",
            variable=self._show_raster_var,
            command=self._toggle_raster,
            indicatoron=True,
            selectcolor="#22cc22",
            bg=_chk_bg, activebackground=_chk_bg,
            font=("TkDefaultFont", 9),
        )
        self._raster_chk.pack(side=tk.LEFT, padx=4)

        self._show_fire_var = tk.BooleanVar(value=True)
        self._fire_chk = tk.Checkbutton(
            row, text="Show Fire Pixels",
            variable=self._show_fire_var,
            command=self._toggle_fire,
            indicatoron=True,
            selectcolor="#22cc22",
            bg=_chk_bg, activebackground=_chk_bg,
            font=("TkDefaultFont", 9),
        )
        self._fire_chk.pack(side=tk.LEFT, padx=4)

        # Trace to swap indicator color between green (on) and black (off)
        def _update_raster_color(*_):
            color = "#22cc22" if self._show_raster_var.get() else "#111111"
            self._raster_chk.configure(selectcolor=color)

        def _update_fire_color(*_):
            color = "#22cc22" if self._show_fire_var.get() else "#111111"
            self._fire_chk.configure(selectcolor=color)

        self._show_raster_var.trace_add("write", _update_raster_color)
        self._show_fire_var.trace_add("write", _update_fire_color)

    # ==================================================================
    # Actions
    # ==================================================================

    def _browse_shapefile_dir(self):
        d = filedialog.askdirectory(title="Select shapefile directory")
        if d:
            self._shapefile_dir_var.set(d)

    def _browse_raster(self):
        f = filedialog.askopenfilename(
            title="Select raster file",
            filetypes=[
                ("ENVI header", "*.hdr"),
                ("ENVI data", "*.bin *.dat *.img *.bsq *.bil *.bip"),
                ("GeoTIFF", "*.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if f:
            self._raster_path_var.set(f)

    def _load_data(self):
        shp_dir = self._shapefile_dir_var.get().strip()
        raster_path = self._raster_path_var.get().strip()

        if not shp_dir or not os.path.isdir(shp_dir):
            messagebox.showerror("Error", "Please select a valid shapefile directory.")
            return

        # 1) Try loading raster (optional)
        has_raster = False
        ext = None
        if raster_path and os.path.exists(raster_path):
            self._status_var.set("Loading raster…")
            self._root.update_idletasks()
            try:
                img = self._raster_loader.load(raster_path)
                ext = self._raster_loader.extent
                self._canvas.display_raster(img, ext)
                has_raster = True
            except Exception as exc:
                messagebox.showwarning("Raster Warning", f"Could not load raster:\n{exc}\n\nContinuing with black background.")

        # 2) Load shapefiles (parallel)
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

        gdf = self._data_mgr.load_all(progress_cb=_lp)

        if gdf.empty:
            messagebox.showwarning("Warning", "All shapefiles were empty.")
            self._status_var.set("No data loaded.")
            return

        n_total = len(gdf)
        n_removed = 0

        # 3) Clip to raster extent if we have one, otherwise derive extent from data
        if has_raster and ext is not None:
            left, right, bottom, top = ext
            n_removed = self._data_mgr.clip_to_extent(left, right, bottom, top)
            n_kept = n_total - n_removed

            if n_kept == 0:
                messagebox.showwarning("Warning", "All pixels fell outside the raster.")
                self._status_var.set("No pixels within raster extent.")
                return
        else:
            # No raster — derive extent from fire pixel coordinates with padding
            n_kept = n_total
            data_ext = self._data_mgr.get_data_extent()
            if data_ext is not None:
                self._canvas.set_black_background(data_ext)

        # 4) Wire up popup lookup
        self._canvas.set_row_lookup(self._data_mgr.get_row_data)

        # 5) Date entries
        min_d, max_d = self._data_mgr.get_date_range_bounds()
        self._start_date_var.set(str(min_d))
        self._end_date_var.set(str(max_d))

        # 6) Precompute frames
        self._status_var.set("Pre-computing frames…")
        self._root.update_idletasks()

        def _pp(done, total):
            self._status_var.set(f"Pre-computing frames… {done}/{total}")
            self._root.update_idletasks()

        self._data_mgr.precompute_frames(progress_cb=_pp)

        # 7) Setup animator
        self._setup_animator()

        raster_note = f" ({n_removed} outside raster discarded)" if has_raster else " (no raster, black background)"
        self._status_var.set(
            f"Loaded {n_kept} pixels{raster_note} "
            f"from {len(file_dict)} files.  {min_d} → {max_d}"
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
        if ext is not None and self._raster_loader.image is not None:
            # Raster exists — clip to it
            self._data_mgr.clip_to_extent(*ext)
        else:
            # No raster — update black background extent from new data
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

    # ---- Playback ----

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
            n = self._skip_n_var.get()
            self._animator.jump_by(n)

    def _skip_back_n(self):
        if self._animator:
            n = self._skip_n_var.get()
            self._animator.jump_by(-n)

    def _reset_animation(self):
        if self._animator:
            self._animator.stop()
            self._play_btn.config(text="▶  Play")
            self._canvas.clear()
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

    def _on_slider(self, val):
        if self._updating_slider:
            return
        if self._animator and not self._animator.is_playing:
            self._animator.jump_to(int(float(val)))

    # ---- Animation frame ----

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

        self._updating_slider = True
        self._frame_slider.set(idx)
        self._updating_slider = False

    def _on_animation_finished(self):
        self._play_btn.config(text="▶  Play")
        self._status_var.set("Animation complete.")

    # ---- Lifecycle ----

    def _on_close(self):
        if self._animator:
            self._animator.pause()
        self._root.destroy()

    def run(self):
        self._root.mainloop()