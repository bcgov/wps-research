"""
viirs/fp_gui/fire_map_canvas.py

FireMapCanvas: matplotlib figure embedded in a tkinter frame.
Renders using pure numpy arrays — no pandas during animation.

Performance:
    - Blitting: the static background (raster + axes) is cached as a bitmap.
      Each animation frame only redraws the scatter artist on top.
    - Raster independence: toggling the raster off *removes* the imshow
      artist entirely (not just set_visible), so matplotlib never touches
      it during animation frames.  Toggling it back on recreates the artist
      from cached image data.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap

from config import (
    N_COLOUR_LEVELS,
    COLOUR_NEWEST,
    COLOUR_OLDEST,
    DEFAULT_SCATTER_SIZE,
    RASTER_ALPHA,
    RASTER_CMAP,
    DEFAULT_POPUP_COLUMNS,
)


def _build_colour_table(n_levels, newest_rgba, oldest_rgba):
    """
    Build an (n_levels, 4) RGBA lookup table.

    Index 0 = newest (red), index n_levels-1 = oldest (pale yellow).
    Each day ages the pixel by one index position.
    """
    table = np.zeros((n_levels, 4), dtype=np.float32)
    for c in range(4):
        table[:, c] = np.linspace(newest_rgba[c], oldest_rgba[c], n_levels)
    return table


class FireMapCanvas:
    """
    Embeds a matplotlib figure inside a tkinter parent widget.
    Accepts pure numpy arrays for rendering — zero pandas per frame.
    """

    def __init__(self, parent: tk.Widget, figsize=(10, 8), dpi=100):
        self._parent = parent
        self._scatter_size = DEFAULT_SCATTER_SIZE
        self._popup_columns: List[str] = list(DEFAULT_POPUP_COLUMNS)

        # Build discrete colour lookup table
        self._n_levels = N_COLOUR_LEVELS
        self._colour_table = _build_colour_table(
            self._n_levels, COLOUR_NEWEST, COLOUR_OLDEST
        )

        # Matplotlib figure — fixed axes positions
        self._fig = plt.figure(figsize=figsize, dpi=dpi)
        self._ax = self._fig.add_axes([0.01, 0.01, 0.86, 0.98])
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        for spine in self._ax.spines.values():
            spine.set_visible(False)
        self._cbar_ax = self._fig.add_axes([0.90, 0.05, 0.02, 0.90])
        self._cbar_ax.set_visible(False)

        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._toolbar = NavigationToolbar2Tk(self._canvas, parent)
        self._toolbar.update()

        # ---- Raster state (stored independently of the matplotlib artist) ----
        self._raster_im = None              # current imshow artist (or None)
        self._raster_image = None           # numpy image array (cached for re-add)
        self._raster_extent = None          # (left, right, bottom, top)
        self._raster_cmap = RASTER_CMAP
        self._raster_alpha = RASTER_ALPHA
        self._raster_visible = True         # user's toggle state
        self._has_raster_data = False       # True once display_raster() succeeds

        # ---- Scatter state ----
        self._scatter = None
        self._scatter_visible = True
        self._colorbar = None

        # ---- Blitting ----
        self._blit_background = None        # cached bitmap of axes background
        self._needs_full_redraw = True      # force full draw on next frame

        # Current frame data for click lookups
        self._frame_x: Optional[np.ndarray] = None
        self._frame_y: Optional[np.ndarray] = None
        self._frame_indices: Optional[np.ndarray] = None  # master row indices
        self._frame_ages: Optional[np.ndarray] = None     # current age per pixel
        self._row_lookup_fn = None  # callable(int) -> pd.Series

        # Click handling
        self._fig.canvas.mpl_connect("button_press_event", self._on_click)

        # Zoom / pan detection — invalidate blit cache when user changes viewport
        self._ignore_limits_change = False  # guard for our own set_xlim/set_ylim calls
        self._ax.callbacks.connect("xlim_changed", self._on_limits_changed)
        self._ax.callbacks.connect("ylim_changed", self._on_limits_changed)

        # Resize detection — invalidate blit cache when canvas size changes
        self._fig.canvas.mpl_connect("resize_event", self._on_resize)

        self._popup_window: Optional[tk.Toplevel] = None
        self._popup_tree = None

        # Saved popup layout (persists across popup opens)
        self._popup_geometry: Optional[str] = None       # e.g. "750x580+100+200"
        self._popup_attr_width: Optional[int] = None     # attribute column px
        self._popup_val_width: Optional[int] = None      # value column px

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def scatter_size(self):
        return self._scatter_size

    @scatter_size.setter
    def scatter_size(self, value):
        self._scatter_size = max(1, value)

    @property
    def popup_columns(self):
        return self._popup_columns

    @popup_columns.setter
    def popup_columns(self, cols):
        self._popup_columns = list(cols)

    @property
    def raster_extent(self):
        return self._raster_extent

    def set_row_lookup(self, fn):
        """Set a callable(pixel_index) -> pd.Series for popup details."""
        self._row_lookup_fn = fn

    def reset_view(self):
        """Reset viewport to the full raster/data extent."""
        if self._raster_extent is not None:
            self._set_extent_limits(self._raster_extent)
            self._needs_full_redraw = True
            self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Raster display — add / remove artist (not just set_visible)
    # ------------------------------------------------------------------

    def display_raster(self, image, extent, cmap=None, alpha=None):
        """Load and display a raster background image."""
        # Remove old artist if present
        if self._raster_im is not None:
            self._raster_im.remove()
            self._raster_im = None

        # Cache the data so we can re-add the artist on toggle
        self._raster_extent = extent
        self._raster_image = image
        self._raster_cmap = cmap or RASTER_CMAP
        self._raster_alpha = alpha if alpha is not None else RASTER_ALPHA
        self._has_raster_data = True
        self._raster_visible = True

        self._ax.set_facecolor("black")
        self._add_raster_artist()

        self._set_extent_limits(extent)
        self._needs_full_redraw = True
        self._canvas.draw_idle()

    def _add_raster_artist(self):
        """Create the imshow artist from cached data."""
        if self._raster_image is None or self._raster_extent is None:
            return

        image = self._raster_image
        extent = self._raster_extent

        if image.ndim == 2:
            self._raster_im = self._ax.imshow(
                image, extent=extent, cmap=self._raster_cmap,
                alpha=self._raster_alpha, aspect="equal", origin="upper",
            )
        else:
            self._raster_im = self._ax.imshow(
                image, extent=extent, alpha=self._raster_alpha,
                aspect="equal", origin="upper",
            )

    def _remove_raster_artist(self):
        """Remove the imshow artist entirely so matplotlib never touches it."""
        if self._raster_im is not None:
            self._raster_im.remove()
            self._raster_im = None

    def set_raster_visible(self, visible: bool):
        """
        Toggle the background raster on/off.

        When off, the imshow artist is fully *removed* from the axes —
        matplotlib will not process it at all during draw().
        When turned back on, the artist is recreated from cached data.
        This makes raster and scatter rendering truly independent.
        """
        self._raster_visible = visible

        if not self._has_raster_data:
            return

        if visible:
            if self._raster_im is None:
                self._add_raster_artist()
        else:
            self._remove_raster_artist()

        self._needs_full_redraw = True
        self._canvas.draw_idle()

    def set_black_background(self, extent):
        """
        Set a pure black background with the given extent.
        Used when no raster is loaded.
        """
        self._raster_extent = extent
        self._ax.set_facecolor("black")
        self._set_extent_limits(extent)
        self._needs_full_redraw = True
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Scatter display
    # ------------------------------------------------------------------

    def set_scatter_visible(self, visible: bool):
        """Toggle fire pixel scatter on/off."""
        self._scatter_visible = visible
        if self._scatter is not None:
            self._scatter.set_visible(visible)
            self._needs_full_redraw = True
            self._canvas.draw_idle()

    def update_scatter(self, x, y, ages, indices, n_levels=None, max_points=None):
        """
        Redraw fire-pixel scatter from pure numpy arrays.

        Uses blitting: the static background (raster + axes chrome) is
        cached as a bitmap.  Only the scatter artist is redrawn per frame.

        Parameters
        ----------
        x, y : np.ndarray — coordinates
        ages : np.ndarray — age in days per pixel
        indices : np.ndarray — master row indices (for click lookup)
        n_levels : int, optional — override N_COLOUR_LEVELS
        max_points : int, optional — if set, subsample to at most this
            many points (uniform stride).  Used during slider drag for
            responsive scrubbing.  Pass None for full resolution.
        """
        # Always store full data for click lookups
        self._frame_x = x
        self._frame_y = y
        self._frame_indices = indices
        self._frame_ages = ages

        # Remove old scatter
        if self._scatter is not None:
            self._scatter.remove()
            self._scatter = None

        if len(x) == 0:
            self._needs_full_redraw = True
            self._canvas.draw()
            self._blit_background = None
            return

        # ── Decimation: uniform stride to keep at most max_points ──
        draw_x, draw_y, draw_ages = x, y, ages
        if max_points is not None and len(x) > max_points:
            stride = max(1, len(x) // max_points)
            draw_x = x[::stride]
            draw_y = y[::stride]
            draw_ages = ages[::stride]

        # Map age → colour index (0 = newest, n-1 = oldest, clamped)
        n = n_levels or self._n_levels
        if n != self._n_levels:
            self._n_levels = n
            self._colour_table = _build_colour_table(
                n, COLOUR_NEWEST, COLOUR_OLDEST
            )
            # Rebuild colorbar to match new levels
            if self._colorbar is not None:
                self._cbar_ax.clear()
                self._colorbar = None
            self._needs_full_redraw = True
        colour_idx = np.clip(draw_ages, 0, n - 1).astype(int)
        colours = self._colour_table[colour_idx]

        self._scatter = self._ax.scatter(
            draw_x, draw_y, c=colours, s=self._scatter_size,
            edgecolors="none", zorder=5,
        )
        self._scatter.set_visible(self._scatter_visible)

        # Create or recreate colourbar
        if self._colorbar is None:
            self._create_colorbar()

        # ── Blitting path ──
        # Full draw only when background has changed (raster toggled,
        # first frame, resize, etc).  Otherwise restore cached background
        # and redraw only the scatter artist.
        if self._needs_full_redraw or self._blit_background is None:
            self._canvas.draw()
            self._blit_background = self._canvas.copy_from_bbox(self._ax.bbox)
            self._needs_full_redraw = False
        else:
            self._canvas.restore_region(self._blit_background)
            self._ax.draw_artist(self._scatter)
            self._canvas.blit(self._ax.bbox)

    def clear(self):
        if self._scatter is not None:
            self._scatter.remove()
            self._scatter = None
        self._frame_x = None
        self._frame_y = None
        self._frame_indices = None
        self._frame_ages = None
        self._needs_full_redraw = True
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Colourbar — created once
    # ------------------------------------------------------------------

    def _create_colorbar(self):
        self._cbar_ax.set_visible(True)

        # Build a colourmap from our table
        cmap = LinearSegmentedColormap.from_list(
            "fire_age",
            [COLOUR_NEWEST, COLOUR_OLDEST],
            N=self._n_levels,
        )
        norm = Normalize(vmin=0, vmax=self._n_levels)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        self._colorbar = self._fig.colorbar(
            sm, cax=self._cbar_ax, orientation="vertical",
        )
        # Label ticks
        self._cbar_ax.set_ylabel("Age (days)")
        tick_positions = np.linspace(0, self._n_levels, 6)
        self._colorbar.set_ticks(tick_positions)
        self._colorbar.set_ticklabels([str(int(t)) for t in tick_positions])

    # ------------------------------------------------------------------
    # Zoom / pan detection
    # ------------------------------------------------------------------

    def _on_limits_changed(self, _ax):
        """
        Called when axes xlim or ylim change (user zoom/pan via toolbar).
        Invalidates the blit cache so the next frame does a full redraw
        with the new viewport.
        """
        if not self._ignore_limits_change:
            self._needs_full_redraw = True

    def _set_extent_limits(self, extent):
        """Set axis limits from extent without triggering blit invalidation."""
        self._ignore_limits_change = True
        self._ax.set_xlim(extent[0], extent[1])
        self._ax.set_ylim(extent[2], extent[3])
        self._ignore_limits_change = False

    def _on_resize(self, _event):
        """Canvas resized (e.g. controls collapsed/expanded). Invalidate blit cache."""
        self._blit_background = None
        self._needs_full_redraw = True

    # ------------------------------------------------------------------
    # Click → popup
    # ------------------------------------------------------------------

    def _on_click(self, event):
        if event.inaxes != self._ax or event.button != 1:
            return
        if self._frame_x is None or len(self._frame_x) == 0:
            return

        dists = (self._frame_x - event.xdata) ** 2 + \
                (self._frame_y - event.ydata) ** 2
        idx = int(np.argmin(dists))

        # Threshold: within 1.5% of axis range
        xlim = self._ax.get_xlim()
        threshold = (0.015 * (xlim[1] - xlim[0])) ** 2
        if dists[idx] > threshold:
            return

        master_idx = int(self._frame_indices[idx])
        current_age = int(self._frame_ages[idx])
        if self._row_lookup_fn:
            row = self._row_lookup_fn(master_idx)
            if row is not None:
                self._show_popup(row, current_age)

    def _show_popup(self, row: pd.Series, current_age: int):
        # Save settings from previous popup before destroying
        if self._popup_window is not None:
            self._save_popup_settings()
            try:
                self._popup_window.destroy()
            except tk.TclError:
                pass

        self._popup_window = tk.Toplevel(self._parent)
        self._popup_window.title("Fire Pixel Details")

        # Restore saved geometry or use default
        if self._popup_geometry:
            self._popup_window.geometry(self._popup_geometry)
        else:
            self._popup_window.geometry("750x580")

        self._popup_window.attributes("-topmost", True)
        self._popup_window.resizable(True, True)
        self._popup_window.protocol("WM_DELETE_WINDOW", self._close_popup)

        self._popup_window.columnconfigure(0, weight=1)
        self._popup_window.rowconfigure(0, weight=1)

        frame = ttk.Frame(self._popup_window, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        # Style
        style = ttk.Style()
        style.configure("Popup.Treeview", rowheight=28, font=("Consolas", 11))
        style.configure("Popup.Treeview.Heading", font=("Consolas", 11, "bold"))

        # Collect all rows first
        def _fmt(val):
            if isinstance(val, float):
                return f"{val:.8f}"
            return str(val)

        rows_data = []
        for col in self._popup_columns:
            if col == "age_days":
                rows_data.append(("age_days", str(current_age)))
            elif col in row.index:
                rows_data.append((col, _fmt(row[col])))

        for col in row.index:
            if col not in self._popup_columns and col not in ("geometry", "age_days"):
                rows_data.append((col, _fmt(row[col])))

        # Column widths: use saved or auto-compute
        max_attr_len = max((len(r[0]) for r in rows_data), default=10)
        attr_width = self._popup_attr_width or max(max_attr_len * 11, 180)
        val_width = self._popup_val_width or 450

        self._popup_tree = ttk.Treeview(
            frame, columns=("attribute", "value"), show="headings",
            style="Popup.Treeview"
        )
        self._popup_tree["displaycolumns"] = ("attribute", "value")
        self._popup_tree.heading("attribute", text="Attribute", anchor="w")
        self._popup_tree.heading("value", text="Value", anchor="w")
        self._popup_tree.column("#0", width=0, stretch=False)
        self._popup_tree.column("attribute", width=attr_width, minwidth=120, anchor="w", stretch=False)
        self._popup_tree.column("value", width=val_width, minwidth=200, anchor="w", stretch=True)

        scrollbar_y = ttk.Scrollbar(frame, orient="vertical", command=self._popup_tree.yview)
        scrollbar_x = ttk.Scrollbar(frame, orient="horizontal", command=self._popup_tree.xview)
        self._popup_tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        self._popup_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar_y.grid(row=0, column=1, sticky="ns")
        scrollbar_x.grid(row=1, column=0, sticky="ew")

        for attr, val in rows_data:
            self._popup_tree.insert("", tk.END, values=(attr, val))

        btn_frame = ttk.Frame(self._popup_window)
        btn_frame.grid(row=1, column=0, pady=8)
        ttk.Button(btn_frame, text="Close",
                   command=self._close_popup).pack()

    def _save_popup_settings(self):
        """Capture current popup geometry and column widths."""
        try:
            if self._popup_window and self._popup_window.winfo_exists():
                self._popup_geometry = self._popup_window.geometry()
            if hasattr(self, "_popup_tree") and self._popup_tree:
                self._popup_attr_width = self._popup_tree.column("attribute", "width")
                self._popup_val_width = self._popup_tree.column("value", "width")
        except (tk.TclError, Exception):
            pass

    def _close_popup(self):
        """Save settings then destroy."""
        self._save_popup_settings()
        if self._popup_window:
            try:
                self._popup_window.destroy()
            except tk.TclError:
                pass
            self._popup_window = None