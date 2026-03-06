"""
viirs/fp_gui/fire_map_canvas.py

FireMapCanvas: matplotlib figure embedded in a tkinter frame.
Renders using pure numpy arrays -- no pandas during animation.

Navigation (QGIS-style):
    ZOOM_IN  -- left-drag green rectangle -> zoom to that area
    ZOOM_OUT -- left-drag red rectangle   -> zoom out
    PAN      -- left-drag to pan the viewport

    Scroll-wheel zoom works in every mode.
    Left-click  shows pixel detail popup (any mode, only if not a drag).
    Right-click shows pixel detail popup (any mode).

Scatter sizing:
    Marker sizes scale with the zoom level so they behave like fixed-size
    objects in data coordinates (just like raster pixels do).
"""

import time
import tkinter as tk
from tkinter import ttk
from typing import Optional, List, Callable, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")

for _key in list(matplotlib.rcParams):
    if _key.startswith("keymap."):
        try:
            matplotlib.rcParams[_key] = []
        except (ValueError, TypeError):
            pass

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle as MplRectangle
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

# Navigation modes
NAV_ZOOM_IN = "zoom_in"
NAV_ZOOM_OUT = "zoom_out"
NAV_PAN = "pan"

_CURSORS = {
    NAV_ZOOM_IN: "cross",
    NAV_ZOOM_OUT: "cross",
    NAV_PAN: "fleur",
}

_CLICK_THRESHOLD_PX = 5


def _build_colour_table(n_levels, newest_rgba, oldest_rgba):
    table = np.zeros((n_levels, 4), dtype=np.float32)
    for c in range(4):
        table[:, c] = np.linspace(newest_rgba[c], oldest_rgba[c], n_levels)
    return table


class FireMapCanvas:
    """
    Embeds a matplotlib figure in a tkinter parent.
    Accepts pure numpy arrays for rendering -- zero pandas per frame.
    """

    def __init__(self, parent: tk.Widget, figsize=(11, 7), dpi=100):
        self._parent = parent
        self._scatter_size = DEFAULT_SCATTER_SIZE
        self._popup_columns: List[str] = list(DEFAULT_POPUP_COLUMNS)

        self._n_levels = N_COLOUR_LEVELS
        self._colour_table = _build_colour_table(
            self._n_levels, COLOUR_NEWEST, COLOUR_OLDEST)

        # -- Figure --
        self._fig = plt.figure(figsize=figsize, dpi=dpi)
        self._fig.patch.set_facecolor("white")

        self._ax = self._fig.add_axes([0.0, 0.0, 0.93, 1.0])
        self._ax.set_facecolor("white")
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        for spine in self._ax.spines.values():
            spine.set_visible(False)

        self._cbar_ax = self._fig.add_axes([0.935, 0.05, 0.014, 0.90])
        self._cbar_ax.set_visible(False)

        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # -- Raster state --
        self._raster_im = None
        self._raster_image = None
        self._raster_extent = None
        self._raster_cmap = RASTER_CMAP
        self._raster_alpha = RASTER_ALPHA
        self._raster_visible = True
        self._has_raster_data = False

        # -- Scatter state --
        self._scatter = None
        self._scatter_visible = True
        self._colorbar = None

        # -- Zoom-proportional scatter sizing --
        self._home_view_width: Optional[float] = None

        # -- Blitting --
        self._blit_background = None
        self._needs_full_redraw = True

        # -- Frame data --
        self._frame_x: Optional[np.ndarray] = None
        self._frame_y: Optional[np.ndarray] = None
        self._frame_indices: Optional[np.ndarray] = None
        self._frame_ages: Optional[np.ndarray] = None
        self._row_lookup_fn = None

        # -- Navigation --
        self._nav_mode: str = NAV_PAN

        self._rect_start: Optional[Tuple[float, float]] = None
        self._rect_patch: Optional[MplRectangle] = None
        self._rect_bg = None

        self._pan_start_px: Optional[Tuple[float, float]] = None
        self._pan_xlim0 = None
        self._pan_ylim0 = None
        self._last_pan_draw: float = 0.0
        self._PAN_MIN_DT: float = 0.020

        # Click detection
        self._press_xy_px: Optional[Tuple[float, float]] = None

        # View history
        self._view_stack: List[Tuple[Tuple, Tuple]] = []
        self._view_index: int = -1
        self._home_extent: Optional[Tuple[float, float, float, float]] = None

        self._on_viewport_changed_cb: Optional[Callable[[], None]] = None

        # -- Mouse events --
        cid = self._fig.canvas.mpl_connect
        cid("button_press_event",   self._on_mouse_press)
        cid("button_release_event", self._on_mouse_release)
        cid("motion_notify_event",  self._on_mouse_move)
        cid("scroll_event",         self._on_scroll)
        cid("resize_event",         self._on_resize)

        self._ignore_limits_change = False
        self._ax.callbacks.connect("xlim_changed", self._on_limits_changed)
        self._ax.callbacks.connect("ylim_changed", self._on_limits_changed)

        # Popup
        self._popup_window: Optional[tk.Toplevel] = None
        self._popup_tree = None
        self._popup_geometry: Optional[str] = None
        self._popup_attr_width: Optional[int] = None
        self._popup_val_width: Optional[int] = None

    # ==================================================================
    # Properties
    # ==================================================================

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

    @property
    def nav_mode(self) -> str:
        return self._nav_mode

    def set_row_lookup(self, fn):
        self._row_lookup_fn = fn

    def set_on_viewport_changed(self, cb: Callable[[], None]):
        self._on_viewport_changed_cb = cb

    # ==================================================================
    # Zoom-proportional scatter size
    # ==================================================================

    def _effective_scatter_s(self) -> float:
        """
        Compute the matplotlib scatter `s` so marker diameter scales
        proportionally with zoom -- exactly like raster pixels do.

        At home zoom, s = scatter_size (base value).
        Zoom in 2x -> diameter doubles -> s quadruples.
        """
        if self._home_view_width is None or self._home_view_width <= 0:
            return float(self._scatter_size)

        xlim = self._ax.get_xlim()
        cur_width = abs(xlim[1] - xlim[0])
        if cur_width <= 0:
            return float(self._scatter_size)

        zoom_ratio = self._home_view_width / cur_width
        return float(self._scatter_size) * (zoom_ratio ** 2)

    # ==================================================================
    # Navigation mode
    # ==================================================================

    def set_nav_mode(self, mode: str):
        self._nav_mode = mode
        cursor = _CURSORS.get(mode, "arrow")
        try:
            self._canvas.get_tk_widget().config(cursor=cursor)
        except tk.TclError:
            pass

    # ==================================================================
    # View history / reset
    # ==================================================================

    def _push_view(self):
        xlim = tuple(self._ax.get_xlim())
        ylim = tuple(self._ax.get_ylim())
        if (self._view_stack
                and 0 <= self._view_index < len(self._view_stack)
                and self._view_stack[self._view_index] == (xlim, ylim)):
            return
        self._view_stack = self._view_stack[: self._view_index + 1]
        self._view_stack.append((xlim, ylim))
        self._view_index = len(self._view_stack) - 1

    def view_back(self):
        if self._view_index > 0:
            self._view_index -= 1
            xlim, ylim = self._view_stack[self._view_index]
            self._apply_view(xlim, ylim)

    def view_forward(self):
        if self._view_index < len(self._view_stack) - 1:
            self._view_index += 1
            xlim, ylim = self._view_stack[self._view_index]
            self._apply_view(xlim, ylim)

    def view_home(self):
        self.reset_view()

    def reset_view(self):
        ext = self._home_extent or self._raster_extent
        if ext is not None:
            self._push_view()
            self._set_extent_limits(ext)
            self._push_view()
            self._needs_full_redraw = True
            self._canvas.draw_idle()
            self._notify_viewport_changed()

    def _apply_view(self, xlim, ylim):
        self._set_extent_limits((xlim[0], xlim[1], ylim[0], ylim[1]))
        self._needs_full_redraw = True
        self._canvas.draw_idle()
        self._notify_viewport_changed()

    # ==================================================================
    # Viewport pixel count
    # ==================================================================

    def count_visible_pixels(self) -> int:
        if self._frame_x is None or len(self._frame_x) == 0:
            return 0
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        x_lo, x_hi = min(xlim), max(xlim)
        y_lo, y_hi = min(ylim), max(ylim)
        mask = (
            (self._frame_x >= x_lo) & (self._frame_x <= x_hi)
            & (self._frame_y >= y_lo) & (self._frame_y <= y_hi)
        )
        return int(mask.sum())

    def _notify_viewport_changed(self):
        if self._on_viewport_changed_cb:
            try:
                self._on_viewport_changed_cb()
            except Exception:
                pass

    # ==================================================================
    # Raster display
    # ==================================================================

    def display_raster(self, image, extent, cmap=None, alpha=None):
        if self._raster_im is not None:
            self._raster_im.remove()
            self._raster_im = None

        self._raster_extent = extent
        self._raster_image = image
        self._raster_cmap = cmap or RASTER_CMAP
        self._raster_alpha = alpha if alpha is not None else RASTER_ALPHA
        self._has_raster_data = True
        self._raster_visible = True

        self._ax.set_facecolor("white")
        self._add_raster_artist()
        self._set_extent_limits(extent)
        self._home_extent = extent
        self._home_view_width = abs(extent[1] - extent[0])

        self._needs_full_redraw = True
        self._canvas.draw()
        self._push_view()
        self._notify_viewport_changed()

    def _add_raster_artist(self):
        if self._raster_image is None or self._raster_extent is None:
            return
        img, ext = self._raster_image, self._raster_extent

        if img.ndim == 2:
            cmap = plt.cm.get_cmap(self._raster_cmap).copy()
            cmap.set_bad(color="white", alpha=1.0)
            self._raster_im = self._ax.imshow(
                np.ma.masked_invalid(img),
                extent=ext, cmap=cmap,
                alpha=self._raster_alpha, aspect="equal",
                origin="upper", interpolation="nearest",
            )
        else:
            self._raster_im = self._ax.imshow(
                img, extent=ext,
                alpha=self._raster_alpha, aspect="equal",
                origin="upper", interpolation="nearest",
            )

    def _remove_raster_artist(self):
        if self._raster_im is not None:
            self._raster_im.remove()
            self._raster_im = None

    def set_raster_visible(self, visible: bool):
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
        self._raster_extent = extent
        self._ax.set_facecolor("white")
        self._set_extent_limits(extent)
        self._home_extent = extent
        self._home_view_width = abs(extent[1] - extent[0])
        self._needs_full_redraw = True
        self._canvas.draw()
        self._push_view()

    # ==================================================================
    # Scatter display
    # ==================================================================

    def set_scatter_visible(self, visible: bool):
        self._scatter_visible = visible
        if self._scatter is not None:
            self._scatter.set_visible(visible)
            self._needs_full_redraw = True
            self._canvas.draw_idle()

    def update_scatter(self, x, y, ages, indices, n_levels=None,
                       max_points=None):
        """Redraw fire-pixel scatter from pure numpy arrays."""
        self._frame_x = x
        self._frame_y = y
        self._frame_indices = indices
        self._frame_ages = ages

        if self._scatter is not None:
            self._scatter.remove()
            self._scatter = None

        if len(x) == 0:
            self._needs_full_redraw = True
            self._canvas.draw()
            self._blit_background = None
            return

        draw_x, draw_y, draw_ages = x, y, ages
        if max_points is not None and len(x) > max_points:
            stride = max(1, len(x) // max_points)
            draw_x = x[::stride]
            draw_y = y[::stride]
            draw_ages = ages[::stride]

        n = n_levels or self._n_levels
        if n != self._n_levels:
            self._n_levels = n
            self._colour_table = _build_colour_table(
                n, COLOUR_NEWEST, COLOUR_OLDEST)
            if self._colorbar is not None:
                self._cbar_ax.clear()
                self._colorbar = None
            self._needs_full_redraw = True

        colour_idx = np.clip(draw_ages, 0, n - 1).astype(int)
        colours = self._colour_table[colour_idx]

        s = self._effective_scatter_s()

        self._scatter = self._ax.scatter(
            draw_x, draw_y, c=colours, s=s,
            edgecolors="none", zorder=5,
        )
        self._scatter.set_visible(self._scatter_visible)

        if self._colorbar is None:
            self._create_colorbar()

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

    # ==================================================================
    # Colourbar
    # ==================================================================

    def _create_colorbar(self):
        self._cbar_ax.set_visible(True)
        cmap = LinearSegmentedColormap.from_list(
            "fire_age", [COLOUR_NEWEST, COLOUR_OLDEST], N=self._n_levels)
        norm = Normalize(vmin=0, vmax=self._n_levels)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        self._colorbar = self._fig.colorbar(
            sm, cax=self._cbar_ax, orientation="vertical")
        self._cbar_ax.set_ylabel("Age (days)")
        ticks = np.linspace(0, self._n_levels, 6)
        self._colorbar.set_ticks(ticks)
        self._colorbar.set_ticklabels([str(int(t)) for t in ticks])

    # ==================================================================
    # Mouse routing
    # ==================================================================

    def _on_mouse_press(self, event):
        if event.inaxes != self._ax:
            return

        self._press_xy_px = (event.x, event.y)

        if event.button == 3:
            self._try_popup(event)
            return

        if event.button != 1:
            return

        if self._nav_mode in (NAV_ZOOM_IN, NAV_ZOOM_OUT):
            self._rect_press(event)
        elif self._nav_mode == NAV_PAN:
            self._pan_press(event)

    def _on_mouse_release(self, event):
        was_click = self._was_click(event)

        if self._nav_mode in (NAV_ZOOM_IN, NAV_ZOOM_OUT):
            self._rect_release(event)
            if was_click and event.button == 1:
                self._try_popup(event)
        elif self._nav_mode == NAV_PAN:
            self._pan_release(event)
            if was_click and event.button == 1:
                self._try_popup(event)

        self._press_xy_px = None

    def _was_click(self, event) -> bool:
        if self._press_xy_px is None:
            return False
        if event.x is None or event.y is None:
            return False
        dx = abs(event.x - self._press_xy_px[0])
        dy = abs(event.y - self._press_xy_px[1])
        return dx < _CLICK_THRESHOLD_PX and dy < _CLICK_THRESHOLD_PX

    def _on_mouse_move(self, event):
        if self._nav_mode in (NAV_ZOOM_IN, NAV_ZOOM_OUT):
            self._rect_move(event)
        elif self._nav_mode == NAV_PAN:
            self._pan_move(event)

    def _on_scroll(self, event):
        if event.inaxes != self._ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        factor = 0.75 if event.button == "up" else 1.333
        self._zoom_at(event.xdata, event.ydata, factor)

    # ==================================================================
    # Rectangle zoom
    # ==================================================================

    def _rect_press(self, event):
        if event.xdata is None or event.ydata is None:
            return
        self._rect_start = (event.xdata, event.ydata)

        if self._nav_mode == NAV_ZOOM_IN:
            fc = (0.0, 0.7, 0.0, 0.15)
            ec = (0.0, 0.7, 0.0, 0.80)
        else:
            fc = (0.8, 0.0, 0.0, 0.15)
            ec = (0.8, 0.0, 0.0, 0.80)

        self._rect_patch = MplRectangle(
            (event.xdata, event.ydata), 0, 0,
            fill=True, facecolor=fc, edgecolor=ec,
            linewidth=1.5, linestyle="--", zorder=100)
        self._ax.add_patch(self._rect_patch)
        self._canvas.draw()
        self._rect_bg = self._canvas.copy_from_bbox(self._ax.bbox)

    def _rect_move(self, event):
        if self._rect_start is None or self._rect_patch is None:
            return
        if event.inaxes != self._ax or event.xdata is None:
            return
        x0, y0 = self._rect_start
        x1, y1 = event.xdata, event.ydata
        self._rect_patch.set_xy((min(x0, x1), min(y0, y1)))
        self._rect_patch.set_width(abs(x1 - x0))
        self._rect_patch.set_height(abs(y1 - y0))

        if self._rect_bg is not None:
            self._canvas.restore_region(self._rect_bg)
            self._ax.draw_artist(self._rect_patch)
            self._canvas.blit(self._ax.bbox)

    def _rect_release(self, event):
        if self._rect_start is None:
            return

        if self._rect_patch is not None:
            self._rect_patch.remove()
            self._rect_patch = None
        self._rect_bg = None

        x0, y0 = self._rect_start
        self._rect_start = None

        if event.inaxes != self._ax or event.xdata is None:
            self._needs_full_redraw = True
            self._canvas.draw_idle()
            return

        x1, y1 = event.xdata, event.ydata
        rw = abs(x1 - x0)
        rh = abs(y1 - y0)

        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        vw = xlim[1] - xlim[0]
        vh = ylim[1] - ylim[0]

        if rw < vw * 0.01 or rh < vh * 0.01:
            self._needs_full_redraw = True
            self._canvas.draw_idle()
            return

        self._push_view()

        if self._nav_mode == NAV_ZOOM_IN:
            new_ext = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
            self._set_extent_limits(new_ext)
        else:
            cx = (x0 + x1) * 0.5
            cy = (y0 + y1) * 0.5
            sx = vw / rw
            sy = vh / rh
            s = max(sx, sy)
            new_hw = vw * s * 0.5
            new_hh = vh * s * 0.5
            new_ext = (cx - new_hw, cx + new_hw, cy - new_hh, cy + new_hh)
            self._set_extent_limits(new_ext)

        self._push_view()
        self._needs_full_redraw = True
        self._canvas.draw_idle()
        self._notify_viewport_changed()

    # ==================================================================
    # Pan
    # ==================================================================

    def _pan_press(self, event):
        if event.xdata is None:
            return
        self._pan_start_px = (event.x, event.y)
        self._pan_xlim0 = self._ax.get_xlim()
        self._pan_ylim0 = self._ax.get_ylim()

    def _pan_move(self, event):
        if self._pan_start_px is None:
            return
        if event.x is None or event.y is None:
            return

        dx_px = event.x - self._pan_start_px[0]
        dy_px = event.y - self._pan_start_px[1]

        bbox = self._ax.get_window_extent()
        if bbox.width == 0 or bbox.height == 0:
            return

        xlim = self._pan_xlim0
        ylim = self._pan_ylim0
        dx = -(dx_px / bbox.width)  * (xlim[1] - xlim[0])
        dy = -(dy_px / bbox.height) * (ylim[1] - ylim[0])

        self._ignore_limits_change = True
        self._ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
        self._ax.set_ylim(ylim[0] + dy, ylim[1] + dy)
        self._ignore_limits_change = False

        now = time.monotonic()
        if now - self._last_pan_draw >= self._PAN_MIN_DT:
            self._needs_full_redraw = True
            self._canvas.draw()
            self._last_pan_draw = now

    def _pan_release(self, event):
        if self._pan_start_px is None:
            return
        self._pan_start_px = None
        self._needs_full_redraw = True
        self._canvas.draw()
        self._push_view()
        self._notify_viewport_changed()

    # ==================================================================
    # Scroll-wheel zoom
    # ==================================================================

    def _zoom_at(self, cx, cy, factor):
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        hw = (xlim[1] - xlim[0]) * 0.5 * factor
        hh = (ylim[1] - ylim[0]) * 0.5 * factor

        self._push_view()
        self._set_extent_limits((cx - hw, cx + hw, cy - hh, cy + hh))
        self._push_view()
        self._needs_full_redraw = True
        self._canvas.draw()
        self._notify_viewport_changed()

    # ==================================================================
    # Pixel popup (click)
    # ==================================================================

    def _try_popup(self, event):
        if self._frame_x is None or len(self._frame_x) == 0:
            return
        if event.xdata is None:
            return

        dists = ((self._frame_x - event.xdata) ** 2
                 + (self._frame_y - event.ydata) ** 2)
        idx = int(np.argmin(dists))

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

    # ==================================================================
    # Axis-limit helpers
    # ==================================================================

    def _on_limits_changed(self, _ax):
        if not self._ignore_limits_change:
            self._needs_full_redraw = True

    def _set_extent_limits(self, extent):
        self._ignore_limits_change = True
        self._ax.set_xlim(extent[0], extent[1])
        self._ax.set_ylim(extent[2], extent[3])
        self._ignore_limits_change = False

    def _on_resize(self, _event):
        self._blit_background = None
        self._needs_full_redraw = True

    # ==================================================================
    # Popup window
    # ==================================================================

    def _show_popup(self, row: pd.Series, current_age: int):
        if self._popup_window is not None:
            self._save_popup_settings()
            try:
                self._popup_window.destroy()
            except tk.TclError:
                pass

        self._popup_window = tk.Toplevel(self._parent)
        self._popup_window.title("Fire Pixel Details")
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

        style = ttk.Style()
        style.configure("Popup.Treeview", rowheight=28,
                        font=("Consolas", 11))
        style.configure("Popup.Treeview.Heading",
                        font=("Consolas", 11, "bold"))

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
            if col not in self._popup_columns and col not in (
                    "geometry", "age_days"):
                rows_data.append((col, _fmt(row[col])))

        max_attr_len = max((len(r[0]) for r in rows_data), default=10)
        attr_w = self._popup_attr_width or max(max_attr_len * 11, 180)
        val_w = self._popup_val_width or 450

        self._popup_tree = ttk.Treeview(
            frame, columns=("attribute", "value"), show="headings",
            style="Popup.Treeview")
        self._popup_tree["displaycolumns"] = ("attribute", "value")
        self._popup_tree.heading("attribute", text="Attribute", anchor="w")
        self._popup_tree.heading("value", text="Value", anchor="w")
        self._popup_tree.column("#0", width=0, stretch=False)
        self._popup_tree.column("attribute", width=attr_w, minwidth=120,
                                anchor="w", stretch=False)
        self._popup_tree.column("value", width=val_w, minwidth=200,
                                anchor="w", stretch=True)

        sy = ttk.Scrollbar(frame, orient="vertical",
                           command=self._popup_tree.yview)
        sx = ttk.Scrollbar(frame, orient="horizontal",
                           command=self._popup_tree.xview)
        self._popup_tree.configure(yscrollcommand=sy.set,
                                   xscrollcommand=sx.set)
        self._popup_tree.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")

        for attr, val in rows_data:
            self._popup_tree.insert("", tk.END, values=(attr, val))

        btn_frame = ttk.Frame(self._popup_window)
        btn_frame.grid(row=1, column=0, pady=8)
        ttk.Button(btn_frame, text="Close",
                   command=self._close_popup).pack()

    def _save_popup_settings(self):
        try:
            if self._popup_window and self._popup_window.winfo_exists():
                self._popup_geometry = self._popup_window.geometry()
            if self._popup_tree:
                self._popup_attr_width = self._popup_tree.column(
                    "attribute", "width")
                self._popup_val_width = self._popup_tree.column(
                    "value", "width")
        except (tk.TclError, Exception):
            pass

    def _close_popup(self):
        self._save_popup_settings()
        if self._popup_window:
            try:
                self._popup_window.destroy()
            except tk.TclError:
                pass
            self._popup_window = None