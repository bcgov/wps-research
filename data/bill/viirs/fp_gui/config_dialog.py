"""
viirs/fp_gui/config_dialog.py

ConfigDialog: modal popup to view / edit all config.py values.
Changes are staged until "Apply" is clicked — Cancel discards them.
"""

import tkinter as tk
from tkinter import ttk

import config as cfg


# All editable config keys with (label, type, min, max) metadata
_CONFIG_FIELDS = [
    ("DEFAULT_SCATTER_SIZE",         "Fire Pixel Scatter Size",     int,   1,     2000),
    ("DEFAULT_ANIMATION_INTERVAL_MS","Animation Interval (ms)",     int,   50,    5000),
    ("N_COLOUR_LEVELS",              "Colour Levels",               int,   10,    500),
    ("MAX_RASTER_DISPLAY_DIM",       "Max Raster Display Dim (px)", int,   500,   99999),
    ("PAN_PREVIEW_MAX_DIM",          "Pan Preview Max Dim (px)",    int,   200,   10000),
]


class ConfigDialog:
    """
    A Toplevel dialog that displays all config.py values and lets the
    user edit them.  Apply writes back to the config module; Cancel
    discards.  The dialog is destroyed on Apply/Cancel/X.
    """

    def __init__(self, parent: tk.Tk):
        self._parent = parent
        self._entries = {}
        self._applied = False

        self._win = tk.Toplevel(parent)
        self._win.title("Configuration")
        self._win.transient(parent)
        self._win.grab_set()

        # Size to fit all fields — no scrollbar needed
        n_fields = len(_CONFIG_FIELDS)
        row_h = 36
        h = n_fields * row_h + 80  # fields + button bar + padding
        w = 520

        sx = parent.winfo_screenwidth()
        sy = parent.winfo_screenheight()
        x = (sx - w) // 2
        y = (sy - h) // 2
        self._win.geometry(f"{w}x{h}+{x}+{y}")
        self._win.resizable(False, False)

        self._win.columnconfigure(0, weight=1)
        self._win.rowconfigure(0, weight=1)

        # Fields grid — no scrollbar
        inner = ttk.Frame(self._win, padding=12)
        inner.grid(row=0, column=0, sticky="nsew")
        inner.columnconfigure(1, weight=1)

        for row_idx, (key, label, typ, lo, hi) in enumerate(_CONFIG_FIELDS):
            ttk.Label(inner, text=label, anchor="w").grid(
                row=row_idx, column=0, sticky="w", padx=(4, 16), pady=5)

            current_val = getattr(cfg, key)
            sv = tk.StringVar(value=str(current_val))
            self._entries[key] = (sv, typ, lo, hi)

            entry = ttk.Entry(inner, textvariable=sv, width=28)
            entry.grid(row=row_idx, column=1, sticky="ew", padx=4, pady=5)

        # Button bar
        btn_frame = ttk.Frame(self._win)
        btn_frame.grid(row=1, column=0, pady=(4, 12))

        apply_btn = tk.Button(
            btn_frame, text="  \u2714  Apply  ", bg="#4CAF50", fg="white",
            font=("TkDefaultFont", 10, "bold"), activebackground="#388E3C",
            command=self._on_apply,
        )
        apply_btn.pack(side=tk.LEFT, padx=12)

        cancel_btn = tk.Button(
            btn_frame, text="  \u2716  Cancel  ", bg="#F44336", fg="white",
            font=("TkDefaultFont", 10, "bold"), activebackground="#C62828",
            command=self._on_cancel,
        )
        cancel_btn.pack(side=tk.LEFT, padx=12)

        self._win.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self._win.wait_window()

    @property
    def applied(self):
        return self._applied

    def _on_apply(self):
        """Validate entries and write back to config module."""
        for key, (sv, typ, lo, hi) in self._entries.items():
            raw = sv.get().strip()
            try:
                if typ == int:
                    val = int(raw)
                elif typ == float:
                    val = float(raw)
                else:
                    val = raw

                # Clamp to [lo, hi]
                if isinstance(val, (int, float)):
                    if lo is not None:
                        val = max(lo, val)
                    if hi is not None:
                        val = min(hi, val)

                setattr(cfg, key, val)
            except Exception as exc:
                print(f"[WARN] Could not apply {key}={raw}: {exc}")

        self._applied = True
        self._win.destroy()

    def _on_cancel(self):
        self._applied = False
        self._win.destroy()