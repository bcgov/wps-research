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
    ("DEFAULT_SCATTER_SIZE",         "Scatter Size (base)",         int,   1,     200),
    ("DEFAULT_ANIMATION_INTERVAL_MS","Animation Interval (ms)",     int,   50,    5000),
    ("N_COLOUR_LEVELS",              "Colour Levels",               int,   10,    500),
    ("COLOUR_NEWEST",                "Colour Newest (R,G,B,A)",     str,   None,  None),
    ("COLOUR_OLDEST",                "Colour Oldest (R,G,B,A)",     str,   None,  None),
    ("RASTER_ALPHA",                 "Raster Alpha (0-1)",          float, 0.0,   1.0),
    ("RASTER_CMAP",                  "Raster Colourmap",            str,   None,  None),
    ("MAX_RASTER_DISPLAY_DIM",       "Max Raster Display Dim (px)", int,   100,   99999)
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

        # Centre on screen
        w, h = 560, 308
        sx = parent.winfo_screenwidth()
        sy = parent.winfo_screenheight()
        x = (sx - w) // 2
        y = (sy - h) // 2
        self._win.geometry(f"{w}x{h}+{x}+{y}")
        self._win.resizable(True, True)

        self._win.columnconfigure(0, weight=1)
        self._win.rowconfigure(0, weight=1)

        # Scrollable area
        outer = ttk.Frame(self._win)
        outer.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)

        canvas = tk.Canvas(outer, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)

        vsb.grid(row=0, column=1, sticky="ns")
        canvas.grid(row=0, column=0, sticky="nsew")

        inner = ttk.Frame(canvas)
        inner.columnconfigure(1, weight=1)
        canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_configure(_e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        inner.bind("<Configure>", _on_configure)

        # Mouse-wheel scroll
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_wheel(_e):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-3, "units"))
            canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(3, "units"))

        def _unbind_wheel(_e):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        canvas.bind("<Enter>", _bind_wheel)
        canvas.bind("<Leave>", _unbind_wheel)

        # Populate fields
        for row_idx, (key, label, typ, lo, hi) in enumerate(_CONFIG_FIELDS):
            ttk.Label(inner, text=label, anchor="w").grid(
                row=row_idx, column=0, sticky="w", padx=(4, 12), pady=4)

            current_val = getattr(cfg, key)
            sv = tk.StringVar(value=str(current_val))
            self._entries[key] = (sv, typ, lo, hi)

            entry = ttk.Entry(inner, textvariable=sv, width=36)
            entry.grid(row=row_idx, column=1, sticky="ew", padx=4, pady=4)

        # Button bar
        btn_frame = ttk.Frame(self._win)
        btn_frame.grid(row=1, column=0, pady=(0, 10))

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
                    if lo is not None:
                        val = max(lo, val)
                    if hi is not None:
                        val = min(hi, val)
                elif typ == float:
                    val = float(raw)
                    if lo is not None:
                        val = max(lo, val)
                    if hi is not None:
                        val = min(hi, val)
                elif typ == str:
                    if raw.startswith("(") and raw.endswith(")"):
                        val = tuple(float(x.strip()) for x in raw[1:-1].split(","))
                    else:
                        val = raw
                else:
                    val = raw

                setattr(cfg, key, val)
            except Exception as exc:
                print(f"[WARN] Could not apply {key}={raw}: {exc}")

        self._applied = True
        self._win.destroy()

    def _on_cancel(self):
        self._applied = False
        self._win.destroy()