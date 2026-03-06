"""
viirs/fp_gui/download_dialog.py

DownloadDialog: modal popup that manages the full pipeline -
  1. Load reference raster -> extract CRS + bbox -> convert to EPSG:4326
  2. Set date range and save directory
  3. Download VNP14IMG data from LAADS DAAC (background thread)
  4. Run shapify to convert .nc -> .shp
  5. Report completion
"""

import os
import sys
import threading
import subprocess
import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_raster_info(bin_path: str):
    """
    Read CRS (as WKT), EPSG code, and native bbox from a GDAL-readable
    raster file.  Returns (wkt, epsg_int_or_None, x_min, x_max, y_min, y_max).
    """
    from osgeo import gdal, osr
    gdal.UseExceptions()

    ds = gdal.Open(bin_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Could not open: {bin_path}")

    wkt = ds.GetProjection()
    if not wkt:
        raise RuntimeError("Raster has no projection metadata.")

    gt = ds.GetGeoTransform()
    nx, ny = ds.RasterXSize, ds.RasterYSize
    x_min = gt[0]
    x_max = gt[0] + nx * gt[1]
    y_max = gt[3]
    y_min = gt[3] + ny * gt[5]
    ds = None

    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    srs.AutoIdentifyEPSG()
    epsg_str = srs.GetAuthorityCode(None)
    epsg = int(epsg_str) if epsg_str else None

    return wkt, epsg, x_min, x_max, y_min, y_max


def _bbox_to_4326(wkt, x_min, x_max, y_min, y_max):
    """
    Convert the four corners of a bbox from the raster's CRS to EPSG:4326.
    Returns (west, south, east, north) in degrees.
    """
    from osgeo import osr
    src = osr.SpatialReference()
    src.ImportFromWkt(wkt)
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(4326)
    ct = osr.CoordinateTransformation(src, dst)

    corners = [
        (x_min, y_min), (x_min, y_max),
        (x_max, y_min), (x_max, y_max),
    ]
    lons, lats = [], []
    for x, y in corners:
        lat, lon, _ = ct.TransformPoint(x, y)
        lons.append(lon)
        lats.append(lat)

    return min(lons), min(lats), max(lons), max(lats)


def _ymd_to_datetime(ymd_str: str):
    return datetime.datetime.strptime(ymd_str.strip(), "%Y-%m-%d")


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class DownloadDialog:
    """
    Modal download-pipeline dialog.
    """

    def __init__(self, parent: tk.Tk):
        self._parent = parent
        self._cancel_event = threading.Event()
        self._download_thread: Optional[threading.Thread] = None
        self._nc_count = 0

        self._ref_wkt: Optional[str] = None
        self._ref_epsg: Optional[int] = None

        self._win = tk.Toplevel(parent)
        self._win.title("Download VIIRS VNP14IMG Data")
        self._win.transient(parent)
        self._win.grab_set()

        w, h = 720, 540
        sx = parent.winfo_screenwidth()
        sy = parent.winfo_screenheight()
        self._win.geometry(f"{w}x{h}+{(sx-w)//2}+{(sy-h)//2}")
        self._win.resizable(True, True)

        self._win.columnconfigure(0, weight=1)
        self._win.rowconfigure(5, weight=1)

        self._build_ui()
        self._win.protocol("WM_DELETE_WINDOW", self._on_close)
        self._win.wait_window()

    # ==================================================================
    # UI
    # ==================================================================

    def _build_ui(self):
        pad = dict(padx=8, pady=4)

        # -- Row 0: Reference file --
        r0 = ttk.LabelFrame(self._win, text="1. Reference Raster (.bin)", padding=6)
        r0.grid(row=0, column=0, sticky="ew", **pad)
        r0.columnconfigure(1, weight=1)

        self._ref_var = tk.StringVar()
        ttk.Label(r0, text="File:").grid(row=0, column=0, sticky="w")
        ttk.Entry(r0, textvariable=self._ref_var, width=48).grid(
            row=0, column=1, sticky="ew", padx=4)
        ttk.Button(r0, text="Browse\u2026", command=self._browse_ref).grid(
            row=0, column=2, padx=4)
        ttk.Button(r0, text="Load Reference", command=self._load_ref).grid(
            row=0, column=3, padx=4)

        # Info table
        info_frame = ttk.Frame(r0)
        info_frame.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(6, 0))

        self._epsg_var = tk.StringVar(value="\u2014")
        ttk.Label(info_frame, text="EPSG:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(info_frame, textvariable=self._epsg_var, width=8,
                  relief="sunken").pack(side=tk.LEFT, padx=(0, 12))

        bbox_labels = ["Min East:", "Max East:", "Min North:", "Max North:"]
        self._bbox_vars = []
        for lbl in bbox_labels:
            ttk.Label(info_frame, text=lbl).pack(side=tk.LEFT, padx=(4, 2))
            sv = tk.StringVar(value="\u2014")
            self._bbox_vars.append(sv)
            ttk.Entry(info_frame, textvariable=sv, width=14).pack(
                side=tk.LEFT, padx=(0, 4))

        # -- Row 1: LAADS Token --
        r1 = ttk.LabelFrame(self._win, text="2. LAADS DAAC Token", padding=6)
        r1.grid(row=1, column=0, sticky="ew", **pad)
        r1.columnconfigure(1, weight=1)

        self._token_var = tk.StringVar()
        try:
            with open("/data/.tokens/laads", "r") as fh:
                self._token_var.set(fh.read().strip())
        except Exception:
            pass

        ttk.Label(r1, text="Token:").grid(row=0, column=0, sticky="w")
        ttk.Entry(r1, textvariable=self._token_var, width=60, show="*").grid(
            row=0, column=1, sticky="ew", padx=4)

        # -- Row 2: Date range --
        r2 = ttk.LabelFrame(self._win, text="3. Date Range", padding=6)
        r2.grid(row=2, column=0, sticky="ew", **pad)

        self._start_var = tk.StringVar(value="2025-08-25")
        self._end_var = tk.StringVar(value="2025-09-20")

        ttk.Label(r2, text="Start (YYYY-MM-DD):").pack(side=tk.LEFT)
        ttk.Entry(r2, textvariable=self._start_var, width=14).pack(
            side=tk.LEFT, padx=4)
        ttk.Label(r2, text="End (YYYY-MM-DD):").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Entry(r2, textvariable=self._end_var, width=14).pack(
            side=tk.LEFT, padx=4)

        # -- Row 3: Save directory --
        r3 = ttk.LabelFrame(self._win, text="4. Save Directory", padding=6)
        r3.grid(row=3, column=0, sticky="ew", **pad)
        r3.columnconfigure(1, weight=1)

        self._save_dir_var = tk.StringVar()
        ttk.Label(r3, text="Dir:").grid(row=0, column=0, sticky="w")
        ttk.Entry(r3, textvariable=self._save_dir_var, width=48).grid(
            row=0, column=1, sticky="ew", padx=4)
        ttk.Button(r3, text="Browse\u2026", command=self._browse_save_dir).grid(
            row=0, column=2, padx=4)

        # -- Row 4: Action buttons --
        r4 = ttk.Frame(self._win)
        r4.grid(row=4, column=0, sticky="ew", **pad)

        self._download_btn = tk.Button(
            r4, text="  \u25b6  Download  ", bg="#4CAF50", fg="white",
            font=("TkDefaultFont", 10, "bold"), activebackground="#388E3C",
            command=self._start_download,
        )
        self._download_btn.pack(side=tk.LEFT, padx=8)

        self._cancel_btn = tk.Button(
            r4, text="  \u2716  Cancel  ", bg="#F44336", fg="white",
            font=("TkDefaultFont", 10, "bold"), activebackground="#C62828",
            command=self._cancel_download, state=tk.DISABLED,
        )
        self._cancel_btn.pack(side=tk.LEFT, padx=8)

        # -- Row 5: Status --
        r5 = ttk.LabelFrame(self._win, text="Status", padding=6)
        r5.grid(row=5, column=0, sticky="nsew", **pad)
        r5.columnconfigure(0, weight=1)
        r5.rowconfigure(1, weight=1)

        self._status_var = tk.StringVar(
            value="Ready. Load a reference file to begin.")
        ttk.Label(r5, textvariable=self._status_var, wraplength=660).grid(
            row=0, column=0, sticky="w")

        self._nc_count_var = tk.StringVar(value="Downloaded .nc files: 0")
        ttk.Label(r5, textvariable=self._nc_count_var).grid(
            row=1, column=0, sticky="nw")

    # ==================================================================
    # Browsing
    # ==================================================================

    def _browse_ref(self):
        f = filedialog.askopenfilename(
            parent=self._win,
            title="Select reference raster (.bin)",
            filetypes=[("ENVI binary", "*.bin"), ("All files", "*.*")],
        )
        if f:
            self._ref_var.set(f)

    def _browse_save_dir(self):
        d = filedialog.askdirectory(parent=self._win, title="Save directory")
        if d:
            self._save_dir_var.set(d)

    # ==================================================================
    # Load reference
    # ==================================================================

    def _load_ref(self):
        path = self._ref_var.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Select a valid .bin file.",
                                 parent=self._win)
            return

        try:
            wkt, epsg, x_min, x_max, y_min, y_max = _read_raster_info(path)
        except Exception as exc:
            messagebox.showerror("Error", f"Could not read raster:\n{exc}",
                                 parent=self._win)
            return

        self._ref_wkt = wkt
        self._ref_epsg = epsg
        self._epsg_var.set(str(epsg) if epsg else "Unknown")
        self._bbox_vars[0].set(f"{x_min:.4f}")
        self._bbox_vars[1].set(f"{x_max:.4f}")
        self._bbox_vars[2].set(f"{y_min:.4f}")
        self._bbox_vars[3].set(f"{y_max:.4f}")
        self._status_var.set(
            f"Reference loaded: EPSG:{epsg}  "
            f"({x_max - x_min:.0f} \u00d7 {y_max - y_min:.0f} map units)"
        )

    # ==================================================================
    # Download
    # ==================================================================

    def _start_download(self):
        ref_path = self._ref_var.get().strip()
        if not ref_path or not os.path.exists(ref_path):
            messagebox.showerror("Error", "Load a reference file first.",
                                 parent=self._win)
            return
        if self._ref_wkt is None:
            messagebox.showerror("Error", "Click 'Load Reference' first.",
                                 parent=self._win)
            return

        token = self._token_var.get().strip()
        if not token:
            messagebox.showerror("Error", "LAADS DAAC token is required.",
                                 parent=self._win)
            return

        try:
            start_dt = _ymd_to_datetime(self._start_var.get())
            end_dt = _ymd_to_datetime(self._end_var.get())
        except ValueError:
            messagebox.showerror("Error", "Dates must be YYYY-MM-DD.",
                                 parent=self._win)
            return
        if start_dt > end_dt:
            messagebox.showerror("Error", "Start must be before End.",
                                 parent=self._win)
            return

        save_dir = self._save_dir_var.get().strip()
        if not save_dir:
            messagebox.showerror("Error", "Set a save directory.",
                                 parent=self._win)
            return

        try:
            x_min = float(self._bbox_vars[0].get())
            x_max = float(self._bbox_vars[1].get())
            y_min = float(self._bbox_vars[2].get())
            y_max = float(self._bbox_vars[3].get())
        except ValueError:
            messagebox.showerror("Error", "Bounding box values must be numbers.",
                                 parent=self._win)
            return

        try:
            west, south, east, north = _bbox_to_4326(
                self._ref_wkt, x_min, x_max, y_min, y_max)
        except Exception as exc:
            messagebox.showerror(
                "Error", f"Could not convert bbox to lat/lon:\n{exc}",
                parent=self._win)
            return

        self._status_var.set(
            f"EPSG:4326 bbox \u2192 W:{west:.4f}  S:{south:.4f}  "
            f"E:{east:.4f}  N:{north:.4f}\nStarting download\u2026"
        )

        self._download_btn.configure(state=tk.DISABLED)
        self._cancel_btn.configure(state=tk.NORMAL)
        self._cancel_event.clear()
        self._nc_count = 0

        self._download_thread = threading.Thread(
            target=self._download_worker,
            args=(token, start_dt, end_dt, save_dir, ref_path,
                  west, south, east, north),
            daemon=True,
        )
        self._download_thread.start()

    def _cancel_download(self):
        self._cancel_event.set()
        self._status_var.set("Cancelling\u2026")
        self._cancel_btn.configure(state=tk.DISABLED)

    # ==================================================================
    # Background worker
    # ==================================================================

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

        # Try importing the sync function
        sync_fn = None
        try:
            from viirs.utils.laads_data_download_v2 import sync
            sync_fn = sync
        except ImportError:
            try:
                # Maybe we're running from a directory where it's accessible
                sys.path.insert(0, os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
                from laads_data_download_v2 import sync
                sync_fn = sync
            except ImportError:
                pass

        if sync_fn is None:
            self._update_status(
                "ERROR: Could not import laads_data_download_v2.sync.\n"
                "Make sure viirs.utils is on the Python path.")
            self._finish_reset()
            return

        # Download loop
        for i, download_day in enumerate(days):
            if self._cancel_event.is_set():
                self._update_status("Download cancelled by user.")
                self._finish_reset()
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

            msg = (f"Downloading day {i+1}/{total_days}: "
                   f"{download_day.strftime('%Y-%m-%d')} \u2026")
            self._update_status(msg)
            print(f"\n[DOWNLOAD] {msg}")
            print(f"  URL: {download_url}")
            print(f"  Dir: {download_path}")

            try:
                sync_fn(download_url, download_path, token)
            except Exception as exc:
                print(f"[WARN] Download error for {download_day}: {exc}")

            new_count = self._count_nc_files(save_dir)
            self._nc_count = new_count
            self._update_nc_count(new_count)

        if self._cancel_event.is_set():
            self._update_status("Download cancelled.")
            self._finish_reset()
            return

        total_nc = self._count_nc_files(save_dir)
        self._update_status(
            f"Download complete: {total_nc} .nc files. Running shapify\u2026")
        print(f"\n[INFO] Download finished. {total_nc} .nc files total.")

        self._run_shapify(save_dir, ref_path)

    def _run_shapify(self, save_dir, ref_path):
        if self._cancel_event.is_set():
            self._finish_reset()
            return

        self._update_status("Running shapify (converting .nc \u2192 .shp)\u2026")
        print("\n[INFO] Starting shapify\u2026")

        cmd = [
            sys.executable, "-m", "viirs.utils.shapify",
            save_dir,
            "-r", ref_path,
            "-w", "32",
        ]
        print(f"[INFO] Command: {' '.join(cmd)}")

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    print(f"  [shapify] {line}")
                if self._cancel_event.is_set():
                    proc.terminate()
                    self._update_status("Shapify cancelled.")
                    self._finish_reset()
                    return

            proc.wait()
            if proc.returncode == 0:
                self._update_status("Shapify complete!")
                print("[INFO] Shapify finished successfully.")
                self._show_done(save_dir)
            else:
                self._update_status(
                    f"Shapify exited with code {proc.returncode}.")
                print(f"[WARN] Shapify exit code: {proc.returncode}")
        except FileNotFoundError:
            self._update_status(
                "ERROR: Could not run shapify. "
                "Check viirs.utils is installed.")
            print("[ERROR] shapify command not found.")
        except Exception as exc:
            self._update_status(f"Shapify error: {exc}")
            print(f"[ERROR] shapify: {exc}")

        self._finish_reset()

    # ==================================================================
    # Thread-safe UI updates
    # ==================================================================

    def _update_status(self, msg):
        try:
            self._win.after(0, lambda: self._status_var.set(msg))
        except Exception:
            pass

    def _update_nc_count(self, n):
        try:
            self._win.after(0, lambda: self._nc_count_var.set(
                f"Downloaded .nc files: {n}"))
        except Exception:
            pass

    def _finish_reset(self):
        try:
            self._win.after(0, self._reset_buttons)
        except Exception:
            pass

    def _reset_buttons(self):
        self._download_btn.configure(state=tk.NORMAL)
        self._cancel_btn.configure(state=tk.DISABLED)

    def _show_done(self, save_dir):
        def _popup():
            messagebox.showinfo(
                "Download & Shapify Complete  \u2714",
                f"All data processed successfully.\n\n"
                f"Save directory:\n{save_dir}",
                parent=self._win,
            )
        try:
            self._win.after(0, _popup)
        except Exception:
            pass

    # ==================================================================
    # Helpers
    # ==================================================================

    @staticmethod
    def _count_nc_files(base_dir):
        count = 0
        for root, dirs, files in os.walk(base_dir):
            for f in files:
                if f.lower().endswith(".nc"):
                    count += 1
        return count

    def _on_close(self):
        if self._download_thread and self._download_thread.is_alive():
            if not messagebox.askyesno(
                "Confirm", "Download is running. Cancel and close?",
                parent=self._win,
            ):
                return
            self._cancel_event.set()
        self._win.destroy()