"""
FireDataManager: loads shapefiles, parses dates from filenames,
maintains numpy arrays for fast animation and a GeoDataFrame for lookups.

Design:
    - GeoDataFrame is used ONLY at load time and for popup detail lookups.
    - During animation, all data flows through pre-extracted numpy arrays.
    - No .loc[], .copy(), or pandas ops happen per frame.
"""

import os
import re
import glob
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import geopandas as gpd
import pandas as pd
import numpy as np

from config import FILENAME_DATETIME_PATTERN, FILENAME_DATETIME_FORMAT, MAX_WORKERS


class FrameData:
    """Lightweight container for one animation frame — pure numpy."""
    __slots__ = ("indices", "x", "y", "ages", "max_age", "n_pixels")

    def __init__(self, indices, x, y, ages, max_age):
        self.indices = indices      # int array — row indices into master arrays
        self.x = x                  # float array — plot x coords
        self.y = y                  # float array — plot y coords
        self.ages = ages            # int array — age in days
        self.max_age = max_age      # int
        self.n_pixels = len(indices)


class FireDataManager:
    """
    Manages fire pixel data from VIIRS VNP14IMG shapefiles.
    """

    def __init__(self, max_workers: int = MAX_WORKERS):
        self._file_dict: Dict[datetime, str] = {}
        self._master_gdf: Optional[gpd.GeoDataFrame] = None
        self._date_range: List[date] = []
        self._max_workers = max_workers

        # Pre-extracted numpy arrays (set once after load + clip)
        self._all_x: Optional[np.ndarray] = None
        self._all_y: Optional[np.ndarray] = None
        self._all_ordinals: Optional[np.ndarray] = None

        # Frame cache: {date: FrameData}
        self._frame_cache: Dict[date, FrameData] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def file_dict(self):
        return self._file_dict

    @property
    def master_gdf(self):
        return self._master_gdf

    @property
    def date_range(self):
        return self._date_range

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def scan_directory(self, directory: str) -> Dict[datetime, str]:
        """
        Scan *directory* recursively for ALL .shp files.
        Parses datetime from filename if possible; assigns a fallback
        datetime for files that don't match the naming convention.
        """
        shp_files = glob.glob(
            os.path.join(directory, "**", "*.shp"), recursive=True
        )
        parsed: Dict[datetime, str] = {}
        no_dt_files: List[str] = []

        for fpath in shp_files:
            dt = self._parse_datetime_from_stem(Path(fpath).stem)
            if dt is not None:
                # Handle duplicate datetimes by adding seconds offset
                while dt in parsed:
                    dt = dt + timedelta(seconds=1)
                parsed[dt] = fpath
            else:
                no_dt_files.append(fpath)

        # Assign fallback datetimes for files without a parseable date.
        # Use file modification time so they sort reasonably.
        for fpath in no_dt_files:
            try:
                mtime = os.path.getmtime(fpath)
                dt = datetime.fromtimestamp(mtime)
            except Exception:
                dt = datetime(1970, 1, 1)
            while dt in parsed:
                dt = dt + timedelta(seconds=1)
            parsed[dt] = fpath
            print(f"[INFO] No datetime in filename, using mtime: {fpath} → {dt}")

        self._file_dict = dict(sorted(parsed.items()))
        return self._file_dict

    def load_all(
        self, progress_cb: Optional[Callable[[int, int], None]] = None
    ) -> gpd.GeoDataFrame:
        """Parallel shapefile loading."""
        items = list(self._file_dict.items())
        total = len(items)
        frames: List = [None] * total

        def _read_one(args):
            idx, dt, fpath = args
            try:
                gdf = gpd.read_file(fpath)
                gdf["detection_datetime"] = dt
                gdf["detection_date"] = dt.date()
                gdf["source_file"] = os.path.basename(fpath)
                return idx, gdf
            except Exception as exc:
                print(f"[WARN] Could not read {fpath}: {exc}")
                return idx, None

        work = [(i, dt, fp) for i, (dt, fp) in enumerate(items)]
        loaded = 0

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futs = {pool.submit(_read_one, w): w for w in work}
            for fut in as_completed(futs):
                idx, gdf = fut.result()
                frames[idx] = gdf
                loaded += 1
                if progress_cb:
                    progress_cb(loaded, total)

        frames = [f for f in frames if f is not None]
        if not frames:
            self._master_gdf = gpd.GeoDataFrame()
            return self._master_gdf

        self._master_gdf = gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True)
        )
        self._master_gdf["pixel_id"] = np.arange(len(self._master_gdf))
        self._master_gdf["age_days"] = 0

        self._extract_numpy_arrays()
        self._build_full_date_range()
        return self._master_gdf

    def load_filtered(self, start_date: date, end_date: date) -> gpd.GeoDataFrame:
        filtered = {
            dt: fp for dt, fp in self._file_dict.items()
            if start_date <= dt.date() <= end_date
        }
        orig = self._file_dict
        self._file_dict = dict(sorted(filtered.items()))
        gdf = self.load_all()
        self._file_dict = orig
        return gdf

    # ------------------------------------------------------------------
    # Numpy extraction (called once after load, once after clip)
    # ------------------------------------------------------------------

    def _extract_numpy_arrays(self):
        """Pull coordinates and ordinals into pure numpy arrays."""
        gdf = self._master_gdf
        if gdf is None or gdf.empty:
            self._all_x = np.array([], dtype=np.float64)
            self._all_y = np.array([], dtype=np.float64)
            self._all_ordinals = np.array([], dtype=np.int64)
            return

        if "utm_x" in gdf.columns and "utm_y" in gdf.columns:
            self._all_x = gdf["utm_x"].values.astype(np.float64).copy()
            self._all_y = gdf["utm_y"].values.astype(np.float64).copy()
        else:
            self._all_x = gdf.geometry.x.values.astype(np.float64).copy()
            self._all_y = gdf.geometry.y.values.astype(np.float64).copy()

        self._all_ordinals = np.array(
            [d.toordinal() for d in gdf["detection_date"]],
            dtype=np.int64,
        )
        self._frame_cache.clear()

    # ------------------------------------------------------------------
    # Frame access — pure numpy, no pandas
    # ------------------------------------------------------------------

    def get_frame(self, current_date: date) -> FrameData:
        """
        Get the animation frame for a given date.
        Returns a FrameData with numpy arrays only — no pandas.
        """
        if current_date in self._frame_cache:
            return self._frame_cache[current_date]

        if self._all_ordinals is None or len(self._all_ordinals) == 0:
            fd = FrameData(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.int64),
                0,
            )
            self._frame_cache[current_date] = fd
            return fd

        cur_ord = current_date.toordinal()
        mask = self._all_ordinals <= cur_ord
        indices = np.where(mask)[0]
        ages = cur_ord - self._all_ordinals[indices]
        max_age = int(ages.max()) if len(ages) > 0 else 0

        fd = FrameData(
            indices,
            self._all_x[indices],
            self._all_y[indices],
            ages,
            max_age,
        )
        self._frame_cache[current_date] = fd
        return fd

    def get_row_data(self, pixel_index: int) -> Optional[pd.Series]:
        """Get full attribute row for a pixel (for popup). Uses GeoDataFrame."""
        if self._master_gdf is None or pixel_index >= len(self._master_gdf):
            return None
        return self._master_gdf.iloc[pixel_index]

    def precompute_frames(
        self,
        dates: Optional[List[date]] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ):
        """Pre-compute all frame data. Pure numpy, very fast."""
        if self._all_ordinals is None or len(self._all_ordinals) == 0:
            return

        dates = dates or self._date_range
        total = len(dates)
        ordinals = self._all_ordinals
        all_x = self._all_x
        all_y = self._all_y

        for i, d in enumerate(dates):
            cur_ord = d.toordinal()
            mask = ordinals <= cur_ord
            indices = np.where(mask)[0]
            ages = cur_ord - ordinals[indices]
            max_age = int(ages.max()) if len(ages) > 0 else 0

            self._frame_cache[d] = FrameData(
                indices, all_x[indices], all_y[indices], ages, max_age
            )

            if progress_cb and (i % 50 == 0 or i == total - 1):
                progress_cb(i + 1, total)

    def clear_cache(self):
        self._frame_cache.clear()

    # ------------------------------------------------------------------
    # Clipping
    # ------------------------------------------------------------------

    def clip_to_extent(self, left, right, bottom, top) -> int:
        if self._master_gdf is None or self._master_gdf.empty:
            return 0

        n_before = len(self._master_gdf)
        x, y = self._all_x, self._all_y
        mask = (x >= left) & (x <= right) & (y >= bottom) & (y <= top)

        self._master_gdf = self._master_gdf.loc[mask].reset_index(drop=True)
        self._master_gdf["pixel_id"] = np.arange(len(self._master_gdf))

        self._extract_numpy_arrays()
        self._build_full_date_range()
        return n_before - len(self._master_gdf)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_data_extent(self, padding_frac: float = 0.05):
        """
        Compute bounding box (left, right, bottom, top) from the fire pixel
        coordinates, with optional padding.  Returns None if no data.
        """
        if self._all_x is None or len(self._all_x) == 0:
            return None
        x_min, x_max = float(self._all_x.min()), float(self._all_x.max())
        y_min, y_max = float(self._all_y.min()), float(self._all_y.max())
        dx = (x_max - x_min) * padding_frac or 1.0
        dy = (y_max - y_min) * padding_frac or 1.0
        return (x_min - dx, x_max + dx, y_min - dy, y_max + dy)

    def get_date_range_bounds(self) -> Tuple[Optional[date], Optional[date]]:
        if self._master_gdf is None or self._master_gdf.empty:
            return None, None
        dates = self._master_gdf["detection_date"]
        return dates.min(), dates.max()

    def _parse_datetime_from_stem(self, stem: str) -> Optional[datetime]:
        match = re.search(FILENAME_DATETIME_PATTERN, stem)
        if match is None:
            return None
        try:
            return datetime.strptime(match.group(1), FILENAME_DATETIME_FORMAT)
        except ValueError:
            return None

    def _build_full_date_range(self):
        min_d, max_d = self.get_date_range_bounds()
        if min_d is None:
            self._date_range = []
            return
        self._date_range = []
        d = min_d
        while d <= max_d:
            self._date_range.append(d)
            d += timedelta(days=1)