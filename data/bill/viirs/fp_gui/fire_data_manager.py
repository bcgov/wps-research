"""
viirs/fp_gui/fire_data_manager.py

FireDataManager: loads shapefiles, parses dates from filenames,
maintains numpy arrays for fast animation and a GeoDataFrame for lookups.
"""

import os
import re
import glob
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

import geopandas as gpd
import pandas as pd
import numpy as np

from numba import njit

from config import FILENAME_DATETIME_PATTERN, FILENAME_DATETIME_FORMAT, MAX_WORKERS


def _read_one_file(args):
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


@njit(cache=True)
def _compute_frame_numba(all_ordinals, cur_ord):
    n = len(all_ordinals)
    buf_idx = np.empty(n, dtype=np.int64)
    buf_age = np.empty(n, dtype=np.int64)
    count = 0
    max_age = 0
    for i in range(n):
        if all_ordinals[i] <= cur_ord:
            buf_idx[count] = i
            age = cur_ord - all_ordinals[i]
            buf_age[count] = age
            if age > max_age:
                max_age = age
            count += 1
    return buf_idx[:count], buf_age[:count], max_age


class FrameData:
    __slots__ = ("indices", "x", "y", "ages", "max_age", "n_pixels")

    def __init__(self, indices, x, y, ages, max_age):
        self.indices = indices
        self.x = x
        self.y = y
        self.ages = ages
        self.max_age = max_age
        self.n_pixels = len(indices)


class FireDataManager:
    def __init__(self, max_workers: int = MAX_WORKERS):
        self._file_dict: Dict[datetime, str] = {}
        self._master_gdf: Optional[gpd.GeoDataFrame] = None
        self._date_range: List[date] = []
        self._max_workers = max_workers
        self._target_crs = None

        self._all_x: Optional[np.ndarray] = None
        self._all_y: Optional[np.ndarray] = None
        self._all_ordinals: Optional[np.ndarray] = None

        self._sort_order: Optional[np.ndarray] = None
        self._sorted_ords: Optional[np.ndarray] = None

        self._frame_cache: Dict[date, FrameData] = {}

    @property
    def file_dict(self):
        return self._file_dict

    @property
    def master_gdf(self):
        return self._master_gdf

    @property
    def date_range(self):
        return self._date_range

    def scan_directory(self, directory: str) -> Dict[datetime, str]:
        shp_files = glob.glob(
            os.path.join(directory, "**", "*.shp"), recursive=True
        )
        parsed: Dict[datetime, str] = {}
        no_dt_files: List[str] = []

        for fpath in shp_files:
            dt = self._parse_datetime_from_stem(Path(fpath).stem)
            if dt is not None:
                while dt in parsed:
                    dt = dt + timedelta(seconds=1)
                parsed[dt] = fpath
            else:
                no_dt_files.append(fpath)

        for fpath in no_dt_files:
            try:
                mtime = os.path.getmtime(fpath)
                dt = datetime.fromtimestamp(mtime)
            except Exception:
                dt = datetime(1970, 1, 1)
            while dt in parsed:
                dt = dt + timedelta(seconds=1)
            parsed[dt] = fpath
            print(f"[INFO] No datetime in filename, using mtime: {fpath} -> {dt}")

        self._file_dict = dict(sorted(parsed.items()))
        return self._file_dict

    def load_all(
        self,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        target_crs=None,
    ) -> gpd.GeoDataFrame:
        items = list(self._file_dict.items())
        total = len(items)
        frames: List = [None] * total

        work = [(i, dt, fp) for i, (dt, fp) in enumerate(items)]
        loaded = 0

        with ProcessPoolExecutor(max_workers=self._max_workers) as pool:
            futs = {pool.submit(_read_one_file, w): w for w in work}
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

        if target_crs is not None:
            self._target_crs = target_crs
        if self._target_crs is not None:
            common_crs = self._target_crs
            print(f"[INFO] Reprojecting all shapefiles to target CRS: {common_crs}")
        else:
            common_crs = frames[0].crs

        for i in range(len(frames)):
            if frames[i].crs is not None and frames[i].crs != common_crs:
                print(f"[INFO] Reprojecting {frames[i]['source_file'].iloc[0]} "
                      f"from {frames[i].crs} -> {common_crs}")
                frames[i] = frames[i].to_crs(common_crs)
                if "utm_x" in frames[i].columns and "utm_y" in frames[i].columns:
                    frames[i]["utm_x"] = frames[i].geometry.x
                    frames[i]["utm_y"] = frames[i].geometry.y

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

    def _extract_numpy_arrays(self):
        gdf = self._master_gdf
        if gdf is None or gdf.empty:
            self._all_x = np.array([], dtype=np.float64)
            self._all_y = np.array([], dtype=np.float64)
            self._all_ordinals = np.array([], dtype=np.int64)
            self._sort_order = None
            self._sorted_ords = None
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

        self._sort_order = np.argsort(self._all_ordinals)
        self._sorted_ords = self._all_ordinals[self._sort_order]
        self._frame_cache.clear()

    def get_frame(self, current_date: date) -> FrameData:
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
        indices, ages, max_age = _compute_frame_numba(self._all_ordinals, cur_ord)

        fd = FrameData(
            indices,
            self._all_x[indices],
            self._all_y[indices],
            ages,
            int(max_age),
        )
        self._frame_cache[current_date] = fd
        return fd

    def get_row_data(self, pixel_index: int) -> Optional[pd.Series]:
        if self._master_gdf is None or pixel_index >= len(self._master_gdf):
            return None
        return self._master_gdf.iloc[pixel_index]

    def precompute_frames(
        self,
        dates: Optional[List[date]] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ):
        if self._all_ordinals is None or len(self._all_ordinals) == 0:
            return

        dates = dates or self._date_range
        total = len(dates)
        if total == 0:
            return

        sort_order = self._sort_order
        sorted_ords = self._sorted_ords

        if sort_order is None or sorted_ords is None:
            sort_order = np.argsort(self._all_ordinals)
            sorted_ords = self._all_ordinals[sort_order]

        date_ordinals = np.array([d.toordinal() for d in dates], dtype=np.int64)
        cutoffs = np.searchsorted(sorted_ords, date_ordinals, side='right')

        all_x = self._all_x
        all_y = self._all_y
        all_ordinals = self._all_ordinals

        for i, d in enumerate(dates):
            n = cutoffs[i]
            if n == 0:
                self._frame_cache[d] = FrameData(
                    np.array([], dtype=np.int64),
                    np.array([], dtype=np.float64),
                    np.array([], dtype=np.float64),
                    np.array([], dtype=np.int64),
                    0,
                )
            else:
                indices = sort_order[:n]
                ages = date_ordinals[i] - all_ordinals[indices]
                max_age = int(ages.max())
                self._frame_cache[d] = FrameData(
                    indices, all_x[indices], all_y[indices], ages, max_age
                )

            if progress_cb and (i % 50 == 0 or i == total - 1):
                progress_cb(i + 1, total)

    def clear(self):
        """Reset all fire-pixel state (called when shapefiles are not loaded)."""
        self._file_dict = {}
        self._master_gdf = None
        self._date_range = []
        self._target_crs = None
        self._all_x = None
        self._all_y = None
        self._all_ordinals = None
        self._sort_order = None
        self._sorted_ords = None
        self._frame_cache.clear()

    def clear_cache(self):
        self._frame_cache.clear()

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

    def get_data_extent(self, padding_frac: float = 0.05):
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