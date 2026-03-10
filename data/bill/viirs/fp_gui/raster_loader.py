"""
viirs/fp_gui/raster_loader.py

RasterLoader: reads an ENVI (or any GDAL-compatible) raster file
using the gdal-based Raster class, and provides the image
array + geographic extent for matplotlib display.

Performance:
    Large rasters are downsampled to at most MAX_RASTER_DISPLAY_DIM
    on their longest edge for the display copy.
NoData handling:
    Non-finite values are replaced with white (1.0) so that background
    areas match the GUI's white canvas.
Pixel resolution:
    After loading, .pixel_size_m exposes the average ground-sampling
    distance in map-unit metres.  .compute_scatter_size() returns the
    rounded ratio VNP14IMG_PIXEL_SIZE_M / pixel_size for use as
    DEFAULT_SCATTER_SIZE.
"""

import numpy as np
from typing import Tuple, Optional, List
from raster import Raster
import config as cfg          # read at call time, not import time


class RasterLoader:
    """
    Wraps the Raster class to expose:
        - image array  (HxW or HxWx3)  -- downsampled for display
        - geographic extent [left, right, bottom, top]
        - CRS / projection string
        - pixel_size_m  -- spatial resolution in map units
    """

    def __init__(self):
        self._image: Optional[np.ndarray] = None
        self._image_full: Optional[np.ndarray] = None
        self._extent: Optional[Tuple[float, float, float, float]] = None
        self._crs: Optional[str] = None
        self._raster: Optional[Raster] = None
        self._pixel_size_m: Optional[float] = None

    @property
    def image(self) -> Optional[np.ndarray]:
        return self._image

    @property
    def extent(self) -> Optional[Tuple[float, float, float, float]]:
        return self._extent

    @property
    def crs(self) -> Optional[str]:
        return self._crs

    @property
    def raster(self) -> Optional[Raster]:
        return self._raster

    @property
    def pixel_size_m(self) -> Optional[float]:
        """Average ground-sampling distance in the raster's map units."""
        return self._pixel_size_m

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(
        self, filepath: str, bands: Optional[List[int]] = None
    ) -> np.ndarray:
        self._raster = Raster(filepath)
        self._crs = self._raster._proj

        gt = self._raster._transform
        x_size = self._raster._xSize
        y_size = self._raster._ySize

        left   = gt[0]
        right  = gt[0] + gt[1] * x_size
        top    = gt[3]
        bottom = gt[3] + gt[5] * y_size
        self._extent = (left, right, bottom, top)

        # Compute spatial resolution from the GeoTransform
        pixel_w = abs(gt[1])
        pixel_h = abs(gt[5])
        self._pixel_size_m = (pixel_w + pixel_h) / 2.0

        if bands is not None:
            data = self._raster.read_bands(band_lst=bands)
        else:
            data = self._raster.read_bands()

        n_bands = data.shape[2]
        if n_bands == 1:
            img = data[:, :, 0]
        elif n_bands >= 3:
            img = data[:, :, :3]
        else:
            img = data[:, :, 0]

        img = img.astype(np.float32)
        finite_mask = np.isfinite(img)

        if finite_mask.any():
            vmin, vmax = np.nanpercentile(img, [1, 99])
            if vmax > vmin:
                img = np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)
            else:
                img = np.zeros_like(img)

        if img.ndim == 2:
            img = np.where(finite_mask, img, 1.0)
        else:
            img = np.where(finite_mask, img, 1.0)

        self._image_full = img
        self._image = self._downsample(img, max_dim=cfg.MAX_RASTER_DISPLAY_DIM)
        return self._image

    # ------------------------------------------------------------------
    # Scatter size from resolution
    # ------------------------------------------------------------------

    def compute_scatter_size(self) -> int:
        """
        Return the scatter marker base size so that a VNP14IMG fire pixel
        (375 m) is drawn proportional to one raster pixel.

        Returns max(1, round(VNP14IMG_PIXEL_SIZE_M / raster_pixel_size)).
        Falls back to DEFAULT_SCATTER_SIZE if no raster is loaded.
        """
        if self._pixel_size_m is None or self._pixel_size_m <= 0:
            return cfg.DEFAULT_SCATTER_SIZE
        ratio = cfg.VNP14IMG_PIXEL_SIZE_M / self._pixel_size_m
        return max(1, round(ratio))

    # ------------------------------------------------------------------
    # Downsampling
    # ------------------------------------------------------------------

    @staticmethod
    def _downsample(image: np.ndarray, max_dim: int = 99999) -> np.ndarray:
        """
        Stride-based downsampling to keep the longest edge <= max_dim.
        """
        h, w = image.shape[:2]
        longest = max(h, w)
        if longest <= max_dim:
            return image
        scale = longest / max_dim
        step = max(1, int(round(scale)))
        if image.ndim == 2:
            return image[::step, ::step].copy()
        return image[::step, ::step, :].copy()