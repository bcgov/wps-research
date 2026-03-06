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
    """

    def __init__(self):
        self._image: Optional[np.ndarray] = None
        self._image_full: Optional[np.ndarray] = None
        self._extent: Optional[Tuple[float, float, float, float]] = None
        self._crs: Optional[str] = None
        self._raster: Optional[Raster] = None

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

        # Read MAX_RASTER_DISPLAY_DIM from config at call time
        # so that Config dialog changes take effect before loading
        self._image = self._downsample(img, max_dim=cfg.MAX_RASTER_DISPLAY_DIM)

        return self._image

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