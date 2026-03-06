"""
viirs/fp_gui/raster_loader.py

RasterLoader: reads an ENVI (or any GDAL-compatible) raster file
using the gdal-based Raster class, and provides the image
array + geographic extent for matplotlib display.

Performance:
    Large rasters (e.g. 4564×5093) are downsampled to at most
    MAX_RASTER_DISPLAY_DIM on their longest edge for the display copy.
    matplotlib only renders this smaller version, making pan/zoom fast.

NoData handling:
    Non-finite values are replaced with white (1.0) so that background
    areas match the GUI's white canvas — same behaviour as QGIS.
"""

import numpy as np
from typing import Tuple, Optional, List

from raster import Raster
from config import MAX_RASTER_DISPLAY_DIM


class RasterLoader:
    """
    Wraps the Raster class to expose:
        - image array  (H×W or H×W×3)  — downsampled for display
        - geographic extent [left, right, bottom, top]
        - CRS / projection string
    """

    def __init__(self):
        self._image: Optional[np.ndarray] = None          # display (downsampled)
        self._image_full: Optional[np.ndarray] = None      # original resolution
        self._extent: Optional[Tuple[float, float, float, float]] = None
        self._crs: Optional[str] = None
        self._raster: Optional[Raster] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def image(self) -> Optional[np.ndarray]:
        return self._image

    @property
    def extent(self) -> Optional[Tuple[float, float, float, float]]:
        """(left, right, bottom, top) for matplotlib imshow."""
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
        """
        Load the raster file via the Raster class.

        Parameters
        ----------
        filepath : str
            Path to the ENVI header (.hdr) or data file.
        bands : list[int], optional
            1-based band indices to load (e.g. [1,2,3] for RGB).
            If None, loads all bands.

        Returns
        -------
        np.ndarray  – the image array ready for imshow (downsampled).
        """
        self._raster = Raster(filepath)
        self._crs = self._raster._proj

        # --- Compute extent from GeoTransform ---
        gt = self._raster._transform
        x_size = self._raster._xSize
        y_size = self._raster._ySize

        left   = gt[0]
        right  = gt[0] + gt[1] * x_size
        top    = gt[3]
        bottom = gt[3] + gt[5] * y_size
        self._extent = (left, right, bottom, top)

        # --- Read bands ---
        if bands is not None:
            data = self._raster.read_bands(band_lst=bands)
        else:
            data = self._raster.read_bands()          # 'all'

        # data shape: (H, W, n_bands) from np.dstack
        n_bands = data.shape[2]
        if n_bands == 1:
            img = data[:, :, 0]
        elif n_bands >= 3:
            img = data[:, :, :3]
        else:
            img = data[:, :, 0]

        # --- Normalise to 0–1 ---
        img = img.astype(np.float32)
        finite_mask = np.isfinite(img)

        if finite_mask.any():
            vmin, vmax = np.nanpercentile(img, [1, 99])
            if vmax > vmin:
                img = np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)
            else:
                img = np.zeros_like(img)

        # --- NoData → white (1.0) so it matches the white background ---
        if img.ndim == 2:
            img = np.where(finite_mask, img, 1.0)
        else:
            # For each channel independently
            img = np.where(finite_mask, img, 1.0)

        self._image_full = img

        # --- Downsample for display ---
        self._image = self._downsample(img)

        return self._image

    # ------------------------------------------------------------------
    # Downsampling
    # ------------------------------------------------------------------

    @staticmethod
    def _downsample(image: np.ndarray, max_dim: int = MAX_RASTER_DISPLAY_DIM) -> np.ndarray:
        """
        Stride-based downsampling to keep the longest edge ≤ max_dim.
        Fast (pure indexing, no interpolation) and perfectly adequate
        for on-screen display where pixels are smaller than physical dots.
        """
        h, w = image.shape[:2]
        longest = max(h, w)
        if longest <= max_dim:
            return image                  # already small enough

        scale = longest / max_dim         # e.g. 5093 / 2000 = 2.55
        step = max(1, int(round(scale)))  # stride 3 for that example

        if image.ndim == 2:
            return image[::step, ::step].copy()
        return image[::step, ::step, :].copy()