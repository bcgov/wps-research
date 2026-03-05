"""
viirs/fp_gui/raster_loader.py

RasterLoader: reads an ENVI (or any GDAL-compatible) raster file
using the gdal-based Raster class, and provides the image
array + geographic extent for matplotlib display.
"""

import numpy as np
from typing import Tuple, Optional, List

from raster import Raster


class RasterLoader:
    """
    Wraps the Raster class to expose:
        - image array  (H×W or H×W×3)
        - geographic extent [left, right, bottom, top]
        - CRS / projection string
    """

    def __init__(self):
        self._image: Optional[np.ndarray] = None
        self._extent: Optional[Tuple[float, float, float, float]] = None
        self._crs: Optional[str] = None
        self._raster: Optional[Raster] = None

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
        """Access the underlying Raster object directly if needed."""
        return self._raster

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
        np.ndarray  – the image array ready for imshow.
        """
        self._raster = Raster(filepath)
        self._crs = self._raster._proj

        # --- Compute extent from GeoTransform ---
        # GeoTransform: (origin_x, pixel_width, 0, origin_y, 0, pixel_height)
        # pixel_height is typically negative
        gt = self._raster._transform
        x_size = self._raster._xSize
        y_size = self._raster._ySize

        left = gt[0]
        right = gt[0] + gt[1] * x_size
        top = gt[3]
        bottom = gt[3] + gt[5] * y_size

        self._extent = (left, right, bottom, top)

        # --- Read bands ---
        if bands is not None:
            data = self._raster.read_bands(band_lst=bands)
        else:
            data = self._raster.read_bands()  # 'all'

        # data is (H, W, n_bands) from np.dstack
        n_bands = data.shape[2]

        if n_bands == 1:
            self._image = data[:, :, 0]
        elif n_bands >= 3:
            self._image = data[:, :, :3]
        else:
            # 2 bands — just use first
            self._image = data[:, :, 0]

        # Replace nodata with NaN
        self._image = np.where(
            np.isfinite(self._image), self._image, np.nan
        )

        # Normalise to 0-1 for display
        vmin, vmax = np.nanpercentile(self._image, [1, 99])
        if vmax > vmin:
            self._image = np.clip(
                (self._image - vmin) / (vmax - vmin), 0, 1
            )
        else:
            self._image = np.zeros_like(self._image)

        return self._image