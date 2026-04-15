"""
py/fire_mapping/fire_mapping_cli.py
====================================
Non-GUI command-line version of fire_mapping.py.

Mirrors the computation inside fire_mapping.GUI but requires no interactive
input.  All parameters are set via command-line arguments.

Key differences from the GUI version
--------------------------------------
* No matplotlib pop-up windows.
* VIIRS accumulated raster (.bin) can be used as the burn hint
  (equivalent to 'polygon file' mode in the GUI).
  If no hint is provided, a mask is generated automatically using the
  dominant-band method (same fallback as the GUI).
* All T-SNE / RF / HDBSCAN hyperparameters are exposed as CLI flags.
* Outputs are written to the same directory as the input raster.
* Generates a 3-panel comparison figure (our mapping | VIIRS | perimeter).
* No QGIS launch.

Usage
-----
    python py/fire_mapping/fire_mapping_cli.py  RASTER.bin  [VIIRS_HINT.bin]  [options]

Example
-------
    python py/fire_mapping/fire_mapping_cli.py  C11659_crop.bin  C11659_viirs.bin  \\
        --fire_numbe C11659                                                         \\
        --start_date 2025-08-20 --end_date 2025-10-14                              \\
        --perimeter  C11659_perimeter.bin
"""

# ---------------------------------------------------------------------------
# Path setup — script lives inside py/fire_mapping/; add that dir to sys.path
# so bare imports (raster, misc, sampling …) resolve correctly.
# ---------------------------------------------------------------------------
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ---------------------------------------------------------------------------
# Project imports — identical to fire_mapping.py
# ---------------------------------------------------------------------------
from raster import Raster

from misc import (
    writeENVI,
    htrim_3d,
    extract_border,
    draw_border,
)

from sampling import regular_sampling

from dominant_band import dominant_band

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import argparse
import subprocess
import time

import numpy as np

# ---------------------------------------------------------------------------
# Matplotlib — headless (no display required)
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===========================================================================
# Helpers
# ===========================================================================

def _status_box(title: str, lines: list = None, char: str = '=', width: int = 64):
    """Print a box-style status block to stdout."""
    all_lines = [title] + (lines or [])
    w = max(width, *(len(l) + 4 for l in all_lines))
    bar = char * w
    print(f'\n{bar}')
    print(f'  {title}')
    if lines:
        print(char * w)
        for l in lines:
            print(f'  {l}')
    print(f'{bar}\n')


# ===========================================================================
# Core class
# ===========================================================================

class FireMappingCLI:
    """
    Non-GUI burn-mapping pipeline.

    The constructor accepts all tuneable parameters so callers (and
    run_fire_mapping.py) never need to touch the argparser.
    """

    def __init__(
        self,
        *,
        image_filename:     str,
        polygon_filename:   str   = None,   # optional — generate mask if absent

        # Metadata for plot titles (set by run_fire_mapping.py)
        fire_numbe:         str   = None,
        start_date:         str   = None,
        end_date:           str   = None,

        # Optional traditional perimeter for the 3rd comparison panel
        perimeter_filename: str   = None,

        # Sampling
        sample_size:        int   = 10_000,
        random_state:       int   = 123,
        embed_band_list:    list  = None,   # None → all bands (set after load_image)

        # Random Forest
        rf_n_estimators:    int   = 100,
        rf_max_depth:       int   = 15,
        rf_max_features:    str   = 'sqrt',
        rf_random_state:    int   = 42,

        # HDBSCAN
        controlled_ratio:      float = 0.5,
        hdbscan_min_samples:   int   = 20,

        # T-SNE (all params exposed)
        tsne_perplexity:     float = 60.0,
        tsne_learning_rate:  float = 200.0,
        tsne_max_iter:       int   = 2000,
        tsne_init:           str   = 'pca',
        tsne_n_components:   int   = 2,
        tsne_random_state:   int   = 42,

        # Figure
        contour_width:       int   = 1,

        # Intermediate state caching (serial mapping optimisation)
        save_state_path:     str   = None,
        load_state_path:     str   = None,
    ):
        self.image_filename     = image_filename
        self.polygon_filename   = polygon_filename
        self.perimeter_filename = perimeter_filename

        self.fire_numbe = fire_numbe or os.path.splitext(
            os.path.basename(image_filename))[0]
        self.start_date = start_date or 'unknown'
        self.end_date   = end_date   or 'unknown'

        # Save outputs next to the input raster
        self.save_dir = os.path.dirname(os.path.abspath(image_filename))

        # Sampling
        self.sample_size      = sample_size
        self.random_state     = random_state
        self.border_thickness = 5

        # embed_band_list is None until load_image() sets n_bands
        self._embed_band_list_arg = embed_band_list
        self.embed_band_list      = None   # resolved in load_image()
        self.img_band_list        = None   # resolved in load_image()

        # Random Forest
        self.rf_params = {
            'n_estimators': rf_n_estimators,
            'max_depth':    rf_max_depth,
            'max_features': rf_max_features,
            'random_state': rf_random_state,
        }

        # HDBSCAN
        self.controlled_ratio = controlled_ratio
        self.hdbscan_params   = {
            'min_cluster_size': None,   # computed from guessed_burn_p
            'min_samples':      hdbscan_min_samples,
            'metric':           'euclidean',
        }

        # T-SNE
        self.tsne_params = {
            'n_components':  tsne_n_components,
            'perplexity':    tsne_perplexity,
            'learning_rate': tsne_learning_rate,
            'max_iter':      tsne_max_iter,
            'init':          tsne_init,
            'random_state':  tsne_random_state,
            'verbose':       1,
        }

        # Figure
        self.contour_width   = contour_width

        # Intermediate state caching
        self.save_state_path = save_state_path
        self.load_state_path = load_state_path

    # -----------------------------------------------------------------------
    # Band selection  (mirrors GUI.find_default_rgb_bands)
    # -----------------------------------------------------------------------

    def find_default_rgb_bands(self):
        """
        Find default RGB bands by searching for B12, B11, B9 patterns.

        Returns
        -------
        List of 3 band indices (1-based) in order [B12, B11, B9].

        Uses the second group when multiple groups exist (post bands),
        otherwise falls back to [1, 2, 3].
        """
        all_band_names = [self.image.band_name(i + 1)
                          for i in range(self.image_dat.shape[2])]

        groups = []
        i = 0
        while i < len(all_band_names):
            if 'B12' in all_band_names[i]:
                b12_idx = i + 1  # 1-based
                b11_idx = None
                for j in range(i + 1, min(i + 3, len(all_band_names))):
                    if 'B11' in all_band_names[j]:
                        b11_idx = j + 1
                        for k in range(j + 1, min(j + 3, len(all_band_names))):
                            if 'B9' in all_band_names[k]:
                                b9_idx = k + 1
                                if b12_idx < b11_idx < b9_idx:
                                    groups.append([b12_idx, b11_idx, b9_idx])
                                    print(f'[CLI] Found RGB group: {[b12_idx, b11_idx, b9_idx]} '
                                          f'= {[all_band_names[b12_idx-1], all_band_names[b11_idx-1], all_band_names[b9_idx-1]]}')
                                break
                        break
            i += 1

        if len(groups) == 0:
            print('[CLI] Warning: could not find B12/B11/B9 pattern. Using bands 1, 2, 3.')
            n = self.image_dat.shape[2]
            return [1, 2, 3] if n >= 3 else [1, 1, 1]
        elif len(groups) == 1:
            print(f'[CLI] Using RGB group: {groups[0]}')
            return groups[0]
        else:
            print(f'[CLI] Found {len(groups)} RGB groups. Using second (post): {groups[1]}')
            return groups[1]

    # -----------------------------------------------------------------------
    # Data loading  (mirrors GUI.load_image / GUI.load_polygon)
    # -----------------------------------------------------------------------

    def load_image(self):
        self.image     = Raster(self.image_filename)
        self.image_dat = self.image.read_bands('all')

        n_bands = self.image_dat.shape[2] if self.image_dat.ndim > 2 else 1

        # embed_band_list: use CLI arg if given, else all bands
        if self._embed_band_list_arg is not None:
            self.embed_band_list = self._embed_band_list_arg
        else:
            self.embed_band_list = list(range(1, n_bands + 1))

        # display bands: B12/B11/B9 post group (or bands 1-3)
        self.img_band_list = self.find_default_rgb_bands()

        print(f'[CLI] Image: {self.image._xSize} x {self.image._ySize} px, '
              f'{n_bands} bands')
        print(f'[CLI] Embed bands: {self.embed_band_list}')
        print(f'[CLI] Display bands: {self.img_band_list}')

    def generate_mask_from_rgb(self):
        """
        Generate mask using dominant-band logic on the first RGB display band
        (same fallback as GUI.generate_mask_from_rgb).
        """
        rgb_bands = self.img_band_list[:3]
        rgb_data  = self.image_dat[..., [b - 1 for b in rgb_bands]]
        mask = dominant_band(X=rgb_data, band_index=1)
        return mask

    def load_polygon(self):
        """
        Load burn hint.

        If polygon_filename is provided, load it as a 0/1 raster.
        Otherwise generate a mask from the RGB bands via dominant_band().
        Mirrors GUI.load_polygon().
        """
        self.polygon_dat  = None
        self.mask_from_file = False

        if self.polygon_filename is not None:
            polygon = Raster(file_name=self.polygon_filename)

            if not polygon.is_polygon():
                raise ValueError(
                    f'VIIRS hint is not a valid single-band 0/1 raster:\n'
                    f'  {self.polygon_filename}'
                )

            self.polygon      = polygon
            self.polygon_dat  = polygon.read_bands('all').squeeze().astype(np.bool_)
            self.mask_from_file = True
            print('[CLI] Using mask from file')
        else:
            print('[CLI] No hint provided — generating mask from RGB bands (dominant band method)')
            self.polygon_dat  = self.generate_mask_from_rgb()
            self.mask_from_file = False

        self.border = extract_border(
            mask=self.polygon_dat, thickness=self.border_thickness)
        self.guessed_burn_p = float(np.nanmean(self.polygon_dat))
        print(f'[CLI] Guessed burn proportion = {self.guessed_burn_p:.4f}')

    # -----------------------------------------------------------------------
    # Sampling  (mirrors GUI.sample_data)
    # -----------------------------------------------------------------------

    def sample_data(self):
        # Clamp sample_size to the number of valid (non-NaN) pixels so
        # small crops or crops with large nodata regions don't crash.
        flat = self.image_dat.reshape(-1, self.image_dat.shape[-1])
        n_valid = int(np.isfinite(flat).all(axis=1).sum())
        if self.sample_size > n_valid:
            print(f'[CLI] Warning: sample_size ({self.sample_size}) > '
                  f'valid pixels ({n_valid}). Clamping to {n_valid}.')
            self.sample_size = n_valid

        self.sample_indices, self.samples = regular_sampling(
            raster_dat=self.image_dat,
            sample_size=self.sample_size,
            seed=self.random_state,
        )
        self.sample_in_polygon = (
            self.polygon_dat.ravel()[self.sample_indices].astype(np.bool_)
        )
        n_in = int(self.sample_in_polygon.sum())
        print(f'[CLI] {len(self.samples)} pixels sampled  '
              f'(inside hint: {n_in},  outside: {len(self.samples) - n_in})')

    # -----------------------------------------------------------------------
    # T-SNE embedding  (mirrors GUI.get_band_embed, with all params exposed)
    # -----------------------------------------------------------------------

    def get_band_embed(self):
        """
        Run T-SNE on sampled pixels using cuml (GPU).
        All T-SNE hyperparameters come from self.tsne_params so they are
        fully controlled via the CLI.
        """
        from cuml.manifold import TSNE
        import cupy as cp

        X_s   = self.samples[..., [b - 1 for b in self.embed_band_list]]
        X_gpu = cp.asarray(X_s, dtype=cp.float32)

        print(f'[CLI] T-SNE params: {self.tsne_params}')
        tsne_model = TSNE(**self.tsne_params)
        embedding  = tsne_model.fit_transform(X_gpu)
        cp.cuda.Stream.null.synchronize()

        self.current_embed     = cp.asnumpy(embedding)
        self.current_band_name = ' | '.join(
            self.image.band_name(i) for i in self.embed_band_list
        )
        print(f'[CLI] T-SNE done.  Bands: {self.current_band_name}')

    # -----------------------------------------------------------------------
    # RF image embedding  (mirrors GUI.load_image_embed_RF)
    # -----------------------------------------------------------------------

    def _get_band_image_2d(self):
        """Full image in embed bands, flattened to (N_pixels, n_bands)."""
        IMAGE = self.image_dat[..., [b - 1 for b in self.embed_band_list]]
        IMAGE = IMAGE.reshape(-1, len(self.embed_band_list))
        return np.nan_to_num(IMAGE, nan=0.0)

    def load_image_embed_RF(self):
        from machine_learning.trees import rf_regressor

        X  = self.samples[..., [b - 1 for b in self.embed_band_list]]
        y1 = self.current_embed[:, 0]
        y2 = self.current_embed[:, 1]

        reg1 = rf_regressor(X, y1, **self.rf_params)
        reg2 = rf_regressor(X, y2, **self.rf_params)

        input_img  = self._get_band_image_2d()
        img_embed1 = reg1.predict(input_img)
        img_embed2 = reg2.predict(input_img)

        return np.column_stack((img_embed1, img_embed2))

    # -----------------------------------------------------------------------
    # HDBSCAN clustering  (mirrors GUI.map_burn)
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Intermediate state caching (for serial mapping optimisation)
    # -----------------------------------------------------------------------

    def _save_intermediate(self, transformed_img):
        """Save t-SNE embedding + RF-transformed image for reuse."""
        np.savez_compressed(
            self.save_state_path,
            current_embed=self.current_embed,
            transformed_img=transformed_img,
            guessed_burn_p=np.float64(self.guessed_burn_p),
            sample_size=np.int64(self.sample_size),
        )
        print(f'[CLI] Intermediate state saved → {self.save_state_path}')

    def _load_intermediate(self):
        """Load cached t-SNE + RF state; returns transformed_img."""
        data = np.load(self.load_state_path)
        self.current_embed  = data['current_embed']
        self.guessed_burn_p = float(data['guessed_burn_p'])
        self.sample_size    = int(data['sample_size'])
        print(f'[CLI] Loaded intermediate state ({self.current_embed.shape[0]}'
              f' samples, burn_p={self.guessed_burn_p:.4f})')
        return data['transformed_img']

    # -----------------------------------------------------------------------
    # HDBSCAN clustering  (mirrors GUI.map_burn)
    # -----------------------------------------------------------------------

    def map_burn(self, transformed_img=None):
        from machine_learning.cluster import hdbscan_fit, hdbscan_approximate

        self.hdbscan_params['min_cluster_size'] = max(5, int(min(
            self.sample_size * self.guessed_burn_p          * self.controlled_ratio,
            self.sample_size * (1.0 - self.guessed_burn_p)  * self.controlled_ratio,
        )))
        print(f'[CLI] HDBSCAN min_cluster_size = '
              f'{self.hdbscan_params["min_cluster_size"]}')

        if transformed_img is None:
            t0 = time.time()
            transformed_img = self.load_image_embed_RF()
            print(f'[CLI] Forest mapping done, cost {time.time() - t0:.2f}s')

        t1 = time.time()
        cluster, _     = hdbscan_fit(self.current_embed, **self.hdbscan_params)
        img_cluster, _ = hdbscan_approximate(transformed_img, cluster)
        print(f'[CLI] Unique clusters: {np.unique(img_cluster)}')
        print(f'[CLI] HDBSCAN done, cost {time.time() - t1:.3f}s')

        return img_cluster, transformed_img

    # -----------------------------------------------------------------------
    # Classification  (mirrors GUI.classify_cluster)
    # -----------------------------------------------------------------------

    def classify_cluster(self, cluster):
        """Use the hint mask to determine which HDBSCAN cluster(s) = burned.

        For each cluster label, compute two overlap ratios:
          - precision: fraction of the cluster's pixels inside the hint
          - recall:    fraction of the hint's pixels belonging to this cluster

        A cluster is burned if EITHER ratio exceeds 50%.  This handles both
        small fires (high precision, low recall) and large fires where the
        burned cluster spills beyond the hint boundary (low precision, high
        recall — e.g. 42% precision but the cluster covers most of the hint).
        """
        classification = np.full(self.polygon_dat.shape, False)
        hint_flat = self.polygon_dat.ravel()
        cluster_flat = cluster.ravel()
        n_hint = int(hint_flat.sum())

        unique_labels = np.unique(cluster_flat)
        burned_labels = []

        for label in unique_labels:
            if label == -1:
                continue
            mask = cluster_flat == label
            n_total = int(mask.sum())
            if n_total == 0:
                continue
            n_inside = int((mask & hint_flat).sum())
            precision = n_inside / n_total
            recall    = n_inside / n_hint if n_hint > 0 else 0.0
            print(f'[CLI]   cluster {label}: {n_total} px, '
                  f'{n_inside} inside hint '
                  f'(precision={precision:.2%}, recall={recall:.2%})')
            if precision > 0.5 or recall > 0.5:
                burned_labels.append(label)

        print(f'[CLI] Burned clusters: {burned_labels}')

        for label in burned_labels:
            classification[cluster == label] = True

        return classification

    # -----------------------------------------------------------------------
    # Save classification  (mirrors GUI.save_classification)
    # -----------------------------------------------------------------------

    def save_classification(self, classification):
        base     = os.path.basename(self.image_filename)
        out_path = os.path.join(self.save_dir, f'{base}_classified.bin')
        writeENVI(
            output_filename=out_path,
            data=classification.astype(np.float32),
            mode='new',
            ref_filename=self.image_filename,
            band_names=['burned(bool)'],
        )
        print(f'[CLI] Classification saved → {out_path}')
        return out_path

    # -----------------------------------------------------------------------
    # Comparison figure
    # -----------------------------------------------------------------------

    @staticmethod
    def _iou(a, b):
        a, b  = a.astype(bool).ravel(), b.astype(bool).ravel()
        inter = int(np.sum(a & b))
        union = int(np.sum(a | b))
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _accuracy(a, b):
        return float(np.mean(a.astype(bool).ravel() == b.astype(bool).ravel()))

    def _background(self):
        """
        Return display-band composite (histogram-stretched) using the
        B12/B11/B9 post bands found by find_default_rgb_bands().
        """
        bands = self.img_band_list[:3]
        dat   = self.image_dat[..., [b - 1 for b in bands]]
        return np.clip(htrim_3d(dat), 0.0, 1.0)

    @staticmethod
    def _mask_to_outlines(mask: np.ndarray, geotransform, projection) -> list:
        """
        Polygonize a boolean mask and return a list of shapely geometries.

        Adapted from binary_polygonize.py (binary_polygonize.polygonize /
        create_in_memory_band) — same GDAL Polygonize approach but operates
        entirely in-memory and returns shapely objects instead of saving a file.
        """
        from osgeo import gdal, ogr, osr
        from shapely import wkb as shapely_wkb

        rows, cols = mask.shape
        mem_drv    = gdal.GetDriverByName('MEM')

        # Data band  (same as create_in_memory_band in binary_polygonize.py)
        data_ds = mem_drv.Create('', cols, rows, 1, gdal.GDT_Byte)
        data_ds.SetProjection(projection)
        data_ds.SetGeoTransform(geotransform)
        data_ds.GetRasterBand(1).WriteArray(mask.astype(np.uint8))

        # Mask band  (non-zero = valid pixels to polygonize)
        mask_ds = mem_drv.Create('', cols, rows, 1, gdal.GDT_Byte)
        mask_ds.SetProjection(projection)
        mask_ds.SetGeoTransform(geotransform)
        mask_ds.GetRasterBand(1).WriteArray((mask != 0).astype(np.uint8))

        # In-memory vector layer (equivalent to the .shp in binary_polygonize.py)
        srs = osr.SpatialReference()
        srs.ImportFromWkt(projection)
        mem_vec = ogr.GetDriverByName('Memory').CreateDataSource('')
        lyr     = mem_vec.CreateLayer('', srs=srs)
        lyr.CreateField(ogr.FieldDefn('v', ogr.OFTInteger))

        gdal.Polygonize(
            data_ds.GetRasterBand(1),
            mask_ds.GetRasterBand(1),
            lyr, 0, [], callback=None,
        )

        shapes = []
        for feat in lyr:
            geom = feat.GetGeometryRef()
            if geom is not None:
                shapes.append(shapely_wkb.loads(bytes(geom.ExportToWkb())))

        data_ds = mask_ds = mem_vec = None
        return shapes

    def _contour_overlay(self, mask: np.ndarray, color: str) -> np.ndarray:
        """Return an RGBA image with the boundary of *mask* drawn in *color*,
        exactly ``self.contour_width`` pixels wide.  Transparent elsewhere."""
        from scipy.ndimage import binary_dilation, binary_erosion
        import matplotlib.colors as mcolors

        mask = np.asarray(mask, dtype=bool)
        eroded = binary_erosion(mask)
        boundary = mask & ~eroded
        if self.contour_width > 1:
            boundary = binary_dilation(boundary, iterations=self.contour_width - 1)

        rgba = mcolors.to_rgba(color)
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[boundary] = rgba
        return overlay

    @staticmethod
    def _add_outlines(ax, shapes, geotransform, color, label, linewidth=1.0):
        """
        Plot polygon outlines (no fill) on *ax* in pixel space.
        Converts projected coordinates → pixel coordinates using geotransform.
        """
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path

        gt          = geotransform
        first_patch = True

        for geom in shapes:
            polys = list(geom.geoms) if hasattr(geom, 'geoms') else [geom]
            for poly in polys:
                if poly.is_empty:
                    continue
                coords = np.array(poly.exterior.coords)
                # projected CRS → pixel coordinates
                px = (coords[:, 0] - gt[0]) / gt[1]
                py = (coords[:, 1] - gt[3]) / gt[5]
                xy    = np.column_stack([px, py])
                codes = ([Path.MOVETO]
                         + [Path.LINETO] * (len(xy) - 2)
                         + [Path.CLOSEPOLY])
                patch = PathPatch(
                    Path(xy, codes),
                    facecolor='none',
                    edgecolor=color,
                    linewidth=linewidth,
                    label=label if first_patch else '_nolegend_',
                )
                ax.add_patch(patch)
                first_patch = False

    def make_brush_comparison_figure(self,
                                      raw: np.ndarray,
                                      brushed: np.ndarray) -> str:
        """
        Two-panel PNG: raw classification (left) vs brushed classification (right).
        Both shown as contour outlines on the false-colour background so the
        difference from class_brush is immediately visible.
        """
        bg  = self._background()
        h, w = bg.shape[:2]

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle(
            f'Fire: {self.fire_numbe}  —  class_brush comparison\n'
            f'Start: {self.start_date}   |   End: {self.end_date}',
            fontsize=10, fontweight='bold',
        )

        after_title = ('After class_brush\n(brushed)'
                       if brushed is not None
                       else 'After class_brush\n(FAILED — exe not found or no output)')
        after_mask  = brushed if brushed is not None else raw

        for ax, mask, title in [
            (axes[0], raw,        'Before class_brush\n(raw classification)'),
            (axes[1], after_mask, after_title),
        ]:
            ax.imshow(bg, interpolation='nearest', origin='upper')
            ax.imshow(self._contour_overlay(mask, 'red'),
                      interpolation='nearest', origin='upper')
            ax.set_title(title, fontsize=9)
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            ax.set_xlabel('Column (px)', fontsize=8)
            ax.set_ylabel('Row (px)', fontsize=8)
            ax.tick_params(labelsize=7)

        plt.tight_layout()
        fig_path = os.path.join(
            self.save_dir, f'{self.fire_numbe}_brush_comparison.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'[CLI] Brush comparison figure → {fig_path}')
        return fig_path

    def make_comparison_figure(self, classification: np.ndarray) -> str:
        """
        Single-panel figure with up to three polygon outlines (no fill)
        overlaid on the false-colour background:
          - Our mapping              — red
          - Classification hint      — lime    (VIIRS or traditional perimeter)
          - Traditional perimeter    — cyan    (only when --perimeter given
                                                and different from the hint)

        Polygon outlines are derived from the binary rasters using the same
        GDAL Polygonize approach as binary_polygonize.py.
        Legend and IoU / accuracy metrics are included in the title.
        """
        bg  = self._background()
        h, w = bg.shape[:2]
        gt  = self.image._transform
        prj = self.image._proj

        # Determine whether the hint came from a file or was auto-generated,
        # and whether the hint and the perimeter are the same file.
        hint_label = 'Hint'
        hint_is_perimeter = False
        if self.mask_from_file and self.polygon_filename:
            if self.perimeter_filename and os.path.abspath(
                    self.polygon_filename) == os.path.abspath(
                    self.perimeter_filename):
                hint_label = 'Traditional perimeter (hint)'
                hint_is_perimeter = True
            else:
                hint_label = 'VIIRS hint'

        # ---- load perimeter data (only if different from hint) ----
        perim_dat = None
        if (not hint_is_perimeter
                and self.perimeter_filename
                and os.path.exists(self.perimeter_filename)):
            try:
                perim_r   = Raster(self.perimeter_filename)
                perim_dat = perim_r.read_bands('all').squeeze().astype(bool)
            except Exception as exc:
                print(f'[CLI] Warning — could not load perimeter: {exc}')

        # ---- metrics ----
        iou_cv  = self._iou(classification, self.polygon_dat)
        acc_cv  = self._accuracy(classification, self.polygon_dat)
        metrics = (f'IoU(ours/{hint_label})={iou_cv:.3f}  '
                   f'Acc(ours/{hint_label})={acc_cv:.3f}')
        if perim_dat is not None:
            iou_cp  = self._iou(classification, perim_dat)
            iou_vp  = self._iou(self.polygon_dat, perim_dat)
            metrics += (f'\nIoU(ours/perim)={iou_cp:.3f}  '
                        f'IoU({hint_label}/perim)={iou_vp:.3f}')

        # ---- figure ----
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(bg, interpolation='nearest', origin='upper')

        # All outlines drawn as pixel-exact RGBA overlays
        ax.imshow(self._contour_overlay(classification, 'red'),
                  interpolation='nearest', origin='upper')
        ax.imshow(self._contour_overlay(self.polygon_dat, 'lime'),
                  interpolation='nearest', origin='upper')

        if perim_dat is not None:
            ax.imshow(self._contour_overlay(perim_dat, 'cyan'),
                      interpolation='nearest', origin='upper')

        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_xlabel('Column (px)', fontsize=8)
        ax.set_ylabel('Row (px)', fontsize=8)
        ax.tick_params(labelsize=7)

        # Build legend
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color='red',  linewidth=2, label='Our mapping'),
            Line2D([0], [0], color='lime', linewidth=2, label=hint_label),
        ]
        if perim_dat is not None:
            handles.append(
                Line2D([0], [0], color='cyan', linewidth=2,
                       label='Traditional perimeter'))
        ax.legend(handles=handles, loc='lower right', fontsize=9,
                  framealpha=0.7, edgecolor='white')
        ax.set_title(
            f'Fire: {self.fire_numbe}   |   '
            f'Start: {self.start_date}   |   End: {self.end_date}\n'
            f'{metrics}',
            fontsize=10, fontweight='bold',
        )

        plt.tight_layout()
        fig_path = os.path.join(
            self.save_dir, f'{self.fire_numbe}_comparison.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'[CLI] Comparison figure → {fig_path}')
        return fig_path

    # -----------------------------------------------------------------------
    # class_brush post-processing  (mirrors GUI's run_brush_then_qgis)
    # -----------------------------------------------------------------------

    def run_class_brush(self, clf_path: str) -> np.ndarray:
        """
        Call class_brush.exe directly (bypass class_brush.py which requires
        a Sentinel-2 timestamp in the image filename).

        Uses the same default parameters as class_brush.py:
          brush_size=15  point_threshold=10

        The exe writes <clf_path>_comp_NNN.bin for each component above
        threshold.  We read them back, keep only the largest (by pixel
        count), clean up, and return a boolean mask.

        Returns the brushed binary mask (same shape as classification), or
        None if the exe is not found or produces no components.
        """
        import glob as _glob

        # class_brush.exe lives at  <repo>/cpp/class_brush.exe
        # _HERE = py/fire_mapping/ → two levels up is the repo root
        brush_exe = os.path.normpath(
            os.path.join(_HERE, '..', '..', 'cpp', 'class_brush.exe'))

        if not os.path.isfile(brush_exe):
            print(f'[CLI] Warning — class_brush.exe not found at {brush_exe}, '
                  f'skipping brush.')
            return None

        # Same defaults as class_brush.py: brush_size=15, point_threshold=10
        BRUSH_SIZE  = 15
        POINT_THRES = 10

        cmd = [brush_exe, clf_path, str(BRUSH_SIZE), str(POINT_THRES)]
        print(f'[CLI] Running class_brush.exe (brush={BRUSH_SIZE}, '
              f'threshold={POINT_THRES}) ...')
        result = subprocess.run(cmd, cwd=self.save_dir)
        if result.returncode != 0:
            print(f'[CLI] Warning — class_brush.exe exited with code '
                  f'{result.returncode}')

        # Find component .bin files and keep only the largest one
        comp_files = sorted(_glob.glob(clf_path + '_comp_*.bin'))

        brushed_mask = None
        if comp_files:
            largest_file  = None
            largest_count = -1
            for cf in comp_files:
                try:
                    dat   = Raster(cf).read_bands('all').squeeze().astype(bool)
                    count = int(dat.sum())
                    if count > largest_count:
                        largest_count = count
                        largest_file  = cf
                        brushed_mask  = dat
                except Exception as exc:
                    print(f'[CLI] Warning — could not read {cf}: {exc}')

            print(f'[CLI] class_brush done — {len(comp_files)} component(s), '
                  f'using largest ({largest_count} px): '
                  f'{os.path.basename(largest_file)}')

            # Clean up all component files and their intermediaries
            for cf in comp_files:
                for p in (cf, os.path.splitext(cf)[0] + '.hdr'):
                    if os.path.exists(p):
                        os.remove(p)
            # C++ intermediaries written alongside clf_path
            for pat in [clf_path + '_flood4.bin',
                        clf_path + '_flood4.hdr',
                        clf_path + '_flood4.bin_link.bin',
                        clf_path + '_flood4.bin_link.hdr',
                        clf_path + '_flood4.bin_link.bin_recode.bin',
                        clf_path + '_flood4.bin_link.bin_recode.hdr',
                        clf_path + '_flood4.bin_link.bin_recode.bin_wheel.bin',
                        clf_path + '_flood4.bin_link.bin_recode.bin_wheel.hdr']:
                if os.path.exists(pat):
                    os.remove(pat)
        else:
            print('[CLI] class_brush.exe produced no component files.')

        return brushed_mask

    # -----------------------------------------------------------------------
    # Diagnostic PNGs  (pre / post / difference bands)
    # -----------------------------------------------------------------------

    def _find_band_groups(self):
        """Detect pre, post, and difference band groups from band names.

        Scans the ENVI band descriptions for keywords:
          - 'pre'      → pre-fire group
          - 'pst'/'post' → post-fire group
          - 'anomaly1' / 'diff' with '(post-pre)/(post+pre)' → difference-1
          - 'anomaly2' / 'ratio' with 'post/pre'             → difference-2

        If no keywords are found, falls back to positional B12/B11/B9 groups
        (first = pre, second = post).

        Returns dict: {'pre': [...], 'post': [...], 'diff1': [...], 'diff2': [...]}
        Each value is a list of 1-based band indices (up to 3 bands).
        """
        n_bands = self.image_dat.shape[2]
        raw_names = [self.image.band_info_list[i] for i in range(n_bands)]

        groups = {'pre': [], 'post': [], 'diff1': [], 'diff2': []}

        for i, name in enumerate(raw_names):
            low = name.lower()
            idx = i + 1  # 1-based
            if 'anomaly2' in low or ('post/pre' in low and 'anomaly' not in low):
                groups['diff2'].append(idx)
            elif 'anomaly1' in low or '(post-pre)/(post+pre)' in low:
                groups['diff1'].append(idx)
            elif low.startswith('pst') or low.startswith('post'):
                groups['post'].append(idx)
            elif low.startswith('pre'):
                groups['pre'].append(idx)

        # If keyword-based detection found something, trim each to first 3
        has_keywords = any(len(v) > 0 for v in groups.values())
        if has_keywords:
            for k in groups:
                groups[k] = groups[k][:3]
            print(f'[CLI] Band groups (keyword): '
                  f'pre={groups["pre"]} post={groups["post"]} '
                  f'diff1={groups["diff1"]} diff2={groups["diff2"]}')
            return groups

        # Fallback: use positional B12/B11/B9 groups
        all_band_names = [self.image.band_name(i + 1) for i in range(n_bands)]
        positional = []
        i = 0
        while i < len(all_band_names):
            if 'B12' in all_band_names[i]:
                for j in range(i + 1, min(i + 3, len(all_band_names))):
                    if 'B11' in all_band_names[j]:
                        for k in range(j + 1, min(j + 3, len(all_band_names))):
                            if 'B9' in all_band_names[k]:
                                positional.append([i + 1, j + 1, k + 1])
                                break
                        break
            i += 1

        if len(positional) >= 2:
            groups['pre'] = positional[0]
            groups['post'] = positional[1]
        elif len(positional) == 1:
            groups['post'] = positional[0]
        else:
            # Last resort: first 3 bands
            groups['post'] = list(range(1, min(4, n_bands + 1)))

        print(f'[CLI] Band groups (positional fallback): '
              f'pre={groups["pre"]} post={groups["post"]} '
              f'diff1={groups["diff1"]} diff2={groups["diff2"]}')
        return groups

    def make_diagnostic_pngs(self):
        """Generate RGB PNGs for each detected band group (pre, post, diff1, diff2)."""
        groups = self._find_band_groups()

        labels = {
            'pre':   'Pre-fire',
            'post':  'Post-fire',
            'diff1': 'Difference (post-pre)/(post+pre)',
            'diff2': 'Difference post/pre',
        }

        paths = []
        for key in ('pre', 'post', 'diff1', 'diff2'):
            band_indices = groups[key]
            if not band_indices:
                continue

            dat = self.image_dat[..., [b - 1 for b in band_indices]]

            # Normalise each channel independently to [0, 1] via 2-98 percentile
            channels = []
            for c in range(dat.shape[2]):
                ch = dat[..., c].astype(np.float32)
                lo, hi = np.nanpercentile(ch, [2, 98])
                channels.append(np.clip((ch - lo) / max(hi - lo, 1e-6), 0, 1))
            rgb = np.stack(channels, axis=2)

            # Build band label string
            n_bands_total = self.image_dat.shape[2]
            band_labels = []
            for b in band_indices:
                raw = self.image.band_info_list[b - 1] if b <= n_bands_total else f'band {b}'
                band_labels.append(raw)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(rgb, interpolation='nearest', origin='upper')
            ax.set_title(f'{self.fire_numbe}  —  {labels[key]}\n'
                         f'Bands: {", ".join(band_labels)}',
                         fontsize=9)
            dh, dw = rgb.shape[:2]
            ax.set_xlim(0, dw)
            ax.set_ylim(dh, 0)
            ax.set_xlabel('Column (px)', fontsize=8)
            ax.set_ylabel('Row (px)', fontsize=8)
            ax.tick_params(labelsize=7)

            fig_path = os.path.join(
                self.save_dir, f'{self.fire_numbe}_{key}.png')
            fig.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'[CLI] Diagnostic PNG → {os.path.basename(fig_path)}')
            paths.append(fig_path)

        return paths

    # -----------------------------------------------------------------------
    # Full pipeline
    # -----------------------------------------------------------------------

    def run(self):
        hint_name = (os.path.basename(self.polygon_filename)
                     if self.polygon_filename else '(dominant band fallback)')
        resume = self.load_state_path is not None

        if resume:
            _status_box(
                f'FIRE MAPPING CLI (resume)  |  {self.fire_numbe}',
                [
                    f'Image   : {os.path.basename(self.image_filename)}',
                    f'Hint    : {hint_name}',
                    f'State   : {os.path.basename(self.load_state_path)}',
                    f'Output  : {self.save_dir}',
                ]
            )
            print('[1/4] Loading image ...')
            self.load_image()

            print('\n[2/4] Loading hint ...')
            self.load_polygon()

            print('\n[3/4] Loading cached t-SNE + RF state ...')
            transformed_img = self._load_intermediate()
        else:
            _status_box(
                f'FIRE MAPPING CLI  |  {self.fire_numbe}',
                [
                    f'Image   : {os.path.basename(self.image_filename)}',
                    f'Hint    : {hint_name}',
                    f'Output  : {self.save_dir}',
                ]
            )
            print('[1/7] Loading image ...')
            self.load_image()

            print('\nGenerating diagnostic PNGs (pre/post/diff) ...')
            self.make_diagnostic_pngs()

            print('\n[2/7] Loading hint ...')
            self.load_polygon()

            print(f'\n[3/7] Sampling {self.sample_size} pixels '
                  f'(seed={self.random_state}) ...')
            self.sample_data()

            print(f'\n[4/7] T-SNE embedding on bands '
                  f'{self.embed_band_list} ...')
            self.get_band_embed()
            transformed_img = None   # map_burn will compute RF

        step = '4/4' if resume else '5/7'
        print(f'\n[{step}] Mapping burn  '
              f'{"(HDBSCAN only)" if resume else "(RF + HDBSCAN)"} ...')
        img_cluster, transformed_img = self.map_burn(
            transformed_img=transformed_img)

        # Save intermediate state after first full run
        if self.save_state_path and not resume:
            self._save_intermediate(transformed_img)

        img_cluster    = img_cluster.reshape(
            self.image._ySize, self.image._xSize)

        classification = self.classify_cluster(img_cluster)
        burned = int(classification.sum())
        total  = int(classification.size)
        print(f'[CLI] Burned pixels: {burned:,} / {total:,}  '
              f'({100.0 * burned / total:.2f} %)')

        print(f'\nSaving classification ...')
        clf_path = self.save_classification(classification)

        print(f'\nRunning class_brush post-processing ...')
        brushed = self.run_class_brush(clf_path)

        print('\nGenerating figures ...')
        self.make_brush_comparison_figure(classification, brushed)
        fig_path = self.make_comparison_figure(
            brushed if brushed is not None else classification)

        _status_box(
            f'DONE  |  {self.fire_numbe}',
            [
                f'Classification : {os.path.basename(clf_path)}',
                f'Figure         : {os.path.basename(fig_path)}',
            ]
        )
        return clf_path, fig_path


# ===========================================================================
# CLI entry point
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='fire_mapping_cli.py',
        description='Non-GUI Sentinel-2 burn mapping: T-SNE + RF + HDBSCAN.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Minimal — hint auto-generated from dominant band
  python py/fire_mapping/fire_mapping_cli.py scene_crop.bin

  # With VIIRS hint
  python py/fire_mapping/fire_mapping_cli.py scene_crop.bin viirs_acc.bin

  # Full metadata + perimeter comparison
  python py/fire_mapping/fire_mapping_cli.py scene_crop.bin viirs_acc.bin  \\
      --fire_numbe C11659 --start_date 2025-08-20 --end_date 2025-10-14   \\
      --perimeter  C11659_perim.bin

  # Custom hyperparameters
  python py/fire_mapping/fire_mapping_cli.py scene_crop.bin viirs_acc.bin  \\
      --embed_bands 1,2,3 --sample_size 5000                               \\
      --tsne_perplexity 40 --hdbscan_min_samples 10
        """,
    )

    # ---- Positional ----
    p.add_argument('image',
                   help='Sentinel-2 ENVI .bin raster (cropped around fire)')
    p.add_argument('viirs_guess', nargs='?', default=None,
                   help='Accumulated VIIRS binary raster (.bin) used as burn hint '
                        '(optional — dominant-band mask is used when omitted)')

    # ---- Metadata ----
    p.add_argument('--fire_numbe',  default=None,
                   help='Fire identifier string, e.g. C11659 (used in output filenames)')
    p.add_argument('--start_date',  default=None,
                   help='Fire start date YYYY-MM-DD (displayed in figure title)')
    p.add_argument('--end_date',    default=None,
                   help='Last VIIRS detection date YYYY-MM-DD (displayed in figure title)')
    p.add_argument('--perimeter',   default=None,
                   help='Rasterized traditional fire perimeter .bin '
                        '(optional — adds 3rd comparison panel)')

    # ---- Sampling ----
    p.add_argument('--sample_size',  type=int,   default=10_000,
                   help='Pixels to sample for T-SNE (default: 10000)')
    p.add_argument('--seed',         type=int,   default=123,
                   help='Random seed for sampling and RF (default: 123)')
    p.add_argument('--embed_bands',  default=None,
                   help='1-indexed band list for T-SNE embedding, '
                        'comma-separated (default: all bands)')

    # ---- Random Forest ----
    p.add_argument('--rf_n_estimators',  type=int,   default=100)
    p.add_argument('--rf_max_depth',     type=int,   default=15)
    p.add_argument('--rf_max_features',  default='sqrt',
                   help="RF max_features ('sqrt', 'log2', or int; default: sqrt)")
    p.add_argument('--rf_random_state',  type=int,   default=42)

    # ---- HDBSCAN ----
    p.add_argument('--controlled_ratio',    type=float, default=0.5,
                   help='Scales HDBSCAN min_cluster_size relative to the '
                        'minority class proportion (default: 0.5)')
    p.add_argument('--hdbscan_min_samples', type=int,   default=20,
                   help='HDBSCAN min_samples — controls cluster conservativeness '
                        '(default: 20)')

    # ---- T-SNE (all params exposed) ----
    p.add_argument('--tsne_perplexity',    type=float, default=60.0)
    p.add_argument('--tsne_learning_rate', type=float, default=200.0)
    p.add_argument('--tsne_max_iter',      type=int,   default=2000)
    p.add_argument('--tsne_init',          default='pca',
                   choices=['pca', 'random'])
    p.add_argument('--tsne_n_components',  type=int,   default=2)
    p.add_argument('--tsne_random_state',  type=int,   default=42)

    # ---- Figure ----
    p.add_argument('--contour_width', type=int, default=1,
                   help='Contour line width in pixels (default: 1)')

    # ---- Intermediate state caching (serial mapping optimisation) ----
    p.add_argument('--save_state', default=None,
                   help='Save t-SNE + RF state to .npz after embedding '
                        '(for reuse by subsequent --load_state runs)')
    p.add_argument('--load_state', default=None,
                   help='Load cached t-SNE + RF state from .npz, '
                        'skip straight to HDBSCAN')

    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)

    embed_band_list = (
        [int(b.strip()) for b in args.embed_bands.split(',')]
        if args.embed_bands is not None else None
    )

    cli = FireMappingCLI(
        image_filename      = args.image,
        polygon_filename    = args.viirs_guess,
        fire_numbe          = args.fire_numbe,
        start_date          = args.start_date,
        end_date            = args.end_date,
        perimeter_filename  = args.perimeter,
        sample_size         = args.sample_size,
        random_state        = args.seed,
        embed_band_list     = embed_band_list,
        rf_n_estimators     = args.rf_n_estimators,
        rf_max_depth        = args.rf_max_depth,
        rf_max_features     = args.rf_max_features,
        rf_random_state     = args.rf_random_state,
        controlled_ratio    = args.controlled_ratio,
        hdbscan_min_samples = args.hdbscan_min_samples,
        tsne_perplexity     = args.tsne_perplexity,
        tsne_learning_rate  = args.tsne_learning_rate,
        tsne_max_iter       = args.tsne_max_iter,
        tsne_init           = args.tsne_init,
        tsne_n_components   = args.tsne_n_components,
        tsne_random_state   = args.tsne_random_state,
        contour_width       = args.contour_width,
        save_state_path     = args.save_state,
        load_state_path     = args.load_state,
    )

    cli.run()


if __name__ == '__main__':
    main()
