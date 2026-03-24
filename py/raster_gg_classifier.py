#!/usr/bin/env python3
"""
gg_classifier.py — Generalized Gaussian Binary Classifier for multi-band raster imagery.

Usage
-----
  # Fit a model on a raster + binary mask:
  python gg_classifier.py --fit --image scene.tif --mask burned.tif --model run1

  # Optionally select best band subset by KL divergence before fitting:
  python gg_classifier.py --fit --KL --image scene.tif --mask burned.tif --model run1

  # Apply a saved model to a new raster:
  python gg_classifier.py --apply --image new_scene.tif --model run1

Outputs (--fit)
  run1.pkl          — pickled model
  run1.txt          — human-readable model summary
  (all diagnostic figures displayed interactively)

Outputs (--apply)
  run1_prediction.tif  — probability map  (Float32)
  run1_classified.tif  — binary classification  (Byte)
  (all diagnostic figures displayed interactively)
"""

import argparse
import os
import pickle
import sys
import warnings
from itertools import combinations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.linalg import inv
from scipy.stats import multivariate_normal, norm
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

try:
    from osgeo import gdal
except ImportError:
    sys.exit("ERROR: osgeo.gdal not found.  Install python3-gdal or gdal Python bindings.")


# ─────────────────────────────────────────────────────────────────────────────
# GDAL raster helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_band_raw(band, cols: int, rows: int) -> np.ndarray:
    """Read a GDAL band as float32 without using gdal_array (numpy-2 compatible)."""
    buf = band.ReadRaster(0, 0, cols, rows,
                          buf_xsize=cols, buf_ysize=rows,
                          buf_type=gdal.GDT_Float32)
    return np.frombuffer(buf, dtype=np.float32).reshape(rows, cols)


def _write_band_raw(band, arr2d: np.ndarray, gdal_dtype: int):
    """Write a 2-D array to a GDAL band without using gdal_array."""
    rows, cols = arr2d.shape
    if gdal_dtype == gdal.GDT_Float32:
        buf = arr2d.astype(np.float32).tobytes()
    elif gdal_dtype == gdal.GDT_Byte:
        buf = arr2d.astype(np.uint8).tobytes()
    else:
        buf = arr2d.astype(np.float32).tobytes()
        gdal_dtype = gdal.GDT_Float32
    band.WriteRaster(0, 0, cols, rows, buf,
                     buf_xsize=cols, buf_ysize=rows,
                     buf_type=gdal_dtype)


def read_raster(path: str) -> tuple[np.ndarray, list[str], dict]:
    """
    Read all bands from a GDAL-readable raster.

    Returns
    -------
    data        : (rows*cols, n_bands) float32 array  (NaN where nodata)
    band_names  : list of str, one per band
    meta        : dict with keys: cols, rows, n_bands, geotransform, projection, nodata_vals, path
    """
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        sys.exit(f"ERROR: Cannot open raster: {path}")

    cols      = ds.RasterXSize
    rows      = ds.RasterYSize
    n_bands   = ds.RasterCount
    gt        = ds.GetGeoTransform()
    proj      = ds.GetProjection()

    band_names  = []
    nodata_vals = []
    arrays      = []

    for b in range(1, n_bands + 1):
        band = ds.GetRasterBand(b)
        desc = band.GetDescription()
        name = desc.strip() if desc and desc.strip() else f"band_{b}"
        band_names.append(name)

        nd = band.GetNoDataValue()
        nodata_vals.append(nd)

        arr = _read_band_raw(band, cols, rows).astype(np.float32)
        if nd is not None:
            arr[arr == nd] = np.nan
        arrays.append(arr.ravel())

    ds = None
    data = np.column_stack(arrays)   # (N, B)

    meta = dict(cols=cols, rows=rows, n_bands=n_bands,
                geotransform=gt, projection=proj,
                nodata_vals=nodata_vals, path=path)
    return data, band_names, meta


def read_mask(path: str, cols: int, rows: int) -> np.ndarray:
    """
    Read a single-band 0/1 classification mask.
    Returns flat (N,) int array.
    """
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        sys.exit(f"ERROR: Cannot open mask raster: {path}")
    band = ds.GetRasterBand(1)
    buf  = band.ReadRaster(0, 0, cols, rows,
                           buf_xsize=cols, buf_ysize=rows,
                           buf_type=gdal.GDT_Byte)
    arr  = np.frombuffer(buf, dtype=np.uint8).reshape(rows, cols)
    ds   = None
    if arr.shape != (rows, cols):
        sys.exit(f"ERROR: Mask shape {arr.shape} does not match image ({rows}, {cols}).")
    return arr.ravel().astype(int)


def write_raster(path: str, array: np.ndarray, cols: int, rows: int,
                 geotransform, projection: str,
                 dtype=gdal.GDT_Float32, nodata=None):
    """Write a single-band result raster."""
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(path, cols, rows, 1, dtype,
                       ["COMPRESS=LZW", "TILED=YES"])
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection)
    band = ds.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(nodata)
    _write_band_raw(band, array.reshape(rows, cols), dtype)
    band.FlushCache()
    ds = None


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class GeneralizedGaussianBinary:
    """
    Binary classifier based on a Generalized Gaussian (power-exponential)
    distribution fitted independently to each class.

    The log-likelihood of class c is:
        log p(x|c) = -[ (x-mu_c)^T Sigma_c^{-1} (x-mu_c) ]^{beta/2}

    Posterior probabilities are obtained via softmax over the two
    class log-likelihoods weighted by class priors.

    Attributes stored after fit()
    --------------------------------
    beta          : shape parameter  (beta=1 -> Laplacian; beta=2 -> Gaussian)
    params        : dict {0: {mu, cov, inv_cov}, 1: {mu, cov, inv_cov}}
    priors        : dict {0: float, 1: float}
    band_names    : list[str]  — band names used during training
    best_threshold: float      — Youden-J optimal decision threshold
    selected_bands: list[int]  — indices into band_names (all bands, or KL subset)
    """

    def __init__(self, beta: float = 1.5):
        self.beta           = beta
        self.params         = {}
        self.priors         = {}
        self.band_names     = []
        self.best_threshold = 0.5
        self.selected_bands = []

    # ── core ─────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray,
            band_names: list[str], selected_bands: list[int] | None = None):
        """
        Parameters
        ----------
        X             : (N, B) array — full band matrix
        y             : (N,)   binary labels {0,1}
        band_names    : length-B list of names for all bands in X
        selected_bands: indices of bands to use (None = use all)
        """
        self.band_names    = list(band_names)
        self.selected_bands = list(selected_bands) if selected_bands is not None \
                              else list(range(X.shape[1]))
        Xs = X[:, self.selected_bands]

        for c in [0, 1]:
            Xc  = Xs[y == c]
            mu  = Xc.mean(axis=0)
            cov = np.cov(Xc, rowvar=False) + 1e-6 * np.eye(Xs.shape[1])
            self.params[c]  = {"mu": mu, "cov": cov, "inv_cov": inv(cov)}
            self.priors[c]  = len(Xc) / len(X)

        # Youden-J threshold on training data
        probs = self.predict_proba(X)[:, 1]
        fpr, tpr, thresholds = roc_curve(y, probs)
        self.best_threshold  = float(thresholds[np.argmax(tpr - fpr)])

    def _log_pdf(self, X: np.ndarray, mu: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
        diff  = X - mu
        mahal = np.sum(diff @ inv_cov * diff, axis=1)
        return -(mahal ** (self.beta / 2.0))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (N, 2) probability array using selected bands."""
        Xs       = X[:, self.selected_bands]
        log_probs = []
        for c in [0, 1]:
            p  = self.params[c]
            ll = self._log_pdf(Xs, p["mu"], p["inv_cov"])
            log_probs.append(ll + np.log(self.priors[c]))
        log_probs = np.vstack(log_probs).T
        probs = np.exp(log_probs - log_probs.max(axis=1, keepdims=True))
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.best_threshold).astype(int)

    def mahalanobis(self, X: np.ndarray) -> np.ndarray:
        Xs     = X[:, self.selected_bands]
        diff   = Xs - self.params[1]["mu"]
        inv_cov = self.params[1]["inv_cov"]
        d2     = np.sum(diff @ inv_cov * diff, axis=1)
        return np.sqrt(np.clip(d2, 0, None))

    # ── serialisation ─────────────────────────────────────────────────────────

    def save(self, stem: str):
        """Save .pkl and .txt representations with the same stem."""
        pkl_path = stem + ".pkl"
        txt_path = stem + ".txt"

        with open(pkl_path, "wb") as f:
            pickle.dump(self, f)

        used_names = [self.band_names[i] for i in self.selected_bands]
        lines = [
            "GeneralizedGaussianBinary Model",
            "=" * 60,
            f"beta (shape parameter)  : {self.beta}",
            f"best_threshold (Youden) : {self.best_threshold:.6f}",
            f"selected band indices   : {self.selected_bands}",
            f"selected band names     : {used_names}",
            f"all training bands      : {self.band_names}",
            "",
            "Class priors",
            "-" * 40,
        ]
        for c in [0, 1]:
            lines.append(f"  class {c}: {self.priors[c]:.4f}")

        for c in [0, 1]:
            lines += [
                "",
                f"Class {c} parameters",
                "-" * 40,
                f"  mu  : {self.params[c]['mu'].tolist()}",
                "  cov (diagonal):",
            ]
            cov = self.params[c]["cov"]
            for bi, name in enumerate(used_names):
                lines.append(f"    [{bi:2d}] {name:20s}  var={cov[bi, bi]:.6f}  std={np.sqrt(cov[bi, bi]):.6f}")
            lines += [
                "  full covariance matrix:",
                np.array2string(cov, precision=6, suppress_small=True,
                                prefix="    "),
            ]

        with open(txt_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Model saved: {pkl_path}")
        print(f"Model text:  {txt_path}")

    @classmethod
    def load(cls, stem: str) -> "GeneralizedGaussianBinary":
        pkl_path = stem + ".pkl"
        if not os.path.exists(pkl_path):
            sys.exit(f"ERROR: Model file not found: {pkl_path}")
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded: {pkl_path}")
        return model


# ─────────────────────────────────────────────────────────────────────────────
# Band validation
# ─────────────────────────────────────────────────────────────────────────────

def resolve_band_indices(required_names: list[str],
                         available_names: list[str]) -> list[int]:
    """
    Return indices into available_names for each name in required_names.
    Exits with an informative error if any required name is missing.
    """
    avail_map = {name: idx for idx, name in enumerate(available_names)}
    indices   = []
    missing   = []
    for name in required_names:
        if name in avail_map:
            indices.append(avail_map[name])
        else:
            missing.append(name)
    if missing:
        sys.exit(
            f"ERROR: The following bands required by the model are missing "
            f"from the input raster:\n  {missing}\n"
            f"Available bands: {available_names}"
        )
    return indices


# ─────────────────────────────────────────────────────────────────────────────
# KL-based band selection
# ─────────────────────────────────────────────────────────────────────────────

def _kl_gauss_nd(mu_p: np.ndarray, cov_p: np.ndarray,
                 mu_q: np.ndarray, cov_q: np.ndarray) -> float:
    """Closed-form KL(N_p || N_q) for two k-dimensional Gaussians."""
    k         = len(mu_p)
    inv_q     = np.linalg.inv(cov_q)
    _, ld_q   = np.linalg.slogdet(cov_q)
    _, ld_p   = np.linalg.slogdet(cov_p)
    diff      = mu_q - mu_p
    return 0.5 * (np.trace(inv_q @ cov_p) + diff @ inv_q @ diff - k + ld_q - ld_p)


def select_bands_by_kl(X: np.ndarray, y: np.ndarray,
                        band_names: list[str],
                        verbose: bool = True) -> list[int]:
    """
    Greedy forward band selection that maximises pairwise KL divergence
    KL(pos || neg) + KL(neg || pos)  (symmetric KL) in the joint subspace.

    Algorithm:
      1. Score every single band by its 1-D symmetric KL.
      2. Start with the best single band.
      3. Greedily add the band that most increases the joint symmetric KL
         until adding another band no longer improves it.

    Returns list of selected band indices (subset of range(n_bands)).
    """
    n_bands  = X.shape[1]
    pos_mask = y == 1
    neg_mask = y == 0

    def sym_kl(idx_list):
        Xp = X[pos_mask][:, idx_list]
        Xn = X[neg_mask][:, idx_list]
        mu_p  = Xp.mean(0)
        mu_q  = Xn.mean(0)
        cov_p = np.cov(Xp, rowvar=False).reshape(len(idx_list), len(idx_list)) \
                + 1e-6 * np.eye(len(idx_list))
        cov_q = np.cov(Xn, rowvar=False).reshape(len(idx_list), len(idx_list)) \
                + 1e-6 * np.eye(len(idx_list))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return (_kl_gauss_nd(mu_p, cov_p, mu_q, cov_q) +
                        _kl_gauss_nd(mu_q, cov_q, mu_p, cov_p))
            except np.linalg.LinAlgError:
                return 0.0

    # Step 1: rank single bands
    kl1 = np.array([sym_kl([b]) for b in range(n_bands)])
    selected    = [int(np.argmax(kl1))]
    best_score  = kl1[selected[0]]
    remaining   = [b for b in range(n_bands) if b not in selected]

    if verbose:
        print(f"[KL selection] seed band: {band_names[selected[0]]}  "
              f"KL={best_score:.4f}")

    # Step 2: greedy forward
    while remaining:
        scores = {}
        for b in remaining:
            candidate = selected + [b]
            scores[b] = sym_kl(candidate)
        best_next = max(scores, key=scores.get)
        if scores[best_next] > best_score:
            best_score = scores[best_next]
            selected.append(best_next)
            remaining.remove(best_next)
            if verbose:
                print(f"[KL selection] +{band_names[best_next]:20s}  "
                      f"cumulative KL={best_score:.4f}")
        else:
            break   # no improvement

    selected.sort()
    if verbose:
        print(f"[KL selection] final bands ({len(selected)}): "
              f"{[band_names[i] for i in selected]}")
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Figures — all original figures from the notebook, parameterised
# ─────────────────────────────────────────────────────────────────────────────

def _draw_ellipse(ax, mu2, cov2, n_std=1.0, **kwargs):
    vals, vecs = np.linalg.eigh(cov2)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h  = 2 * n_std * np.sqrt(np.clip(vals, 0, None))
    ell   = Ellipse(xy=mu2, width=w, height=h, angle=theta, **kwargs)
    ax.add_patch(ell)


def plot_fit_figures(X: np.ndarray, y: np.ndarray,
                     model: "GeneralizedGaussianBinary",
                     feature_names: list[str],
                     x_size: int, y_size: int,
                     map_result: np.ndarray | None = None):
    """All diagnostic figures for --fit mode."""

    probs       = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)
    maha        = model.mahalanobis(X)
    auc         = roc_auc_score(y, probs)
    fpr, tpr, _ = roc_curve(y, probs)

    sel   = model.selected_bands
    n_sel = len(sel)
    fnames_sel = [feature_names[i] for i in sel]

    mu1  = model.params[1]["mu"]
    mu0  = model.params[0]["mu"]
    cov1 = model.params[1]["cov"]
    cov0 = model.params[0]["cov"]
    std1 = np.sqrt(np.diag(cov1))
    std0 = np.sqrt(np.diag(cov0))

    pos_mask = y == 1
    neg_mask = y == 0

    # ── 1. Map comparison ────────────────────────────────────────────────────
    imgs = [predictions.reshape(x_size, y_size)]
    titles = ["GG Prediction"]
    if map_result is not None:
        imgs.append(map_result)
        titles.append("Actual")
    fig, axes = plt.subplots(1, len(imgs), figsize=(7 * len(imgs), 5))
    if len(imgs) == 1:
        axes = [axes]
    for ax, img, t in zip(axes, imgs, titles):
        ax.imshow(img, cmap="gray"); ax.set_title(t); ax.axis("off")
    plt.suptitle("Generalized Gaussian — Classification Map")
    plt.tight_layout(); plt.show()

    # ── 2. Probability map ───────────────────────────────────────────────────
    plt.figure(figsize=(7, 5))
    im = plt.imshow(probs.reshape(x_size, y_size), cmap="hot", vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046)
    plt.title("Probability Map"); plt.axis("off")
    plt.tight_layout(); plt.show()

    # ── 3. Mahalanobis distance map ──────────────────────────────────────────
    plt.figure(figsize=(7, 5))
    im2 = plt.imshow(maha.reshape(x_size, y_size), cmap="viridis_r",
                     vmin=0, vmax=np.percentile(maha, 95))
    plt.colorbar(im2, fraction=0.046)
    plt.title("Mahalanobis Distance (positive class centroid)")
    plt.axis("off"); plt.tight_layout(); plt.show()

    # ── 4. Band correlation matrix ───────────────────────────────────────────
    corr = np.corrcoef(X[:, sel].T)
    plt.figure(figsize=(max(5, n_sel), max(4, n_sel - 1)))
    im3 = plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im3, fraction=0.046)
    plt.xticks(range(n_sel), fnames_sel, rotation=40, ha="right", fontsize=8)
    plt.yticks(range(n_sel), fnames_sel, fontsize=8)
    plt.title("Band Correlation Matrix (selected bands)")
    plt.tight_layout(); plt.show()

    # ── 5. ROC curve ─────────────────────────────────────────────────────────
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, lw=2, color="#e8775a", label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.fill_between(fpr, tpr, alpha=0.12, color="#e8775a")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC — Generalized Gaussian")
    plt.legend(); plt.tight_layout(); plt.show()

    # ── 6. Confusion matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(y, predictions)
    fig, ax = plt.subplots(figsize=(4, 3))
    ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"]).plot(
        ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix"); plt.tight_layout(); plt.show()

    # ── 7. Permutation importance ────────────────────────────────────────────
    baseline   = roc_auc_score(y, probs)
    importance = np.zeros(n_sel)
    rng        = np.random.default_rng(42)
    for ji, bi in enumerate(sel):
        Xp = X.copy()
        perm = rng.permutation(Xp[:, bi])
        Xp[:, bi] = perm
        importance[ji] = baseline - roc_auc_score(y, model.predict_proba(Xp)[:, 1])

    idx = np.argsort(importance)[::-1]
    plt.figure(figsize=(8, 4))
    plt.bar(range(n_sel), importance[idx],
            color=plt.cm.viridis(np.linspace(0.2, 0.85, n_sel)))
    plt.xticks(range(n_sel), [fnames_sel[i] for i in idx], rotation=35, ha="right")
    plt.title("Permutation Importance  (AUC drop)")
    plt.ylabel("ΔAUC"); plt.tight_layout(); plt.show()

    # ── 8. Mahalanobis distribution by class ─────────────────────────────────
    lo, hi = np.nanpercentile(maha, [0.5, 99.5])
    bins   = np.linspace(lo, hi, 60)
    plt.figure(figsize=(7, 4))
    plt.hist(maha[neg_mask], bins=bins, density=True, alpha=0.45,
             color="#6a9fd8", label="Negative (0)")
    plt.hist(maha[pos_mask], bins=bins, density=True, alpha=0.65,
             color="#e8775a", label="Positive (1)")
    plt.xlabel("Mahalanobis Distance"); plt.ylabel("Density")
    plt.title("Mahalanobis Distribution by Class")
    plt.legend(); plt.tight_layout(); plt.show()

    # ── 9a. 1-D projections + decision boundary + KL ─────────────────────────
    cols_per_row = min(4, n_sel)
    hist_rows    = int(np.ceil(n_sel / cols_per_row))
    fig, axes    = plt.subplots(hist_rows, cols_per_row,
                                 figsize=(4.5 * cols_per_row, 3.5 * hist_rows))
    axes = np.array(axes).reshape(hist_rows, cols_per_row)

    kl_per_band = []
    for ki in range(n_sel):
        r, c = divmod(ki, cols_per_row)
        ax   = axes[r, c]
        bi   = sel[ki]
        vals = X[:, bi]
        lo_b, hi_b = np.nanpercentile(vals, [1, 99])
        band_bins  = np.linspace(lo_b, hi_b, 60)

        ax.hist(vals[neg_mask], bins=band_bins, density=True, alpha=0.40,
                color="#6a9fd8", label="Negative (0)")
        ax.hist(vals[pos_mask], bins=band_bins, density=True, alpha=0.60,
                color="#e8775a", label="Positive (1)")

        x_range = np.linspace(lo_b, hi_b, 300)
        p1 = norm.pdf(x_range, mu1[ki], std1[ki])
        p0 = norm.pdf(x_range, mu0[ki], std0[ki])
        ax.plot(x_range, p1, color="#c0392b", lw=2, label="N(μ₁,σ₁)")
        ax.plot(x_range, p0, color="#2471a3", lw=2, linestyle="--", label="N(μ₀,σ₀)")

        # Decision boundary: where p1 = p0
        db_x = x_range[np.argmin(np.abs(p1 - p0))]
        ax.axvline(db_x, color="black", lw=1.4, linestyle="--",
                   label=f"Boundary ({db_x:.3f})")

        # KL divergence
        kl = (np.log(std0[ki] / std1[ki])
              + (std1[ki]**2 + (mu1[ki] - mu0[ki])**2) / (2 * std0[ki]**2)
              - 0.5)
        kl_per_band.append(kl)

        ax.set_title(f"{fnames_sel[ki]}\nKL(+‖−)={kl:.3f}", fontsize=9)
        ax.set_xlabel("Value", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right")

    for ki in range(n_sel, hist_rows * cols_per_row):
        r, c = divmod(ki, cols_per_row)
        axes[r, c].set_visible(False)

    kl_per_band = np.array(kl_per_band)
    fig.suptitle("1-D Projections of N-D Gaussian | Red/Blue = Positive/Negative\n"
                 "Black dashed = decision boundary  |  KL = class separability",
                 fontsize=9)
    plt.tight_layout(); plt.show()

    # ── 9b. KL bar chart ─────────────────────────────────────────────────────
    kl_idx = np.argsort(kl_per_band)[::-1]
    plt.figure(figsize=(8, 4))
    plt.bar(range(n_sel), kl_per_band[kl_idx],
            color=plt.cm.plasma(np.linspace(0.15, 0.85, n_sel)))
    plt.xticks(range(n_sel), [fnames_sel[i] for i in kl_idx],
               rotation=35, ha="right")
    plt.title("Per-band KL Divergence  KL(+‖−)  (Gaussian marginals)")
    plt.ylabel("KL Divergence (nats)"); plt.tight_layout(); plt.show()

    # ── 10. Mutual information ────────────────────────────────────────────────
    mi     = mutual_info_classif(X[:, sel], y, random_state=42)
    mi_idx = np.argsort(mi)[::-1]
    plt.figure(figsize=(8, 4))
    plt.bar(range(n_sel), mi[mi_idx],
            color=plt.cm.YlOrRd(np.linspace(0.3, 0.9, n_sel)))
    plt.xticks(range(n_sel), [fnames_sel[i] for i in mi_idx],
               rotation=35, ha="right")
    plt.title("Mutual Information  (non-Gaussian, k-NN estimator)")
    plt.ylabel("MI (nats)"); plt.tight_layout(); plt.show()

    # ── 10b. MI vs KL scatter ────────────────────────────────────────────────
    if n_sel > 1:
        kl_n = kl_per_band / (kl_per_band.sum() + 1e-12)
        mi_n = mi / (mi.sum() + 1e-12)
        fig, ax_sc = plt.subplots(figsize=(5, 4))
        ax_sc.scatter(kl_n, mi_n, s=80, color="#8e44ad", zorder=3)
        for ki in range(n_sel):
            ax_sc.annotate(fnames_sel[ki], (kl_n[ki], mi_n[ki]),
                           fontsize=8, xytext=(4, 3), textcoords="offset points")
        _ref = np.linspace(0, max(kl_n.max(), mi_n.max()) * 1.1, 100)
        ax_sc.plot(_ref, _ref, "k--", lw=0.8, label="MI=KL baseline")
        ax_sc.set_xlabel("KL (norm.)"); ax_sc.set_ylabel("MI (norm.)")
        ax_sc.set_title("MI vs KL — points above diagonal → non-Gaussian structure")
        ax_sc.legend(fontsize=8); plt.tight_layout(); plt.show()

    # ── 11. 2-D interaction plots ─────────────────────────────────────────────
    if n_sel >= 2:
        _MAX_SC  = 5000
        _rng_sub = np.random.default_rng(0)

        def _subsample(mask, n=_MAX_SC):
            idx = np.where(mask)[0]
            return _rng_sub.choice(idx, min(n, len(idx)), replace=False)

        pos_idx = _subsample(pos_mask)
        neg_idx = _subsample(neg_mask)

        # Full interaction matrix (scatter / density / diagonal)
        fig2d, axes2d = plt.subplots(n_sel, n_sel,
                                      figsize=(2.8 * n_sel, 2.8 * n_sel))
        axes2d = np.array(axes2d).reshape(n_sel, n_sel)

        for row in range(n_sel):
            for col in range(n_sel):
                ax2 = axes2d[row, col]
                bi_row, bi_col = sel[row], sel[col]

                if row == col:
                    vals = X[:, bi_col]
                    lo_d, hi_d = np.nanpercentile(vals, [1, 99])
                    bb = np.linspace(lo_d, hi_d, 50)
                    ax2.hist(X[neg_mask, bi_col], bins=bb, density=True,
                             alpha=0.40, color="#6a9fd8")
                    ax2.hist(X[pos_mask, bi_col], bins=bb, density=True,
                             alpha=0.60, color="#e8775a")
                    xr = np.linspace(lo_d, hi_d, 200)
                    ax2.plot(xr, norm.pdf(xr, mu1[col], std1[col]), "k-", lw=1.5)
                    ax2.set_title(fnames_sel[col], fontsize=8, pad=2)

                elif col > row:
                    ax2.scatter(X[neg_idx, bi_col], X[neg_idx, bi_row],
                                s=2, alpha=0.25, color="#6a9fd8", rasterized=True)
                    ax2.scatter(X[pos_idx, bi_col], X[pos_idx, bi_row],
                                s=4, alpha=0.45, color="#e8775a", rasterized=True)
                    for n_std, alpha_e in [(1, 0.55), (2, 0.30)]:
                        _draw_ellipse(ax2, mu1[[col, row]],
                                      cov1[np.ix_([col, row], [col, row])],
                                      n_std=n_std, edgecolor="#c0392b",
                                      facecolor="none", linewidth=1.2,
                                      alpha=alpha_e,
                                      linestyle="-" if n_std == 1 else "--")
                        _draw_ellipse(ax2, mu0[[col, row]],
                                      cov0[np.ix_([col, row], [col, row])],
                                      n_std=n_std, edgecolor="#2471a3",
                                      facecolor="none", linewidth=1.2,
                                      alpha=alpha_e,
                                      linestyle="-" if n_std == 1 else "--")
                else:
                    _xv = X[:, bi_col]; _yv = X[:, bi_row]
                    lo_x, hi_x = np.nanpercentile(_xv, [1, 99])
                    lo_y, hi_y = np.nanpercentile(_yv, [1, 99])
                    _nb = 30
                    _Hp, _xe, _ye = np.histogram2d(
                        X[pos_mask, bi_col], X[pos_mask, bi_row],
                        bins=_nb, range=[[lo_x, hi_x], [lo_y, hi_y]])
                    _Ha, _, _ = np.histogram2d(
                        _xv, _yv, bins=_nb, range=[[lo_x, hi_x], [lo_y, hi_y]])
                    with np.errstate(invalid="ignore", divide="ignore"):
                        _frac = np.where(_Ha > 0, _Hp / _Ha, np.nan)
                    _xc = 0.5 * (_xe[:-1] + _xe[1:])
                    _yc = 0.5 * (_ye[:-1] + _ye[1:])
                    _XX, _YY = np.meshgrid(_xc, _yc)
                    ax2.pcolormesh(_XX, _YY, _frac.T, cmap="RdBu_r",
                                   vmin=0, vmax=1, shading="auto", rasterized=True)

                ax2.tick_params(labelsize=6)
                if col == 0:
                    ax2.set_ylabel(fnames_sel[row], fontsize=7)
                if row == n_sel - 1:
                    ax2.set_xlabel(fnames_sel[col], fontsize=7)

        _pp = mpatches.Patch(color="#e8775a", alpha=0.7, label="Positive (1)")
        _pn = mpatches.Patch(color="#6a9fd8", alpha=0.7, label="Negative (0)")
        _pe = mpatches.Patch(edgecolor="#c0392b", facecolor="none",
                              linewidth=1.2, label="GG ellipse pos/neg")
        fig2d.legend(handles=[_pp, _pn, _pe], loc="upper right",
                     fontsize=8, framealpha=0.9)
        fig2d.suptitle(
            "2-D Band Interaction Matrix\n"
            "Upper: scatter + GG ellipses  |  Diagonal: 1-D marginals  |  "
            "Lower: positive-class fraction density",
            fontsize=9, y=1.005)
        plt.tight_layout(); plt.show()

    # ── 12. Pairwise 2-D KL matrix ───────────────────────────────────────────
    if n_sel >= 2:
        def kl_gaussian_nd(mu_p, cov_p, mu_q, cov_q):
            k     = len(mu_p)
            inv_q = np.linalg.inv(cov_q)
            _, ld_q = np.linalg.slogdet(cov_q)
            _, ld_p = np.linalg.slogdet(cov_p)
            diff  = mu_q - mu_p
            return 0.5 * (np.trace(inv_q @ cov_p)
                          + diff @ inv_q @ diff - k + ld_q - ld_p)

        kl2_pn = np.zeros((n_sel, n_sel))
        kl2_np = np.zeros((n_sel, n_sel))
        for i in range(n_sel):
            for j in range(n_sel):
                if i == j:
                    kl2_pn[i, j] = kl_per_band[i]
                    kl2_np[i, j] = kl_per_band[i]
                else:
                    ij = [i, j]
                    pp = X[pos_mask][:, [sel[i], sel[j]]]
                    nn = X[neg_mask][:, [sel[i], sel[j]]]
                    mp = pp.mean(0); mn = nn.mean(0)
                    cp = np.cov(pp, rowvar=False) + 1e-6 * np.eye(2)
                    cn = np.cov(nn, rowvar=False) + 1e-6 * np.eye(2)
                    kl2_pn[i, j] = kl_gaussian_nd(mp, cp, mn, cn)
                    kl2_np[i, j] = kl_gaussian_nd(mn, cn, mp, cp)

        fig, axes_kl = plt.subplots(1, 2, figsize=(14, 5.5))
        for ax_k, (mat, lbl) in enumerate([
            (kl2_pn, "KL( Positive ‖ Negative )"),
            (kl2_np, "KL( Negative ‖ Positive )"),
        ]):
            ax = axes_kl[ax_k]
            vmax_k = np.nanpercentile(mat, 97)
            im_k = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=vmax_k)
            plt.colorbar(im_k, ax=ax, fraction=0.046, label="KL (nats)")
            ax.set_xticks(range(n_sel)); ax.set_yticks(range(n_sel))
            ax.set_xticklabels(fnames_sel, rotation=40, ha="right", fontsize=8)
            ax.set_yticklabels(fnames_sel, fontsize=8)
            ax.set_title(f"Pairwise 2-D {lbl}\n"
                         f"(diagonal = 1-D KL; off-diagonal = bivariate KL)",
                         fontsize=9)
            for i in range(n_sel):
                for j in range(n_sel):
                    v = mat[i, j]
                    tc = "white" if v > 0.6 * vmax_k else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color=tc)
        fig.suptitle("Pairwise 2-D KL Divergence — Band Interaction Separability",
                     fontsize=9)
        plt.tight_layout(); plt.show()

        # Interaction gain
        gain_pn = np.zeros((n_sel, n_sel))
        gain_np = np.zeros((n_sel, n_sel))
        for i in range(n_sel):
            for j in range(n_sel):
                exp = 0.5 * (kl_per_band[i] + kl_per_band[j])
                gain_pn[i, j] = kl2_pn[i, j] - exp
                gain_np[i, j] = kl2_np[i, j] - exp

        fig, axes_gain = plt.subplots(1, 2, figsize=(14, 5.5))
        for ax_g, (mat_g, lbl_g) in enumerate([
            (gain_pn, "KL(+‖−)"),
            (gain_np, "KL(−‖+)"),
        ]):
            ax = axes_gain[ax_g]
            _vext = np.nanpercentile(np.abs(mat_g), 97)
            im_g = ax.imshow(mat_g, cmap="PuOr", vmin=-_vext, vmax=_vext)
            plt.colorbar(im_g, ax=ax, fraction=0.046, label="KL surplus (nats)")
            ax.set_xticks(range(n_sel)); ax.set_yticks(range(n_sel))
            ax.set_xticklabels(fnames_sel, rotation=40, ha="right", fontsize=8)
            ax.set_yticklabels(fnames_sel, fontsize=8)
            ax.set_title(f"Interaction Gain {lbl_g}\n"
                         f"= 2-D KL − mean(1-D KL)  "
                         f"(orange=synergistic, purple=redundant)",
                         fontsize=9)
            for i in range(n_sel):
                for j in range(n_sel):
                    v = mat_g[i, j]
                    tc = "white" if abs(v) > 0.55 * _vext else "black"
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                            fontsize=7, color=tc)
        fig.suptitle("Band Pair Interaction Gain\n"
                     "Orange = synergistic  |  Purple = redundant",
                     fontsize=9)
        plt.tight_layout(); plt.show()

    print(f"\nClassification report (threshold={model.best_threshold:.4f}):")
    print(classification_report(y, predictions, target_names=["Negative", "Positive"]))


def plot_apply_figures(X_apply: np.ndarray, probs: np.ndarray,
                       predictions: np.ndarray, maha: np.ndarray,
                       feature_names_apply: list[str],
                       model: "GeneralizedGaussianBinary",
                       x_size: int, y_size: int):
    """Spatial output figures for --apply mode."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(probs.reshape(x_size, y_size), cmap="hot", vmin=0, vmax=1)
    axes[0].set_title("Predicted Probability"); axes[0].axis("off")
    axes[1].imshow(predictions.reshape(x_size, y_size), cmap="gray")
    axes[1].set_title("Predicted Classification"); axes[1].axis("off")
    p95 = np.percentile(maha, 95)
    im3 = axes[2].imshow(maha.reshape(x_size, y_size), cmap="viridis_r",
                         vmin=0, vmax=p95)
    plt.colorbar(im3, ax=axes[2], fraction=0.046)
    axes[2].set_title("Mahalanobis Distance"); axes[2].axis("off")
    plt.suptitle("Generalized Gaussian — Apply Results")
    plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generalized Gaussian Binary Classifier for multi-band rasters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fit",   action="store_true",
                      help="Fit a new model on --image and --mask.")
    mode.add_argument("--apply", action="store_true",
                      help="Apply a saved model to --image.")

    p.add_argument("--image",  required=True,
                   help="Path to input raster (GDAL-readable).")
    p.add_argument("--mask",
                   help="Path to binary 0/1 classification mask (required for --fit).")
    p.add_argument("--model",  required=True,
                   help="Model file stem (no extension).  "
                        "Saves/loads <stem>.pkl and <stem>.txt.")
    p.add_argument("--beta",   type=float, default=1.5,
                   help="Generalized Gaussian shape parameter beta "
                        "(1=Laplacian, 2=Gaussian, default=1.5).")
    p.add_argument("--bands",  nargs="+", default=None,
                   help="Subset of band names to load from the raster "
                        "(default: all bands).")
    p.add_argument("--KL",     action="store_true",
                   help="(--fit only) Use greedy KL-divergence band selection "
                        "to find the most discriminative subset before fitting.")
    p.add_argument("--out_dir", default=".",
                   help="Directory for output rasters (--apply, default: .).")
    return p


def cmd_fit(args):
    if not args.mask:
        sys.exit("ERROR: --mask is required with --fit.")

    print(f"Reading image:  {args.image}")
    X, band_names, meta = read_raster(args.image)
    print(f"  Shape: {meta['rows']} x {meta['cols']}  |  {meta['n_bands']} bands")
    print(f"  Bands: {band_names}")

    # Optionally restrict to a user-specified band subset
    if args.bands:
        load_idx = resolve_band_indices(args.bands, band_names)
        print(f"  Using bands: {args.bands}")
    else:
        load_idx = list(range(meta["n_bands"]))

    X_use  = X[:, load_idx]
    names_use = [band_names[i] for i in load_idx]

    print(f"Reading mask:   {args.mask}")
    y = read_mask(args.mask, meta["cols"], meta["rows"])
    print(f"  Positive pixels: {y.sum()}  Negative: {(y == 0).sum()}")

    # Remove NaN rows (pixels where any band is nodata)
    valid = np.all(np.isfinite(X_use), axis=1) & np.isin(y, [0, 1])
    X_fit = X_use[valid]
    y_fit = y[valid]
    print(f"  Valid pixels for fitting: {valid.sum()}")

    # KL-based band selection
    if args.KL:
        print("\nRunning KL-based greedy band selection...")
        selected = select_bands_by_kl(X_fit, y_fit, names_use, verbose=True)
    else:
        selected = list(range(X_fit.shape[1]))

    # Fit
    model = GeneralizedGaussianBinary(beta=args.beta)
    model.fit(X_fit, y_fit, band_names=names_use, selected_bands=selected)
    model.save(args.model)

    print(f"\nFitted model:  beta={args.beta}  "
          f"threshold={model.best_threshold:.4f}")
    print(f"Selected bands: {[names_use[i] for i in selected]}")

    # Figures (use all valid pixels; full spatial arrays for maps)
    x_size, y_size = meta["rows"], meta["cols"]
    # For spatial plots we need full-length arrays (NaN for invalid pixels)
    X_full = X_use.copy()

    plot_fit_figures(
        X_full, y,
        model,
        feature_names=names_use,
        x_size=x_size, y_size=y_size,
        map_result=y.reshape(x_size, y_size),
    )


def cmd_apply(args):
    if args.KL:
        print("WARNING: --KL is only used during --fit; ignored in --apply mode.")

    print(f"Loading model:  {args.model}.pkl")
    model = GeneralizedGaussianBinary.load(args.model)

    required_names = [model.band_names[i] for i in model.selected_bands]
    print(f"Model requires bands: {required_names}")

    print(f"Reading image:  {args.image}")
    X, band_names, meta = read_raster(args.image)
    print(f"  Shape: {meta['rows']} x {meta['cols']}  |  {meta['n_bands']} bands")
    print(f"  Bands: {band_names}")

    # ── Band name validation (crash on missing) ───────────────────────────────
    # Build a full X matrix aligned to what the model expects.
    # The model's selected_bands index into model.band_names; we need to find
    # the same named bands in the new raster, in any order.
    all_required = list(model.band_names)   # all bands model was trained on
    apply_indices = resolve_band_indices(all_required, band_names)
    # apply_indices[i] = index in the new raster for model.band_names[i]

    # Re-order X so column i corresponds to model.band_names[i]
    X_reordered = X[:, apply_indices]

    print("Band mapping (model training order → new raster index):")
    for ti, (mname, ai) in enumerate(zip(model.band_names, apply_indices)):
        print(f"  [{ti:2d}] {mname:20s} → raster band {ai + 1} ({band_names[ai]})")

    # Apply
    probs       = model.predict_proba(X_reordered)[:, 1]
    predictions = model.predict(X_reordered)
    maha        = model.mahalanobis(X_reordered)

    # Write output rasters
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, os.path.basename(args.model))

    prob_path  = stem + "_prediction.tif"
    cls_path   = stem + "_classified.tif"

    write_raster(prob_path,  probs.astype(np.float32),
                 meta["cols"], meta["rows"],
                 meta["geotransform"], meta["projection"],
                 dtype=gdal.GDT_Float32)
    write_raster(cls_path, predictions.astype(np.uint8),
                 meta["cols"], meta["rows"],
                 meta["geotransform"], meta["projection"],
                 dtype=gdal.GDT_Byte)

    print(f"\nProbability map written: {prob_path}")
    print(f"Classification written:  {cls_path}")

    x_size, y_size = meta["rows"], meta["cols"]
    plot_apply_figures(X_reordered, probs, predictions, maha,
                       band_names, model, x_size, y_size)


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.fit:
        cmd_fit(args)
    elif args.apply:
        cmd_apply(args)


if __name__ == "__main__":
    main()

