/*
 * miniccdc_cuda.cu
 * ================
 * GPU-accelerated CCDC-like change detection for 4-band Sentinel-2
 * ENVI BSQ stacks (B12, B11, B9, B8 -- no green band required).
 *
 * Each CUDA thread processes one pixel's complete time series
 * independently, which maps perfectly onto the GPU's SIMT model.
 *
 * Algorithm (mirrors run_miniccdc.py):
 *   1. Drop NaN observations and check minimum span / count guards.
 *   2. Fit a robust harmonic regression (IRLS, Tukey bisquare weights)
 *      to the first min_history clean observations.
 *   3. Expand the stable window forward; refit every refit_interval
 *      accepted observations.
 *   4. CUSUM anomaly accumulator: declare a break when the accumulator
 *      exceeds cusum_thresh_k * chi2_crit.
 *   5. Record break ordinal date, count, RMS magnitude, per-band magnitude.
 *   6. Restart from break and continue.
 *
 * Build (L40s = sm_89):
 *   nvcc -O3 -arch=sm_89 -o miniccdc_cuda miniccdc_cuda.cu \
 *        -lgdal -I/usr/include/gdal
 *
 * Or with a Makefile target (see bottom of this file).
 *
 * Usage:
 *   ./miniccdc_cuda  --input_list files.txt  --output_dir ./ccdc/ \
 *                    --lines 1433  --samples 1491  --bands 4     \
 *                    --min_history 24  --consecutive 5            \
 *                    --alpha 0.99  --harmonics 2  --refit 8       \
 *                    --min_years 1.5  --min_hist_years 0.8
 *
 *   files.txt: one absolute path per line, already sorted by date.
 *              The companion utility sort_s2_files.py (below) produces this.
 *
 * Input data layout:
 *   Binary float32 ENVI BSQ: bands x lines x samples
 *   Band order in file: B12(idx0), B11(idx1), B9(idx2), B8(idx3)
 *
 * Output files (ENVI BSQ, float32 or int32, written via GDAL):
 *   tBreak_first.bin  -- int32  ordinal date of first break (0=none,-9999=nodata)
 *   tBreak_count.bin  -- int32  number of breaks            (-9999=nodata)
 *   MAG_rms.bin       -- float32 RMS magnitude at first break
 *   MAG_B12.bin       -- float32 signed magnitude B12
 *   MAG_B11.bin       -- float32 signed magnitude B11
 *   MAG_B9.bin        -- float32 signed magnitude B9
 *   MAG_B8.bin        -- float32 signed magnitude B8
 *
 * Memory notes for L40s (48 GB VRAM):
 *   800 files x 4 bands x 1491x1433 pixels x 4 bytes = ~27 GB
 *   All outputs together < 200 MB.
 *   The full stack fits comfortably in a single L40s.
 *
 * Dependencies:
 *   CUDA >= 12.0, GDAL >= 3.x
 *   Ubuntu: sudo apt install nvidia-cuda-toolkit libgdal-dev
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

/* GDAL C API */
#include "gdal.h"
#include "cpl_conv.h"
#include "ogr_srs_api.h"

/* =========================================================================
 * Compile-time constants
 * ========================================================================= */
#define MAX_BANDS    4
#define MAX_TIMES    1536   /* max time steps; supports ~1300 S2 scenes */
#define MAX_PRED     5      /* max design-matrix columns: 1 + 2*harmonics */
#define MAX_HISTORY  512    /* max initial history window (expanded to span min_hist_years) */
#define NODATA_INT   (-9999)
#define NODATA_FLOAT (NAN)

/* =========================================================================
 * Parameter struct (passed as a single __constant__ to device)
 * ========================================================================= */
struct Params {
    int   n_times;
    int   n_lines;
    int   n_samples;
    int   n_bands;
    int   min_history;
    int   harmonics;
    int   refit_interval;
    float alpha;
    float cusum_k;
    float cusum_thresh_k;
    float min_years;
    float min_hist_years;
    float chi2_crit;        /* precomputed on host */
    float omega;            /* 2*pi/365.25          */
    int   tile_n_pixels;    /* pixels in current tile (for tiling) */
};

__constant__ Params d_params;

/* =========================================================================
 * Device helpers
 * ========================================================================= */

/* Build one row of the harmonic design matrix into row[0..n_pred-1].
   n_pred = 1 + 2*harmonics */
__device__ __forceinline__
void design_row(float t, int harmonics, float omega, float *row)
{
    row[0] = 1.0f;
    for (int k = 1; k <= harmonics; ++k) {
        row[2*k-1] = sinf(k * omega * t);
        row[2*k  ] = cosf(k * omega * t);
    }
}

/* Dot product of two length-n vectors */
__device__ __forceinline__
float dot(const float *a, const float *b, int n)
{
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

/* -------------------------------------------------------------------------
 * Tiny least-squares solver for small systems.
 * Solves X^T W X coef = X^T W Y  using normal equations + Cholesky.
 * X  : (n_obs x n_pred) stored row-major in local array
 * Y  : (n_obs x n_bands)
 * W  : (n_obs) weight vector (NULL => uniform weights)
 * coef: (n_pred x n_bands) output, row-major
 * Returns 0 on success, -1 if singular.
 * -------------------------------------------------------------------------*/
__device__
int wlsq(const float *X, const float *Y, const float *W,
          int n_obs, int n_pred, int n_bands,
          float *coef)
{
    /* XtWX (n_pred x n_pred) and XtWY (n_pred x n_bands) */
    float A[MAX_PRED * MAX_PRED] = {0};
    float B[MAX_PRED * MAX_BANDS] = {0};

    for (int i = 0; i < n_obs; ++i) {
        float w = W ? W[i] : 1.0f;
        const float *xi = X + i * n_pred;
        for (int r = 0; r < n_pred; ++r) {
            for (int c = r; c < n_pred; ++c)
                A[r*n_pred + c] += w * xi[r] * xi[c];
            for (int b = 0; b < n_bands; ++b)
                B[r*n_bands + b] += w * xi[r] * Y[i*n_bands + b];
        }
    }
    /* symmetrise */
    for (int r = 0; r < n_pred; ++r)
        for (int c = r+1; c < n_pred; ++c)
            A[c*n_pred + r] = A[r*n_pred + c];

    /* Cholesky decomposition of A in-place */
    for (int i = 0; i < n_pred; ++i) {
        for (int j = 0; j <= i; ++j) {
            float s = A[i*n_pred + j];
            for (int k = 0; k < j; ++k)
                s -= A[i*n_pred + k] * A[j*n_pred + k];
            if (i == j) {
                if (s <= 0.0f) return -1;  /* not positive definite */
                A[i*n_pred + i] = sqrtf(s);
            } else {
                A[i*n_pred + j] = s / A[j*n_pred + j];
            }
        }
    }

    /* Forward / backward substitution for each RHS band */
    for (int b = 0; b < n_bands; ++b) {
        float y[MAX_PRED], x[MAX_PRED];
        for (int i = 0; i < n_pred; ++i) y[i] = B[i*n_bands + b];

        /* Forward substitution: L y = rhs */
        for (int i = 0; i < n_pred; ++i) {
            float s = y[i];
            for (int k = 0; k < i; ++k) s -= A[i*n_pred+k] * y[k];
            y[i] = s / A[i*n_pred+i];
        }
        /* Backward substitution: L^T x = y */
        for (int i = n_pred-1; i >= 0; --i) {
            float s = y[i];
            for (int k = i+1; k < n_pred; ++k) s -= A[k*n_pred+i] * x[k];
            x[i] = s / A[i*n_pred+i];
        }
        for (int i = 0; i < n_pred; ++i)
            coef[i*n_bands + b] = x[i];
    }
    return 0;
}

/* -------------------------------------------------------------------------
 * Robust IRLS fit (Tukey bisquare, 5 iterations).
 * X_rows: scratch (n_obs x n_pred), pre-filled with design rows.
 * Y     : (n_obs x n_bands)
 * coef  : (n_pred x n_bands) output
 * Returns 0 on success.
 * -------------------------------------------------------------------------*/
__device__
int fit_robust(const float *X_rows, const float *Y,
               int n_obs, int n_pred, int n_bands,
               float *coef)
{
    float W[MAX_TIMES];   /* MAX_TIMES: called from refit with up to n_stable obs */
    for (int i = 0; i < n_obs; ++i) W[i] = 1.0f;

    for (int iter = 0; iter < 5; ++iter) {
        if (wlsq(X_rows, Y, W, n_obs, n_pred, n_bands, coef) != 0)
            return -1;

        /* compute per-observation RMS residual */
        float rms[MAX_TIMES];
        for (int i = 0; i < n_obs; ++i) {
            float s = 0.0f;
            for (int b = 0; b < n_bands; ++b) {
                float pred_ib = 0.0f;
                for (int p = 0; p < n_pred; ++p)
                    pred_ib += X_rows[i*n_pred+p] * coef[p*n_bands+b];
                float r = Y[i*n_bands+b] - pred_ib;
                s += r * r;
            }
            rms[i] = sqrtf(s / n_bands);
        }

        /* MAD of rms values */
        /* simple insertion sort for small n */
        float sorted[MAX_TIMES];
        for (int i = 0; i < n_obs; ++i) sorted[i] = rms[i];
        for (int i = 1; i < n_obs; ++i) {
            float v = sorted[i]; int j = i-1;
            while (j >= 0 && sorted[j] > v) { sorted[j+1] = sorted[j]; --j; }
            sorted[j+1] = v;
        }
        float med = (n_obs % 2) ? sorted[n_obs/2]
                                 : 0.5f*(sorted[n_obs/2-1]+sorted[n_obs/2]);
        float devs[MAX_TIMES];
        for (int i = 0; i < n_obs; ++i)
            devs[i] = fabsf(rms[i] - med);
        for (int i = 1; i < n_obs; ++i) {
            float v = devs[i]; int j = i-1;
            while (j >= 0 && devs[j] > v) { devs[j+1] = devs[j]; --j; }
            devs[j+1] = v;
        }
        float mad = (n_obs % 2) ? devs[n_obs/2]
                                 : 0.5f*(devs[n_obs/2-1]+devs[n_obs/2]);
        float sigma = fmaxf(mad * 1.4826f, 1e-6f);
        float cutoff = 2.5f * sigma;

        for (int i = 0; i < n_obs; ++i) {
            float u = rms[i] / cutoff;
            float w = (u < 1.0f) ? (1.0f - u*u) * (1.0f - u*u) : 0.0f;
            W[i] = fmaxf(w, 1e-3f);
        }
    }
    return 0;
}

/* -------------------------------------------------------------------------
 * Compute inverse of a symmetric positive-definite (n x n) matrix.
 * Uses Cholesky. n <= MAX_BANDS = 4.
 * Returns 0 on success, -1 if singular.
 * -------------------------------------------------------------------------*/
__device__
int sym_inv(const float *A, float *Ainv, int n)
{
    float L[MAX_BANDS * MAX_BANDS] = {0};

    /* Cholesky */
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            float s = A[i*n+j];
            for (int k = 0; k < j; ++k) s -= L[i*n+k]*L[j*n+k];
            if (i == j) {
                if (s <= 0.0f) return -1;
                L[i*n+i] = sqrtf(s);
            } else {
                L[i*n+j] = s / L[j*n+j];
            }
        }
    }

    /* Invert via solving L L^T x = e_i for each column */
    for (int col = 0; col < n; ++col) {
        float y[MAX_BANDS] = {0}, x[MAX_BANDS] = {0};
        y[col] = 1.0f;
        for (int i = 0; i < n; ++i) {
            float s = y[i];
            for (int k = 0; k < i; ++k) s -= L[i*n+k]*y[k];
            y[i] = s / L[i*n+i];
        }
        for (int i = n-1; i >= 0; --i) {
            float s = y[i];
            for (int k = i+1; k < n; ++k) s -= L[k*n+i]*x[k];
            x[i] = s / L[i*n+i];
        }
        for (int r = 0; r < n; ++r) Ainv[r*n+col] = x[r];
    }
    return 0;
}

/* -------------------------------------------------------------------------
 * Compute trimmed covariance of residuals.
 * resid : (n_obs x n_bands)
 * cov   : (n_bands x n_bands) output
 * -------------------------------------------------------------------------*/
__device__
void trimmed_cov(const float *resid, int n_obs, int n_bands, float *cov)
{
    /* norms */
    /* norms and idx need to handle up to MAX_TIMES entries (refit path) */
    float norms[MAX_TIMES];
    for (int i = 0; i < n_obs; ++i) {
        float s = 0.0f;
        for (int b = 0; b < n_bands; ++b)
            s += resid[i*n_bands+b] * resid[i*n_bands+b];
        norms[i] = sqrtf(s);
    }
    /* argsort by norm (insertion) */
    int idx[MAX_TIMES];
    for (int i = 0; i < n_obs; ++i) idx[i] = i;
    for (int i = 1; i < n_obs; ++i) {
        int v = idx[i]; int j = i-1;
        while (j >= 0 && norms[idx[j]] > norms[v]) { idx[j+1]=idx[j]; --j; }
        idx[j+1] = v;
    }
    int lo = (int)(n_obs * 0.15f);
    int hi = (int)(n_obs * 0.85f);
    if (hi <= lo + 2) hi = lo + 3;
    int m = hi - lo;

    /* mean of trimmed set */
    float mu[MAX_BANDS] = {0};
    for (int i = lo; i < hi; ++i)
        for (int b = 0; b < n_bands; ++b)
            mu[b] += resid[idx[i]*n_bands+b];
    for (int b = 0; b < n_bands; ++b) mu[b] /= m;

    /* covariance */
    for (int r = 0; r < n_bands; ++r)
        for (int c = 0; c < n_bands; ++c)
            cov[r*n_bands+c] = 0.0f;

    for (int i = lo; i < hi; ++i) {
        for (int r = 0; r < n_bands; ++r) {
            float dr = resid[idx[i]*n_bands+r] - mu[r];
            for (int c = r; c < n_bands; ++c) {
                float dc = resid[idx[i]*n_bands+c] - mu[c];
                cov[r*n_bands+c] += dr * dc;
            }
        }
    }
    float norm = (m > 1) ? 1.0f/(m-1) : 1.0f;
    for (int r = 0; r < n_bands; ++r) {
        cov[r*n_bands+r] = cov[r*n_bands+r]*norm + 1e-6f;
        for (int c = r+1; c < n_bands; ++c) {
            cov[r*n_bands+c] *= norm;
            cov[c*n_bands+r]  = cov[r*n_bands+c];
        }
    }
}

/* =========================================================================
 * Convert Python ordinal date to year and month.
 * Works for both __device__ (kernel) and host (diagnostic) code.
 * Python ordinal: 1 = Jan 1, year 1 (proleptic Gregorian).
 * =========================================================================*/
__device__ __host__
void ordinal_to_ym(int ord, int *out_year, int *out_month)
{
    int d0   = ord - 1;
    int n400 = d0 / 146097;
    int d1   = d0 % 146097;
    int n100 = d1 / 36524;
    if (n100 == 4) n100 = 3;
    int d2   = d1 - n100 * 36524;
    int n4   = d2 / 1461;
    int d3   = d2 % 1461;
    int n1   = d3 / 365;
    if (n1 == 4) n1 = 3;

    int year = n400*400 + n100*100 + n4*4 + n1 + 1;
    int doy  = d3 - n1*365;

    int leap = (year%4==0 && (year%100!=0 || year%400==0)) ? 1 : 0;
    int mdays[] = {31, 28+leap, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    int month = 1;
    for (int m = 0; m < 12; ++m) {
        if (doy < mdays[m]) { month = m + 1; break; }
        doy -= mdays[m];
    }
    *out_year  = year;
    *out_month = month;
}

/* =========================================================================
 * Main kernel: one thread per pixel
 * =========================================================================
 * stack_gpu : float32, shape (n_times, n_bands, n_pixels), C-order
 * dates_gpu : float32, ordinal days, shape (n_times,)
 * out_tbreak, out_count : int32, shape (n_pixels,)
 * out_magrms, out_magb  : float32
 * =========================================================================*/
__global__
void ccdc_kernel(const float * __restrict__ stack_gpu,
                 const float * __restrict__ dates_gpu,
                 int32_t      *out_tbreak,
                 int32_t      *out_count,
                 float        *out_magrms,
                 float        *out_magb)   /* n_bands * n_pixels, band-major */
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    const Params &p = d_params;
    int n_pixels = p.tile_n_pixels;
    if (pid >= n_pixels) return;

    int n_bands  = p.n_bands;
    int n_times  = p.n_times;
    int n_pred   = 1 + 2 * p.harmonics;

    /* ---- 1. Collect valid (non-NaN) observations for this pixel ---- */
    float t_cl[MAX_TIMES];
    float Y_cl[MAX_TIMES * MAX_BANDS];
    int   n_valid = 0;

    for (int ti = 0; ti < n_times; ++ti) {
        /* stack layout: (n_times, n_bands, n_pixels) */
        float vals[MAX_BANDS];
        bool any_nan = false;
        for (int b = 0; b < n_bands; ++b) {
            float v = stack_gpu[(ti * n_bands + b) * n_pixels + pid];
            vals[b] = v;
            if (isnan(v)) { any_nan = true; break; }
        }
        if (!any_nan) {
            t_cl[n_valid] = dates_gpu[ti];
            for (int b = 0; b < n_bands; ++b)
                Y_cl[n_valid * n_bands + b] = vals[b];
            ++n_valid;
        }
    }

    /* ---- 2. Center time values for numerical stability ----
     * Raw ordinal dates (~738,000) cause sinf(k*omega*t) to operate
     * at ~12,700 radians where float32 argument reduction loses all
     * precision.  Subtracting t_base brings the range to [0, ~2200]
     * (~38 radians max) which sinf handles accurately.
     * The harmonic model is periodic so this is just a phase shift. */
    float t_base = (n_valid > 0) ? t_cl[0] : 0.0f;
    for (int i = 0; i < n_valid; ++i)
        t_cl[i] -= t_base;

    /* ---- 2b. Remove consecutive duplicate observations ----
     * Carry-forward compositing creates long runs of identical values
     * at different dates.  CCDC interprets the transition between
     * runs as a spectral break.  Deduplication keeps only the FIRST
     * observation in each run of identical values, preserving the
     * true change timestamps while eliminating artificial inflation
     * of the observation count.
     *
     * This is safe for all data — if the input is actual per-scene
     * observations with no carry-forward, consecutive duplicates are
     * extremely rare and dedup is essentially a no-op. */
    {
        int out = 0;
        for (int i = 0; i < n_valid; ++i) {
            if (out == 0) {
                /* always keep the first observation */
                t_cl[out]  = t_cl[i];
                for (int b = 0; b < n_bands; ++b)
                    Y_cl[out*n_bands+b] = Y_cl[i*n_bands+b];
                out++;
            } else {
                /* check if ANY band value differs from the last kept obs */
                bool changed = false;
                for (int b = 0; b < n_bands; ++b) {
                    if (Y_cl[i*n_bands+b] != Y_cl[(out-1)*n_bands+b]) {
                        changed = true;
                        break;
                    }
                }
                if (changed) {
                    t_cl[out]  = t_cl[i];
                    for (int b = 0; b < n_bands; ++b)
                        Y_cl[out*n_bands+b] = Y_cl[i*n_bands+b];
                    out++;
                }
            }
        }
        n_valid = out;
    }

    /* ---- 2c. Filter out winter months (Nov-Mar inclusive) ----
     * Low sun angles, snow, ice, and shadow dominate winter imagery
     * at high latitudes and inject noise that harmonic models cannot
     * separate from real land-cover change. */
    {
        int out = 0;
        for (int i = 0; i < n_valid; ++i) {
            int ord = (int)(t_cl[i] + t_base);
            int y, m;
            ordinal_to_ym(ord, &y, &m);
            if (m >= 4 && m <= 10) {   /* keep April through October only */
                t_cl[out] = t_cl[i];
                for (int b = 0; b < n_bands; ++b)
                    Y_cl[out*n_bands+b] = Y_cl[i*n_bands+b];
                out++;
            }
        }
        n_valid = out;
    }

    /* ---- 2d. Compute monthly medians ----
     * Collapse all observations within each calendar month into a
     * single representative value (median per band, median date).
     * This suppresses within-month noise (residual clouds, BRDF
     * effects, atmospheric variation) and produces a clean seasonal
     * signal that the harmonic model can fit reliably.
     * Observations are already sorted chronologically, so month
     * groups are contiguous. */
    {
        int n_monthly = 0;
        int grp = 0;
        while (grp < n_valid) {
            /* identify this month's group */
            int ord0 = (int)(t_cl[grp] + t_base);
            int y0, m0;
            ordinal_to_ym(ord0, &y0, &m0);
            int ym0 = y0 * 100 + m0;

            int grp_end = grp + 1;
            while (grp_end < n_valid) {
                int oe = (int)(t_cl[grp_end] + t_base);
                int ye, me;
                ordinal_to_ym(oe, &ye, &me);
                if (ye*100 + me != ym0) break;
                grp_end++;
            }
            int gsz = grp_end - grp;

            /* median date (series is sorted, so just pick middle) */
            float med_t;
            if (gsz % 2 == 1)
                med_t = t_cl[grp + gsz/2];
            else
                med_t = (t_cl[grp + gsz/2 - 1] + t_cl[grp + gsz/2]) * 0.5f;

            /* median per band via insertion sort on small buffer */
            float med_b[MAX_BANDS];
            for (int b = 0; b < n_bands; ++b) {
                float sv[64];
                int nv = 0;
                for (int i = grp; i < grp_end && nv < 64; ++i)
                    sv[nv++] = Y_cl[i*n_bands+b];
                /* insertion sort */
                for (int i = 1; i < nv; ++i) {
                    float v = sv[i]; int j = i-1;
                    while (j >= 0 && sv[j] > v) { sv[j+1]=sv[j]; --j; }
                    sv[j+1] = v;
                }
                med_b[b] = (nv % 2 == 1) ? sv[nv/2]
                                           : (sv[nv/2-1]+sv[nv/2])*0.5f;
            }

            t_cl[n_monthly] = med_t;
            for (int b = 0; b < n_bands; ++b)
                Y_cl[n_monthly*n_bands+b] = med_b[b];
            n_monthly++;

            grp = grp_end;
        }
        n_valid = n_monthly;
    }

    /* ---- 3. Guard checks ---- */
    int nodata_tbreak = NODATA_INT, nodata_count = NODATA_INT;
    auto set_nodata = [&]() {
        out_tbreak[pid] = nodata_tbreak;
        out_count[pid]  = nodata_count;
        out_magrms[pid] = NODATA_FLOAT;
        for (int b = 0; b < n_bands; ++b)
            out_magb[b * n_pixels + pid] = NODATA_FLOAT;
    };

    if (n_valid < p.min_history + 3) { set_nodata(); return; }

    float t_span_years = (t_cl[n_valid-1] - t_cl[0]) / 365.25f;
    if (t_span_years < p.min_years) { set_nodata(); return; }

    /* ---- 3. Segment loop ---- */
    int   seg_start   = 0;
    int   break_count = 0;
    int32_t first_break_date  = 0;
    float   first_break_magrms = 0.0f;
    float   first_break_magb[MAX_BANDS] = {0};

    while (1) {
        int remaining = n_valid - seg_start;
        if (remaining < p.min_history + 3) break;

        int hist_end = seg_start + p.min_history;

        /* Expand history window until it spans min_hist_years.
         * With S2 revisit every 2-3 days, min_history=24 only spans
         * ~70 days.  We need ~300 days (0.8 yr) for a stable 2-harmonic
         * fit, so expand to include enough observations. */
        while (hist_end < n_valid &&
               hist_end - seg_start < MAX_HISTORY &&
               (t_cl[hist_end-1] - t_cl[seg_start]) / 365.25f < p.min_hist_years) {
            hist_end++;
        }

        /* still too short after expansion, or not enough monitoring obs left */
        if ((t_cl[hist_end-1] - t_cl[seg_start]) / 365.25f < p.min_hist_years) break;
        if (n_valid - hist_end < 3) break;

        /* build design matrix for history window */
        float X_hist[MAX_HISTORY * MAX_PRED];
        for (int i = seg_start; i < hist_end; ++i)
            design_row(t_cl[i], p.harmonics, p.omega,
                       X_hist + (i-seg_start)*n_pred);

        float coef[MAX_PRED * MAX_BANDS];
        if (fit_robust(X_hist, Y_cl + seg_start*n_bands,
                       hist_end - seg_start, n_pred, n_bands, coef) != 0)
            break;

        /* compute residuals for history window */
        float resid_hist[MAX_HISTORY * MAX_BANDS];
        for (int i = seg_start; i < hist_end; ++i) {
            float xrow[MAX_PRED];
            design_row(t_cl[i], p.harmonics, p.omega, xrow);
            for (int b = 0; b < n_bands; ++b) {
                float pred = 0.0f;
                for (int pp = 0; pp < n_pred; ++pp)
                    pred += xrow[pp] * coef[pp*n_bands+b];
                resid_hist[(i-seg_start)*n_bands+b] =
                    Y_cl[i*n_bands+b] - pred;
            }
        }

        float cov[MAX_BANDS * MAX_BANDS];
        trimmed_cov(resid_hist, hist_end - seg_start, n_bands, cov);

        float cov_inv[MAX_BANDS * MAX_BANDS];
        if (sym_inv(cov, cov_inv, n_bands) != 0) break;

        /* stable window bookkeeping */
        /* We track the indices of stable obs in a small scratch array */
        int stable_buf[MAX_TIMES];
        int n_stable = 0;
        for (int i = seg_start; i < hist_end; ++i)
            stable_buf[n_stable++] = i;

        int   n_since_refit = 0;
        float cusum         = 0.0f;
        int   consec_start  = -1;
        int   break_idx     = -1;

        for (int i = hist_end; i < n_valid; ++i) {
            float xrow[MAX_PRED];
            design_row(t_cl[i], p.harmonics, p.omega, xrow);

            float r[MAX_BANDS];
            for (int b = 0; b < n_bands; ++b) {
                float pred = 0.0f;
                for (int pp = 0; pp < n_pred; ++pp)
                    pred += xrow[pp] * coef[pp*n_bands+b];
                r[b] = Y_cl[i*n_bands+b] - pred;
            }

            /* Mahalanobis distance squared */
            float md2 = 0.0f;
            for (int rb = 0; rb < n_bands; ++rb) {
                float tmp = 0.0f;
                for (int cb = 0; cb < n_bands; ++cb)
                    tmp += cov_inv[rb*n_bands+cb] * r[cb];
                md2 += r[rb] * tmp;
            }

            if (md2 <= p.chi2_crit) {
                /* inlier */
                if (n_stable < MAX_TIMES) stable_buf[n_stable++] = i;
                cusum = fmaxf(0.0f, cusum - p.cusum_k * p.chi2_crit);
                consec_start = -1;
                ++n_since_refit;

                if (n_since_refit >= p.refit_interval && n_stable >= p.min_history) {
                    /* refit on growing stable window */
                    float X_st[MAX_TIMES * MAX_PRED];
                    float Y_st[MAX_TIMES * MAX_BANDS];
                    for (int si = 0; si < n_stable; ++si) {
                        int gi = stable_buf[si];
                        design_row(t_cl[gi], p.harmonics, p.omega,
                                   X_st + si * n_pred);
                        for (int b = 0; b < n_bands; ++b)
                            Y_st[si*n_bands+b] = Y_cl[gi*n_bands+b];
                    }
                    float new_coef[MAX_PRED * MAX_BANDS];
                    if (fit_robust(X_st, Y_st, n_stable, n_pred, n_bands, new_coef) == 0) {
                        /* recompute residuals and covariance */
                        float resid_st[MAX_TIMES * MAX_BANDS];
                        for (int si = 0; si < n_stable; ++si) {
                            for (int b = 0; b < n_bands; ++b) {
                                float pred = 0.0f;
                                for (int pp = 0; pp < n_pred; ++pp)
                                    pred += X_st[si*n_pred+pp]*new_coef[pp*n_bands+b];
                                resid_st[si*n_bands+b] = Y_st[si*n_bands+b] - pred;
                            }
                        }
                        float new_cov[MAX_BANDS*MAX_BANDS];
                        trimmed_cov(resid_st, n_stable, n_bands, new_cov);
                        float new_inv[MAX_BANDS*MAX_BANDS];
                        if (sym_inv(new_cov, new_inv, n_bands) == 0) {
                            for (int k=0; k<n_pred*n_bands; ++k) coef[k]=new_coef[k];
                            for (int k=0; k<n_bands*n_bands; ++k) cov_inv[k]=new_inv[k];
                        }
                    }
                    n_since_refit = 0;
                }
            } else {
                /* anomaly */
                cusum += (md2 - p.chi2_crit);
                if (consec_start < 0) consec_start = i;

                if (cusum >= p.cusum_thresh_k * p.chi2_crit) {
                    break_idx = consec_start;
                    break;
                }
            }
        } /* monitoring loop */

        if (break_idx < 0) break;  /* no more breaks */

        /* ---- compute magnitude at break epoch ---- */
        int epoch_end = break_idx + 8;
        if (epoch_end > n_valid) epoch_end = n_valid;
        int ep_len = epoch_end - break_idx;

        float mag_b[MAX_BANDS] = {0};
        for (int i = break_idx; i < epoch_end; ++i) {
            float xrow[MAX_PRED];
            design_row(t_cl[i], p.harmonics, p.omega, xrow);
            for (int b = 0; b < n_bands; ++b) {
                float pred = 0.0f;
                for (int pp = 0; pp < n_pred; ++pp)
                    pred += xrow[pp] * coef[pp*n_bands+b];
                mag_b[b] += (Y_cl[i*n_bands+b] - pred);
            }
        }
        float magrms = 0.0f;
        for (int b = 0; b < n_bands; ++b) {
            mag_b[b] /= ep_len;
            magrms += mag_b[b] * mag_b[b];
        }
        magrms = sqrtf(magrms / n_bands);

        if (break_count == 0) {
            first_break_date  = (int32_t)(t_cl[break_idx] + t_base);
            first_break_magrms = magrms;
            for (int b = 0; b < n_bands; ++b)
                first_break_magb[b] = mag_b[b];
        }
        ++break_count;
        seg_start = break_idx;
    } /* segment loop */

    /* ---- 4. Write outputs ---- */
    if (break_count == 0) {
        out_tbreak[pid] = 0;
        out_count[pid]  = 0;
        out_magrms[pid] = 0.0f;
        for (int b = 0; b < n_bands; ++b)
            out_magb[b * n_pixels + pid] = 0.0f;
    } else {
        out_tbreak[pid] = first_break_date;
        out_count[pid]  = break_count;
        out_magrms[pid] = first_break_magrms;
        for (int b = 0; b < n_bands; ++b)
            out_magb[b * n_pixels + pid] = first_break_magb[b];
    }
}

/* =========================================================================
 * Chi-squared PPF approximation (Wilson-Hilferty, good to ~0.1% for df<=8)
 * =========================================================================*/
static float chi2_ppf(float alpha, int df)
{
    /* Normal quantile via Beasley-Springer-Moro approximation */
    static const float a[] = {2.515517f, 0.802853f, 0.010328f};
    static const float b[] = {1.432788f, 0.189269f, 0.001308f};
    float p = (alpha > 0.5f) ? (1.0f - alpha) : alpha;
    float t2 = sqrtf(-2.0f * logf(p));
    float zp = t2 - (a[0]+t2*(a[1]+t2*a[2])) /
                     (1.0f+t2*(b[0]+t2*(b[1]+t2*b[2])));
    if (alpha <= 0.5f) zp = -zp;

    float d = (float)df;
    float mu = 1.0f - 2.0f/(9.0f*d);
    float sigma = sqrtf(2.0f/(9.0f*d));
    float x = mu + sigma * zp;
    return d * x * x * x;
}

/* =========================================================================
 * GDAL output helpers
 * =========================================================================*/
#ifdef __CUDACC__
#pragma diag_suppress 1650
#endif
static void write_envi_band(const char *path, void *data,
                             int cols, int rows, GDALDataType dtype,
                             double *geotransform, const char *wkt,
                             double nodata, const char *desc)
{
    GDALDriverH drv = GDALGetDriverByName("ENVI");
    char *opts[] = { (char*)"INTERLEAVE=BSQ", NULL };
    GDALDatasetH ds = GDALCreate(drv, path, cols, rows, 1, dtype, opts);
    if (!ds) { fprintf(stderr, "GDAL create failed: %s\n", path); return; }
    GDALSetGeoTransform(ds, geotransform);
    if (wkt && wkt[0]) {
        OGRSpatialReferenceH srs = OSRNewSpatialReference(NULL);
        /* OSRImportFromWkt takes char** (non-const) in the GDAL C API.
           Copy into a local mutable buffer to avoid the const violation. */
        char wkt_copy[4096];
        strncpy(wkt_copy, wkt, sizeof(wkt_copy) - 1);
        wkt_copy[sizeof(wkt_copy) - 1] = '\0';
        char *wkt_ptr = wkt_copy;
        OSRImportFromWkt(srs, &wkt_ptr);
        char *proj = NULL;
        OSRExportToWkt(srs, &proj);
        GDALSetProjection(ds, proj);
        CPLFree(proj);
        OSRDestroySpatialReference(srs);
    }
    GDALRasterBandH band = GDALGetRasterBand(ds, 1);
    GDALSetRasterNoDataValue(band, nodata);
    GDALSetDescription(band, desc);
    if (GDALRasterIO(band, GF_Write, 0, 0, cols, rows,
                     data, cols, rows, dtype, 0, 0) != CE_None)
        fprintf(stderr, "Warning: GDALRasterIO write failed for %s\n", path);
    GDALFlushCache(ds);
    GDALClose(ds);
}

/* =========================================================================
 * ENVI raw binary reader (float32, BSQ, no GDAL dependency for input)
 * =========================================================================*/
static float* read_envi_float(const char *path, long n_elem)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }
    float *buf = (float*)malloc(n_elem * sizeof(float));
    if (!buf) { fclose(f); return NULL; }
    size_t nr = fread(buf, sizeof(float), n_elem, f);
    fclose(f);
    if ((long)nr != n_elem) {
        fprintf(stderr, "Short read %s: got %zu expected %ld\n", path, nr, n_elem);
        free(buf); return NULL;
    }
    return buf;
}

/* =========================================================================
 * Argument parsing
 * =========================================================================*/
struct Args {
    char  input_list[512];
    char  output_dir[512];
    int   lines, samples, bands;
    int   min_history;
    int   harmonics;
    int   refit_interval;
    float alpha;
    float cusum_k;
    float cusum_thresh_k;
    float min_years;
    float min_hist_years;
    int   threads_per_block;
};

static void parse_args(int argc, char **argv, Args &a)
{
    /* defaults */
    strcpy(a.input_list, "files.txt");
    strcpy(a.output_dir, "./ccdc/");
    a.lines=1433; a.samples=1491; a.bands=4;
    a.min_history=24; a.harmonics=2; a.refit_interval=8;
    a.alpha=0.99f; a.cusum_k=0.5f; a.cusum_thresh_k=5.0f;
    a.min_years=1.5f; a.min_hist_years=0.8f;
    a.threads_per_block=256;

    for (int i=1; i<argc; ++i) {
#define SARG(flag, field) if(!strcmp(argv[i],flag)&&i+1<argc) { strncpy(a.field,argv[++i],511); continue; }
#define IARG(flag, field) if(!strcmp(argv[i],flag)&&i+1<argc) { a.field=atoi(argv[++i]); continue; }
#define FARG(flag, field) if(!strcmp(argv[i],flag)&&i+1<argc) { a.field=(float)atof(argv[++i]); continue; }
        SARG("--input_list",     input_list)
        SARG("--output_dir",     output_dir)
        IARG("--lines",          lines)
        IARG("--samples",        samples)
        IARG("--bands",          bands)
        IARG("--min_history",    min_history)
        IARG("--harmonics",      harmonics)
        IARG("--refit",          refit_interval)
        IARG("--threads",        threads_per_block)
        FARG("--alpha",          alpha)
        FARG("--cusum_k",        cusum_k)
        FARG("--consecutive",    cusum_thresh_k)
        FARG("--min_years",      min_years)
        FARG("--min_hist_years", min_hist_years)
#undef SARG
#undef IARG
#undef FARG
    }
}

/* =========================================================================
 * Detect B9 (945nm water vapor) band index from ENVI header.
 * Returns 0-based index of B9 band, or -1 if not found.
 * Looks for "B9 " or "945" in the "band names" header field.
 * Will NOT match "B8A" / "B9A" / other similar names.
 * =========================================================================*/
static int detect_b9_band(const char *bin_path)
{
    char hdr_path[512];
    strncpy(hdr_path, bin_path, 507);
    hdr_path[507] = '\0';
    char *dot = strrchr(hdr_path, '.');
    if (dot) strcpy(dot, ".hdr");
    else     strcat(hdr_path, ".hdr");

    FILE *f = fopen(hdr_path, "r");
    if (!f) {
        snprintf(hdr_path, sizeof(hdr_path), "%s.hdr", bin_path);
        f = fopen(hdr_path, "r");
    }
    if (!f) return -1;

    /* read entire header into buffer */
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char*)malloc(fsize + 1);
    if (!buf) { fclose(f); return -1; }
    size_t nr = fread(buf, 1, fsize, f);
    buf[nr] = '\0';
    fclose(f);

    /* find "band names" section */
    char *bn = strstr(buf, "band names");
    if (!bn) { free(buf); return -1; }
    char *open = strchr(bn, '{');
    if (!open) { free(buf); return -1; }
    char *close = strchr(open, '}');
    if (!close) { free(buf); return -1; }

    /* iterate through comma-separated band entries */
    int band_idx = 0;
    char *p = open + 1;
    while (p < close) {
        char *comma = (char*)memchr(p, ',', close - p);
        char *end = comma ? comma : close;

        /* extract this entry */
        int len = (int)(end - p);
        if (len > 255) len = 255;
        char entry[256];
        memcpy(entry, p, len);
        entry[len] = '\0';

        /* check for B9 specifically (not B9A, B19, etc.)
         * look for ": B9 " or " B9 " or "B9\t" patterns,
         * OR the wavelength "945" */
        char *hit = strstr(entry, "B9");
        if (hit) {
            char after = hit[2];
            if (after == ' ' || after == '\t' || after == '\n' ||
                after == '\r' || after == '\0' || after == ',') {
                free(buf);
                return band_idx;
            }
        }
        if (strstr(entry, "945")) {
            free(buf);
            return band_idx;
        }

        band_idx++;
        p = end + 1;
    }

    free(buf);
    return -1;
}

/* =========================================================================
 * Host-side diagnostic: sample a few pixels and trace the algorithm
 * =========================================================================*/
static void run_diagnostics(const float *stack_host, const float *dates_host,
                            int n_times, long n_pixels, int n_bands,
                            int n_lines, int n_samples, const Args &a,
                            int drop_band_idx)
{
    printf("\n");
    printf("========================================================\n");
    printf("  D I A G N O S T I C S   (CPU, %d sample pixels)\n", 12);
    printf("========================================================\n");

    /* build band name list matching the stack layout */
    const char *all_bn[] = {"B12","B11","B9","B8"};
    const char *bnames_diag[4];
    int nbn = 0;
    for (int b = 0; b < 4 && nbn < n_bands; ++b) {
        if (b == drop_band_idx) continue;
        bnames_diag[nbn++] = all_bn[b];
    }

    float omega = 2.0f * 3.14159265f / 365.25f;
    float chi2_crit = chi2_ppf(a.alpha, n_bands);
    int n_pred = 1 + 2 * a.harmonics;

    /* Pick 10 evenly-spaced pixels + center + corner */
    long sample_pids[12];
    for (int i = 0; i < 10; ++i)
        sample_pids[i] = (long)i * n_pixels / 10;
    sample_pids[10] = (long)(n_lines/2) * n_samples + n_samples/2;  /* center */
    sample_pids[11] = n_pixels - 1;                                  /* last */

    /* --- Global stack stats (sample first 100 pixels for speed) --- */
    printf("\n--- GLOBAL DATA CHECKS ---\n");
    {
        long check_n = (n_pixels < 100) ? n_pixels : 100;
        int all_zero=1, all_nan=1, all_same=1;
        float first_val = 0;
        int first_set = 0;
        for (long p = 0; p < check_n; ++p) {
            for (int ti = 0; ti < n_times && ti < 5; ++ti) {
                for (int b = 0; b < n_bands; ++b) {
                    float v = stack_host[((long)ti*n_bands+b)*n_pixels + p];
                    if (!isnan(v)) {
                        all_nan = 0;
                        if (v != 0.0f) all_zero = 0;
                        if (!first_set) { first_val=v; first_set=1; }
                        else if (v != first_val) all_same = 0;
                    }
                }
            }
        }
        printf("  First 100 pixels x first 5 times:\n");
        printf("    all_NaN=%d  all_zero=%d  all_same_value=%d",
               all_nan, all_zero, all_same);
        if (first_set) printf("  (first_val=%.6f)", first_val);
        printf("\n");
    }

    /* --- Per-pixel diagnostics --- */
    for (int si = 0; si < 12; ++si) {
        long pid = sample_pids[si];
        if (pid < 0 || pid >= n_pixels) continue;

        int row = (int)(pid / n_samples);
        int col = (int)(pid % n_samples);

        printf("\n--- PIXEL %ld  (row=%d, col=%d) ---\n", pid, row, col);

        /* 1. Collect valid observations */
        float *t_cl = (float*)malloc(n_times * sizeof(float));
        float *Y_cl = (float*)malloc(n_times * n_bands * sizeof(float));
        int n_valid = 0;
        int n_nan = 0;

        for (int ti = 0; ti < n_times; ++ti) {
            int any_nan = 0;
            float vals[8];
            for (int b = 0; b < n_bands; ++b) {
                float v = stack_host[((long)ti*n_bands+b)*n_pixels + pid];
                vals[b] = v;
                if (isnan(v)) any_nan = 1;
            }
            if (!any_nan) {
                t_cl[n_valid] = dates_host[ti];
                for (int b = 0; b < n_bands; ++b)
                    Y_cl[n_valid*n_bands+b] = vals[b];
                n_valid++;
            } else {
                n_nan++;
            }
        }

        printf("  Valid obs: %d / %d  (NaN: %d)\n", n_valid, n_times, n_nan);

        if (n_valid == 0) {
            printf("  ALL NaN -- skipping\n");
            free(t_cl); free(Y_cl); continue;
        }

        /* 2. Show raw date range */
        printf("  Raw dates: first=%.0f  last=%.0f\n", t_cl[0], t_cl[n_valid-1]);

        /* 3. Center times */
        float t_base = t_cl[0];
        for (int i = 0; i < n_valid; ++i) t_cl[i] -= t_base;
        printf("  t_base=%.0f  centered range=[0, %.1f]  span=%.2f years\n",
               t_base, t_cl[n_valid-1], t_cl[n_valid-1]/365.25);

        /* 3b. Remove consecutive duplicate observations */
        int n_before_dedup = n_valid;
        {
            int out = 0;
            for (int i = 0; i < n_valid; ++i) {
                if (out == 0) {
                    t_cl[out] = t_cl[i];
                    for (int b = 0; b < n_bands; ++b)
                        Y_cl[out*n_bands+b] = Y_cl[i*n_bands+b];
                    out++;
                } else {
                    bool changed = false;
                    for (int b = 0; b < n_bands; ++b) {
                        if (Y_cl[i*n_bands+b] != Y_cl[(out-1)*n_bands+b]) {
                            changed = true; break;
                        }
                    }
                    if (changed) {
                        t_cl[out] = t_cl[i];
                        for (int b = 0; b < n_bands; ++b)
                            Y_cl[out*n_bands+b] = Y_cl[i*n_bands+b];
                        out++;
                    }
                }
            }
            n_valid = out;
        }
        printf("  Dedup: %d -> %d unique obs (%.0f%% were duplicates)\n",
               n_before_dedup, n_valid,
               100.0*(n_before_dedup - n_valid) / n_before_dedup);

        /* 3c. Filter out winter months (Nov-Mar) */
        int n_before_winter = n_valid;
        {
            int out = 0;
            for (int i = 0; i < n_valid; ++i) {
                int ord = (int)(t_cl[i] + t_base);
                int y, m;
                ordinal_to_ym(ord, &y, &m);
                if (m >= 4 && m <= 10) {
                    t_cl[out] = t_cl[i];
                    for (int b = 0; b < n_bands; ++b)
                        Y_cl[out*n_bands+b] = Y_cl[i*n_bands+b];
                    out++;
                }
            }
            n_valid = out;
        }
        printf("  Winter filter: %d -> %d obs (removed %d Nov-Mar obs)\n",
               n_before_winter, n_valid, n_before_winter - n_valid);

        /* 3d. Monthly medians */
        int n_before_median = n_valid;
        {
            int n_monthly = 0;
            int grp = 0;
            while (grp < n_valid) {
                int ord0 = (int)(t_cl[grp] + t_base);
                int y0, m0;
                ordinal_to_ym(ord0, &y0, &m0);
                int ym0 = y0 * 100 + m0;
                int grp_end = grp + 1;
                while (grp_end < n_valid) {
                    int oe = (int)(t_cl[grp_end] + t_base);
                    int ye, me;
                    ordinal_to_ym(oe, &ye, &me);
                    if (ye*100+me != ym0) break;
                    grp_end++;
                }
                int gsz = grp_end - grp;
                float med_t = (gsz%2==1) ? t_cl[grp+gsz/2]
                              : (t_cl[grp+gsz/2-1]+t_cl[grp+gsz/2])*0.5f;
                float med_b[8];
                for (int b = 0; b < n_bands; ++b) {
                    float sv[64]; int nv=0;
                    for (int i=grp; i<grp_end && nv<64; ++i)
                        sv[nv++] = Y_cl[i*n_bands+b];
                    for (int i=1; i<nv; ++i) {
                        float v=sv[i]; int j=i-1;
                        while(j>=0 && sv[j]>v) { sv[j+1]=sv[j]; --j; }
                        sv[j+1]=v;
                    }
                    med_b[b] = (nv%2==1) ? sv[nv/2] : (sv[nv/2-1]+sv[nv/2])*0.5f;
                }
                t_cl[n_monthly] = med_t;
                for (int b=0; b<n_bands; ++b)
                    Y_cl[n_monthly*n_bands+b] = med_b[b];
                n_monthly++;
                grp = grp_end;
            }
            n_valid = n_monthly;
        }
        printf("  Monthly medians: %d obs -> %d monthly values\n",
               n_before_median, n_valid);

        /* 4. Data value ranges (after monthly medians) */
        float bmin[8]={1e30f,1e30f,1e30f,1e30f}, bmax[8]={-1e30f,-1e30f,-1e30f,-1e30f};
        float bmean[8]={0};
        for (int i = 0; i < n_valid; ++i) {
            for (int b = 0; b < n_bands; ++b) {
                float v = Y_cl[i*n_bands+b];
                if (v < bmin[b]) bmin[b] = v;
                if (v > bmax[b]) bmax[b] = v;
                bmean[b] += v;
            }
        }
        printf("  Band value ranges (min/mean/max):\n");
        for (int b = 0; b < n_bands; ++b) {
            bmean[b] /= n_valid;
            printf("    %s: %.4f / %.4f / %.4f\n",
                   (b<4)?bnames_diag[b]:"Bx", bmin[b], bmean[b], bmax[b]);
        }

        /* 5. Show first 3 observations */
        printf("  First 3 observations:\n");
        for (int i = 0; i < 3 && i < n_valid; ++i) {
            printf("    t=%.1f d  vals=[", t_cl[i]);
            for (int b = 0; b < n_bands; ++b)
                printf("%.4f%s", Y_cl[i*n_bands+b], b<n_bands-1?", ":"");
            printf("]\n");
        }

        /* 6. Guard checks */
        if (n_valid < a.min_history + 3) {
            printf("  FAIL: n_valid=%d < min_history+3=%d -> NODATA\n",
                   n_valid, a.min_history+3);
            free(t_cl); free(Y_cl); continue;
        }
        float span_yr = t_cl[n_valid-1] / 365.25f;
        if (span_yr < a.min_years) {
            printf("  FAIL: span=%.2f years < min_years=%.2f -> NODATA\n",
                   span_yr, a.min_years);
            free(t_cl); free(Y_cl); continue;
        }
        float hist_span = (t_cl[a.min_history-1] - t_cl[0]) / 365.25f;
        /* Expand history window like the kernel does */
        int h_end = a.min_history;
        while (h_end < n_valid && h_end < 512 &&
               (t_cl[h_end-1] - t_cl[0]) / 365.25f < a.min_hist_years) {
            h_end++;
        }
        float expanded_span = (t_cl[h_end-1] - t_cl[0]) / 365.25f;
        if (expanded_span < a.min_hist_years) {
            printf("  FAIL: expanded history %d obs, span=%.2f yr < min_hist_years=%.2f\n",
                   h_end, expanded_span, a.min_hist_years);
            free(t_cl); free(Y_cl); continue;
        }
        if (n_valid - h_end < 3) {
            printf("  FAIL: only %d monitoring obs left after %d history obs\n",
                   n_valid - h_end, h_end);
            free(t_cl); free(Y_cl); continue;
        }
        printf("  Guards passed: initial span=%.2f yr at %d obs, expanded to %d obs "
               "(span=%.2f yr), %d monitoring obs remain\n",
               hist_span, a.min_history, h_end, expanded_span, n_valid - h_end);

        /* 7. Build design matrix for history window */
        printf("  Design matrix (first & last row of %d-obs history):\n", h_end);
        {
            float row_first[8], row_last[8];
            /* first row */
            row_first[0] = 1.0f;
            for (int k = 1; k <= a.harmonics; ++k) {
                row_first[2*k-1] = sinf(k * omega * t_cl[0]);
                row_first[2*k  ] = cosf(k * omega * t_cl[0]);
            }
            /* last row */
            row_last[0] = 1.0f;
            for (int k = 1; k <= a.harmonics; ++k) {
                row_last[2*k-1] = sinf(k * omega * t_cl[h_end-1]);
                row_last[2*k  ] = cosf(k * omega * t_cl[h_end-1]);
            }
            printf("    t=%.1f: [", t_cl[0]);
            for (int p=0;p<n_pred;++p) printf("%.4f%s",row_first[p],p<n_pred-1?", ":"");
            printf("]\n");
            printf("    t=%.1f: [", t_cl[h_end-1]);
            for (int p=0;p<n_pred;++p) printf("%.4f%s",row_last[p],p<n_pred-1?", ":"");
            printf("]\n");
        }

        /* 8. Try OLS fit (simplified, on CPU) */
        printf("  Attempting OLS fit (%d history obs, n_pred=%d)...\n",
               h_end, n_pred);
        {
            /* Build X (h_end x n_pred) */
            float *X = (float*)calloc(h_end * n_pred, sizeof(float));
            for (int i = 0; i < h_end; ++i) {
                X[i*n_pred] = 1.0f;
                for (int k = 1; k <= a.harmonics; ++k) {
                    X[i*n_pred + 2*k-1] = sinf(k * omega * t_cl[i]);
                    X[i*n_pred + 2*k  ] = cosf(k * omega * t_cl[i]);
                }
            }
            /* XtX (n_pred x n_pred) */
            float XtX[25] = {0};
            for (int r = 0; r < n_pred; ++r)
                for (int c = 0; c < n_pred; ++c)
                    for (int i = 0; i < h_end; ++i)
                        XtX[r*n_pred+c] += X[i*n_pred+r] * X[i*n_pred+c];

            printf("    XtX diagonal: [");
            for (int p=0;p<n_pred;++p) printf("%.2f%s",XtX[p*n_pred+p],p<n_pred-1?", ":"");
            printf("]\n");

            /* Cholesky check */
            float L[25] = {0};
            int chol_ok = 1;
            for (int i = 0; i < n_pred && chol_ok; ++i) {
                for (int j = 0; j <= i && chol_ok; ++j) {
                    float s = XtX[i*n_pred+j];
                    for (int k = 0; k < j; ++k) s -= L[i*n_pred+k]*L[j*n_pred+k];
                    if (i == j) {
                        if (s <= 0.0f) { chol_ok = 0; printf("    CHOLESKY FAIL at diag[%d] = %.6e\n", i, s); }
                        else L[i*n_pred+i] = sqrtf(s);
                    } else {
                        L[i*n_pred+j] = s / L[j*n_pred+j];
                    }
                }
            }
            if (chol_ok) {
                printf("    Cholesky: OK\n");

                /* Solve for coefficients: XtX * coef = XtY */
                float XtY[20] = {0}; /* n_pred x n_bands */
                for (int r = 0; r < n_pred; ++r)
                    for (int b = 0; b < n_bands; ++b)
                        for (int i = 0; i < h_end; ++i)
                            XtY[r*n_bands+b] += X[i*n_pred+r] * Y_cl[i*n_bands+b];

                /* Forward/back substitution for band 0 just to show coefs */
                float y2[5], x2[5];
                for (int i=0;i<n_pred;++i) y2[i] = XtY[i*n_bands+0];
                for (int i=0;i<n_pred;++i) {
                    float s=y2[i];
                    for (int k=0;k<i;++k) s-=L[i*n_pred+k]*y2[k];
                    y2[i]=s/L[i*n_pred+i];
                }
                for (int i=n_pred-1;i>=0;--i) {
                    float s=y2[i];
                    for (int k=i+1;k<n_pred;++k) s-=L[k*n_pred+i]*x2[k];
                    x2[i]=s/L[i*n_pred+i];
                }
                printf("    Coefficients (band 0): [");
                for (int p=0;p<n_pred;++p) printf("%.6f%s",x2[p],p<n_pred-1?", ":"");
                printf("]\n");

                /* Compute residuals and show md2 for first 10 monitoring obs */
                /* (very simplified - just uses OLS coefs, no robust refit) */
                float coef_all[20] = {0}; /* n_pred x n_bands */
                for (int b = 0; b < n_bands; ++b) {
                    float yy[5], xx[5];
                    for (int i=0;i<n_pred;++i) yy[i] = XtY[i*n_bands+b];
                    for (int i=0;i<n_pred;++i) {
                        float s=yy[i]; for(int k=0;k<i;++k) s-=L[i*n_pred+k]*yy[k];
                        yy[i]=s/L[i*n_pred+i];
                    }
                    for (int i=n_pred-1;i>=0;--i) {
                        float s=yy[i]; for(int k=i+1;k<n_pred;++k) s-=L[k*n_pred+i]*xx[k];
                        xx[i]=s/L[i*n_pred+i];
                    }
                    for (int p=0;p<n_pred;++p) coef_all[p*n_bands+b] = xx[p];
                }

                /* Compute residual covariance */
                float resid_cov[16] = {0}; /* n_bands x n_bands */
                for (int i = 0; i < h_end; ++i) {
                    float r[4];
                    for (int b=0;b<n_bands;++b) {
                        float pred=0;
                        for (int p=0;p<n_pred;++p)
                            pred += X[i*n_pred+p]*coef_all[p*n_bands+b];
                        r[b] = Y_cl[i*n_bands+b] - pred;
                    }
                    for (int r1=0;r1<n_bands;++r1)
                        for (int c1=0;c1<n_bands;++c1)
                            resid_cov[r1*n_bands+c1] += r[r1]*r[c1];
                }
                float nf = (h_end>1) ? 1.0f/(h_end-1) : 1.0f;
                printf("    Residual cov diag: [");
                for (int b=0;b<n_bands;++b) {
                    resid_cov[b*n_bands+b] = resid_cov[b*n_bands+b]*nf + 1e-6f;
                    printf("%.6e%s", resid_cov[b*n_bands+b], b<n_bands-1?", ":"");
                }
                printf("]\n");

                /* Show md2 for first 10 monitoring obs */
                printf("    md2 for first 10 monitoring obs (chi2_crit=%.2f):\n      ",
                       chi2_crit);
                /* invert cov (just use diagonal for quick diag) */
                float cov_diag_inv[4];
                for (int b=0;b<n_bands;++b)
                    cov_diag_inv[b] = 1.0f / resid_cov[b*n_bands+b];

                int shown = 0;
                for (int i = h_end; i < n_valid && shown < 10; ++i, ++shown) {
                    float xrow[5];
                    xrow[0] = 1.0f;
                    for (int k=1;k<=a.harmonics;++k) {
                        xrow[2*k-1] = sinf(k*omega*t_cl[i]);
                        xrow[2*k  ] = cosf(k*omega*t_cl[i]);
                    }
                    float md2 = 0;
                    for (int b=0;b<n_bands;++b) {
                        float pred=0;
                        for (int p=0;p<n_pred;++p) pred+=xrow[p]*coef_all[p*n_bands+b];
                        float r = Y_cl[i*n_bands+b] - pred;
                        md2 += r*r * cov_diag_inv[b];
                    }
                    printf("%.1f%s", md2, (shown<9)?", ":"");
                }
                printf("\n");
            }
            free(X);
        }

        free(t_cl);
        free(Y_cl);
    }

    printf("\n========================================================\n");
    printf("  END DIAGNOSTICS\n");
    printf("========================================================\n\n");
}

/* =========================================================================
 * main
 * =========================================================================*/
int main(int argc, char **argv)
{
    Args a;
    parse_args(argc, argv, a);

    /* --- read file list --- */
    FILE *fl = fopen(a.input_list, "r");
    if (!fl) { fprintf(stderr, "Cannot open %s\n", a.input_list); return 1; }
    char paths[MAX_TIMES][512];
    float dates_host[MAX_TIMES];
    int n_times = 0;

    /* Each line: YYYYMMDD /abs/path/to/file.bin
       The companion sort_s2_files.py writes this format.          */
    char line[600];
    while (fgets(line, sizeof(line), fl) && n_times < MAX_TIMES) {
        int y,mo,d;
        char path[512];
        if (sscanf(line, "%4d%2d%2d %511s", &y, &mo, &d, path) == 4) {
            /* ordinal date: days since 0001-01-01 (matches Python date.toordinal) */
            /* Zeller's congruence variant for proleptic Gregorian */
            int m = mo, yr = y;
            if (m <= 2) { m += 12; yr -= 1; }
            int K = yr % 100, J = yr / 100;
            int dow = (d + (int)(13*(m+1)/5.0) + K + K/4 + J/4 - 2*J) % 7;
            (void)dow;
            /* Simple ordinal: use struct tm */
            struct tm tm0 = {0};
            tm0.tm_year = y - 1900; tm0.tm_mon = mo-1; tm0.tm_mday = d;
            mktime(&tm0);
            /* Python ordinal = Unix days since 0001-01-01 */
            /* 1970-01-01 = ordinal 719163 */
            time_t epoch = mktime(&tm0);
            dates_host[n_times] = 719163.0f + (float)(epoch / 86400);
            strncpy(paths[n_times], path, 511);
            ++n_times;
        }
    }
    fclose(fl);
    printf("Found %d time steps\n", n_times);

    long n_pixels = (long)a.lines * a.samples;

    /* --- detect and drop B9 (945nm water vapor) band --- */
    int drop_band_idx = (n_times > 0) ? detect_b9_band(paths[0]) : -1;
    int n_bands_file = a.bands;   /* original band count in files */
    if (drop_band_idx >= 0 && drop_band_idx < n_bands_file) {
        printf("Dropping band %d (B9 945nm water vapor) from %d-band input\n",
               drop_band_idx, n_bands_file);
        a.bands = n_bands_file - 1;  /* all subsequent a.bands refs use reduced count */
    } else {
        printf("No B9 band detected in header -- using all %d bands\n", n_bands_file);
        drop_band_idx = -1;
    }

    long n_per_image_file = (long)n_bands_file * n_pixels;
    long n_per_image_out  = (long)a.bands * n_pixels;
    long n_total = (long)n_times * n_per_image_out;

    /* --- allocate host stack (using reduced band count) --- */
    printf("Allocating %.1f GB host RAM (%d bands x %d times x %ld pixels)...\n",
           n_total*4.0/1e9, a.bands, n_times, n_pixels);
    float *stack_host = (float*)malloc(n_total * sizeof(float));
    if (!stack_host) { fprintf(stderr,"malloc failed\n"); return 1; }

    /* --- load all images, skipping dropped band --- */
    printf("Loading images...\n");
    for (int ti = 0; ti < n_times; ++ti) {
        float *img = read_envi_float(paths[ti], n_per_image_file);
        if (!img) { fprintf(stderr,"Failed loading %s\n",paths[ti]); return 1; }

        if (drop_band_idx >= 0) {
            /* copy bands, skipping the dropped one */
            int dst_b = 0;
            for (int b = 0; b < n_bands_file; ++b) {
                if (b == drop_band_idx) continue;
                memcpy(stack_host + (long)ti * n_per_image_out + (long)dst_b * n_pixels,
                       img + (long)b * n_pixels,
                       n_pixels * sizeof(float));
                dst_b++;
            }
        } else {
            memcpy(stack_host + (long)ti * n_per_image_out,
                   img, n_per_image_out * sizeof(float));
        }
        free(img);
        if ((ti+1) % 50 == 0)
            printf("  Loaded %d / %d\n", ti+1, n_times);
    }
    printf("  All images loaded (%d bands per image after B9 filter)\n", a.bands);

    /* --- run CPU-side diagnostics on sample pixels --- */
    run_diagnostics(stack_host, dates_host, n_times, n_pixels,
                    a.bands, a.lines, a.samples, a, drop_band_idx);

    /* --- GPU setup --- */
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("GPU: %s  (%.0f GB VRAM)\n",
           prop.name, prop.totalGlobalMem/1e9);

    /* --- query free GPU memory and compute tile size --- */
    size_t gpu_free = 0, gpu_total = 0;
    cudaMemGetInfo(&gpu_free, &gpu_total);
    printf("GPU memory: %.1f GB free / %.1f GB total\n",
           gpu_free/1e9, gpu_total/1e9);

    /*
     * Per-pixel GPU memory breakdown:
     *   stack data  : n_times * n_bands * 4  bytes
     *   outputs     : 2*4 (int32) + (1+n_bands)*4 (float) bytes
     *   local memory: each thread allocates large scratch arrays from
     *                 global memory (t_cl, Y_cl, stable_buf, X_st, Y_st,
     *                 resid_st).  CUDA allocates local mem for ALL launched
     *                 threads, not just resident ones.
     *                 Estimate: ~(3*MAX_TIMES + 3*MAX_TIMES*MAX_BANDS
     *                           + MAX_TIMES*MAX_PRED) * 4  + 4096
     */
    size_t stack_per_px  = (size_t)n_times * a.bands * sizeof(float);
    size_t out_per_px    = 2*sizeof(int32_t) + (size_t)(1 + a.bands)*sizeof(float);
    size_t local_per_thr = (
        /* MAX_TIMES-sized arrays: t_cl, stable_buf, norms, idx,
           Y_cl, Y_st, resid_st, X_st, W, rms, sorted, devs (fit_robust) */
        (size_t)(8*MAX_TIMES + 3*MAX_TIMES*MAX_BANDS + MAX_TIMES*MAX_PRED)
        /* MAX_HISTORY-sized arrays: X_hist, resid_hist */
        + (size_t)(MAX_HISTORY*MAX_PRED + MAX_HISTORY*MAX_BANDS)
    ) * sizeof(float) + 8192; /* coef, cov, cov_inv, small scratch */
    size_t total_per_px  = stack_per_px + out_per_px + local_per_thr;

    /* leave 2 GB headroom for driver / CUDA runtime */
    size_t budget = gpu_free - (size_t)2ULL * 1024 * 1024 * 1024;
    long tile_pixels = (long)(budget / total_per_px);

    /* round to multiple of threads_per_block */
    int tpb = a.threads_per_block;
    tile_pixels = (tile_pixels / tpb) * tpb;
    if (tile_pixels < tpb) tile_pixels = tpb;
    if (tile_pixels > n_pixels) tile_pixels = n_pixels;

    int n_tiles = (int)((n_pixels + tile_pixels - 1) / tile_pixels);
    printf("Tile size: %ld pixels  (%d tiles to cover %ld pixels)\n",
           tile_pixels, n_tiles, n_pixels);
    printf("Per-pixel GPU footprint: %.1f KB "
           "(stack %.1f + local %.1f + out %.0f B)\n",
           total_per_px/1024.0, stack_per_px/1024.0,
           local_per_thr/1024.0, (double)out_per_px);

    /* --- allocate host output arrays (full image) --- */
    int32_t *h_tbreak = (int32_t*)malloc(n_pixels*sizeof(int32_t));
    int32_t *h_count  = (int32_t*)malloc(n_pixels*sizeof(int32_t));
    float   *h_magrms = (float*)  malloc(n_pixels*sizeof(float));
    float   *h_magb   = (float*)  malloc((long)a.bands*n_pixels*sizeof(float));
    if (!h_tbreak||!h_count||!h_magrms||!h_magb) {
        fprintf(stderr, "Host output malloc failed\n"); return 1;
    }

    /* --- dates to GPU (shared across all tiles, tiny) --- */
    float *d_dates;
    cudaMalloc(&d_dates, n_times * sizeof(float));
    cudaMemcpy(d_dates, dates_host, n_times*sizeof(float), cudaMemcpyHostToDevice);

    /* --- allocate GPU buffers sized for one tile --- */
    float   *d_stack;
    int32_t *d_tbreak, *d_count;
    float   *d_magrms, *d_magb;

    size_t tile_stack_bytes = (size_t)tile_pixels * n_times * a.bands * sizeof(float);
    cudaMalloc(&d_stack,  tile_stack_bytes);
    cudaMalloc(&d_tbreak, tile_pixels * sizeof(int32_t));
    cudaMalloc(&d_count,  tile_pixels * sizeof(int32_t));
    cudaMalloc(&d_magrms, tile_pixels * sizeof(float));
    cudaMalloc(&d_magb,   (long)a.bands * tile_pixels * sizeof(float));

    cudaError_t alloc_err = cudaGetLastError();
    if (alloc_err != cudaSuccess) {
        fprintf(stderr, "GPU alloc failed: %s\n", cudaGetErrorString(alloc_err));
        return 1;
    }

    /* host-side tile extraction buffer (reused each tile) */
    float *tile_buf = (float*)malloc(tile_stack_bytes);
    if (!tile_buf) { fprintf(stderr,"tile_buf malloc failed\n"); return 1; }

    /* --- fill constant params (updated per tile for tile_n_pixels) --- */
    Params hp;
    hp.n_times        = n_times;
    hp.n_lines        = a.lines;
    hp.n_samples      = a.samples;
    hp.n_bands        = a.bands;
    hp.min_history    = a.min_history;
    hp.harmonics      = a.harmonics;
    hp.refit_interval = a.refit_interval;
    hp.alpha          = a.alpha;
    hp.cusum_k        = a.cusum_k;
    hp.cusum_thresh_k = a.cusum_thresh_k;
    hp.min_years      = a.min_years;
    hp.min_hist_years = a.min_hist_years;
    hp.chi2_crit      = chi2_ppf(a.alpha, a.bands);
    hp.omega          = 2.0f * 3.14159265f / 365.25f;
    hp.tile_n_pixels  = 0; /* set per tile below */

    printf("chi2_crit (alpha=%.2f, df=%d) = %.4f\n",
           a.alpha, a.bands, hp.chi2_crit);

    /* --- tiled processing loop --- */
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    float total_kernel_ms = 0.0f;

    for (int tile = 0; tile < n_tiles; ++tile) {
        long p_off  = (long)tile * tile_pixels;
        long p_end  = p_off + tile_pixels;
        if (p_end > n_pixels) p_end = n_pixels;
        long tile_sz = p_end - p_off;

        printf("Tile %d/%d : pixels %ld .. %ld  (%ld px)\n",
               tile+1, n_tiles, p_off, p_end-1, tile_sz);

        /* --- extract tile pixels from host stack ---
         * Host stack layout: (n_times, n_bands, n_pixels)
         * Each (t, b) plane has n_pixels contiguous floats.
         * We copy tile_sz contiguous floats starting at offset p_off
         * from each plane into tile_buf at (t*n_bands+b)*tile_sz.
         */
        for (int ti = 0; ti < n_times; ++ti) {
            for (int b = 0; b < a.bands; ++b) {
                long src_off = ((long)ti * a.bands + b) * n_pixels + p_off;
                long dst_off = ((long)ti * a.bands + b) * tile_sz;
                memcpy(tile_buf + dst_off,
                       stack_host + src_off,
                       tile_sz * sizeof(float));
            }
        }

        /* --- copy tile to GPU --- */
        size_t tile_bytes = (size_t)tile_sz * n_times * a.bands * sizeof(float);
        cudaMemcpy(d_stack, tile_buf, tile_bytes, cudaMemcpyHostToDevice);

        /* --- update params with this tile's pixel count --- */
        hp.tile_n_pixels = (int)tile_sz;
        cudaMemcpyToSymbol(d_params, &hp, sizeof(Params));

        /* --- launch kernel --- */
        int blocks = ((int)tile_sz + tpb - 1) / tpb;
        cudaEventRecord(ev0);

        ccdc_kernel<<<blocks, tpb>>>(d_stack, d_dates,
                                      d_tbreak, d_count,
                                      d_magrms, d_magb);

        cudaEventRecord(ev1);
        cudaEventSynchronize(ev1);

        cudaError_t kerr = cudaGetLastError();
        if (kerr != cudaSuccess) {
            fprintf(stderr, "Kernel error tile %d: %s\n",
                    tile+1, cudaGetErrorString(kerr));
            return 1;
        }
        float tile_ms;
        cudaEventElapsedTime(&tile_ms, ev0, ev1);
        total_kernel_ms += tile_ms;
        printf("  Kernel: %.2f s\n", tile_ms/1000.0f);

        /* --- copy tile results back to correct host offset --- */
        cudaMemcpy(h_tbreak + p_off, d_tbreak,
                   tile_sz*sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_count  + p_off, d_count,
                   tile_sz*sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_magrms + p_off, d_magrms,
                   tile_sz*sizeof(float),   cudaMemcpyDeviceToHost);
        /* per-band magnitude: d_magb layout is (n_bands, tile_sz) on GPU
         * but h_magb layout is (n_bands, n_pixels) on host */
        for (int b = 0; b < a.bands; ++b) {
            cudaMemcpy(h_magb + (long)b*n_pixels + p_off,
                       d_magb + (long)b*tile_sz,
                       tile_sz*sizeof(float), cudaMemcpyDeviceToHost);
        }
    }

    printf("All tiles done. Total kernel time: %.2f s\n",
           total_kernel_ms/1000.0f);

    /* --- cleanup GPU tile buffers --- */
    cudaFree(d_stack); cudaFree(d_dates);
    cudaFree(d_tbreak); cudaFree(d_count);
    cudaFree(d_magrms); cudaFree(d_magb);
    free(tile_buf);
    free(stack_host);

    /* --- summary stats --- */
    long n_changed=0, n_nodata=0;
    for (long i=0; i<n_pixels; ++i) {
        if (h_count[i] == NODATA_INT) ++n_nodata;
        else if (h_count[i] > 0)     ++n_changed;
    }
    printf("Pixels with >= 1 break : %ld (%.1f%%)\n",
           n_changed, 100.0*n_changed/n_pixels);
    printf("Nodata pixels          : %ld (%.1f%%)\n",
           n_nodata,  100.0*n_nodata/n_pixels);

    /* --- read georef from first file (GDAL) --- */
    GDALAllRegister();
    double geotransform[6] = {631840,20,0,6479840,0,-20}; /* fallback */
    char wkt_buf[4096] = "";

    GDALDatasetH ref_ds = GDALOpen(paths[0], GA_ReadOnly);
    if (ref_ds) {
        GDALGetGeoTransform(ref_ds, geotransform);
        const char *proj = GDALGetProjectionRef(ref_ds);
        if (proj) strncpy(wkt_buf, proj, sizeof(wkt_buf)-1);
        GDALClose(ref_ds);
    }

    /* --- build output band name list (excluding dropped band) --- */
    const char *all_bnames[] = {"B12","B11","B9","B8"};
    const char *bnames[MAX_BANDS];
    {
        int nb = 0;
        for (int b = 0; b < n_bands_file && nb < MAX_BANDS; ++b) {
            if (b == drop_band_idx) continue;
            bnames[nb++] = (b < 4) ? all_bnames[b] : "Bx";
        }
    }

    /* --- write outputs --- */
    char outpath[600];

    /* create output dir */
    char mkdir_cmd[600];
    snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s", a.output_dir);
    if (system(mkdir_cmd) != 0)
        fprintf(stderr, "Warning: mkdir -p failed\n");

    snprintf(outpath, sizeof(outpath), "%s/tBreak_first.bin", a.output_dir);
    write_envi_band(outpath, h_tbreak, a.samples, a.lines, GDT_Int32,
                    geotransform, wkt_buf, NODATA_INT,
                    "Ordinal date of first break (0=none,-9999=nodata)");

    snprintf(outpath, sizeof(outpath), "%s/tBreak_count.bin", a.output_dir);
    write_envi_band(outpath, h_count, a.samples, a.lines, GDT_Int32,
                    geotransform, wkt_buf, NODATA_INT,
                    "Number of breaks detected (-9999=nodata)");

    snprintf(outpath, sizeof(outpath), "%s/MAG_rms.bin", a.output_dir);
    write_envi_band(outpath, h_magrms, a.samples, a.lines, GDT_Float32,
                    geotransform, wkt_buf, -9999.0,
                    "RMS change magnitude at first break");

    for (int b=0; b<a.bands; ++b) {
        snprintf(outpath, sizeof(outpath), "%s/MAG_%s.bin",
                 a.output_dir, bnames[b]);
        char desc[128];
        snprintf(desc, sizeof(desc),
                 "Signed magnitude %s at first break (negative=loss)",
                 bnames[b]);
        write_envi_band(outpath, h_magb + (long)b*n_pixels,
                        a.samples, a.lines, GDT_Float32,
                        geotransform, wkt_buf, -9999.0, desc);
    }

    printf("Outputs written to %s\n", a.output_dir);

    /* cleanup host outputs */
    free(h_tbreak); free(h_count); free(h_magrms); free(h_magb);

    return 0;
}

/*
 * ===========================================================================
 * Companion: sort_s2_files.py
 * ===========================================================================
 * Writes files.txt in the format expected by the --input_list argument:
 *
 *   python3 sort_s2_files.py [input_dir] > files.txt
 *
 * ---------------------------------------------------------------------------
 * import re, sys
 * from pathlib import Path
 * from datetime import datetime
 *
 * d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
 * pat = re.compile(r"S2[ABC]_MSIL2A_(\d{8})T\d{6}_")
 * rows = []
 * for f in d.glob("S2*.bin"):
 *     m = pat.search(f.name)
 *     if m:
 *         rows.append((m.group(1), str(f.resolve())))
 * for date_str, path in sorted(rows):
 *     print(date_str, path)
 * ---------------------------------------------------------------------------
 *
 * ===========================================================================
 * Build:
 *   nvcc -O3 -arch=sm_89 -o miniccdc_cuda miniccdc_cuda.cu -lgdal
 *
 * For older CUDA without C++14 lambdas in device code, replace the lambda
 * set_nodata() with an inline macro or helper function.
 * ===========================================================================
 */



