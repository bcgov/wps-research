/* 20260609
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
#define MAX_TIMES    1024   /* max time steps; increase if needed */
#define MAX_PRED     5      /* max design-matrix columns: 1 + 2*harmonics */
#define MAX_HISTORY  64     /* max min_history value accepted */
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
    float W[MAX_HISTORY];
    for (int i = 0; i < n_obs; ++i) W[i] = 1.0f;

    for (int iter = 0; iter < 5; ++iter) {
        if (wlsq(X_rows, Y, W, n_obs, n_pred, n_bands, coef) != 0)
            return -1;

        /* compute per-observation RMS residual */
        float rms[MAX_HISTORY];
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
        float sorted[MAX_HISTORY];
        for (int i = 0; i < n_obs; ++i) sorted[i] = rms[i];
        for (int i = 1; i < n_obs; ++i) {
            float v = sorted[i]; int j = i-1;
            while (j >= 0 && sorted[j] > v) { sorted[j+1] = sorted[j]; --j; }
            sorted[j+1] = v;
        }
        float med = (n_obs % 2) ? sorted[n_obs/2]
                                 : 0.5f*(sorted[n_obs/2-1]+sorted[n_obs/2]);
        float devs[MAX_HISTORY];
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
    float norms[MAX_HISTORY];
    for (int i = 0; i < n_obs; ++i) {
        float s = 0.0f;
        for (int b = 0; b < n_bands; ++b)
            s += resid[i*n_bands+b] * resid[i*n_bands+b];
        norms[i] = sqrtf(s);
    }
    /* argsort by norm (insertion) */
    int idx[MAX_HISTORY];
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
    int n_pixels = p.n_lines * p.n_samples;
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

    /* ---- 2. Guard checks ---- */
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

        /* guard: history window must span enough time */
        float hist_span = (t_cl[hist_end-1] - t_cl[seg_start]) / 365.25f;
        if (hist_span < p.min_hist_years) break;

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
            first_break_date  = (int32_t)t_cl[break_idx];
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
    (void)GDALSetDescription(band, desc);
    GDALRasterIO(band, GF_Write, 0, 0, cols, rows,
                 data, cols, rows, dtype, 0, 0);
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
    long n_per_image = (long)a.bands * n_pixels;
    long n_total = (long)n_times * n_per_image;

    /* --- allocate host stack --- */
    printf("Allocating %.1f GB host RAM...\n",
           n_total*4.0/1e9);
    float *stack_host = (float*)malloc(n_total * sizeof(float));
    if (!stack_host) { fprintf(stderr,"malloc failed\n"); return 1; }

    /* --- load all images --- */
    printf("Loading images...\n");
    for (int ti = 0; ti < n_times; ++ti) {
        float *img = read_envi_float(paths[ti], n_per_image);
        if (!img) { fprintf(stderr,"Failed loading %s\n",paths[ti]); return 1; }
        memcpy(stack_host + (long)ti * n_per_image,
               img, n_per_image * sizeof(float));
        free(img);
        if ((ti+1) % 50 == 0)
            printf("  Loaded %d / %d\n", ti+1, n_times);
    }
    printf("  All images loaded\n");

    /* --- GPU setup --- */
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("GPU: %s  (%.0f GB VRAM)\n",
           prop.name, prop.totalGlobalMem/1e9);

    /* --- copy stack to GPU --- */
    printf("Copying stack to GPU (%.1f GB)...\n", n_total*4.0/1e9);
    float *d_stack;
    cudaMalloc(&d_stack, n_total * sizeof(float));
    cudaMemcpy(d_stack, stack_host, n_total*sizeof(float), cudaMemcpyHostToDevice);
    free(stack_host);

    float *d_dates;
    cudaMalloc(&d_dates, n_times * sizeof(float));
    cudaMemcpy(d_dates, dates_host, n_times*sizeof(float), cudaMemcpyHostToDevice);

    /* --- output buffers on GPU --- */
    int32_t *d_tbreak, *d_count;
    float   *d_magrms, *d_magb;
    cudaMalloc(&d_tbreak, n_pixels * sizeof(int32_t));
    cudaMalloc(&d_count,  n_pixels * sizeof(int32_t));
    cudaMalloc(&d_magrms, n_pixels * sizeof(float));
    cudaMalloc(&d_magb,   (long)a.bands * n_pixels * sizeof(float));

    /* --- fill constant params --- */
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

    printf("chi2_crit (alpha=%.2f, df=%d) = %.4f\n",
           a.alpha, a.bands, hp.chi2_crit);

    cudaMemcpyToSymbol(d_params, &hp, sizeof(Params));

    /* --- launch kernel --- */
    int tpb    = a.threads_per_block;
    int blocks = ((int)n_pixels + tpb - 1) / tpb;
    printf("Launching %d blocks x %d threads = %d threads\n",
           blocks, tpb, blocks*tpb);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    ccdc_kernel<<<blocks, tpb>>>(d_stack, d_dates,
                                  d_tbreak, d_count,
                                  d_magrms, d_magb);

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("Kernel finished in %.2f s\n", ms/1000.0f);

    /* --- copy results back --- */
    int32_t *h_tbreak = (int32_t*)malloc(n_pixels*sizeof(int32_t));
    int32_t *h_count  = (int32_t*)malloc(n_pixels*sizeof(int32_t));
    float   *h_magrms = (float*)  malloc(n_pixels*sizeof(float));
    float   *h_magb   = (float*)  malloc((long)a.bands*n_pixels*sizeof(float));

    cudaMemcpy(h_tbreak, d_tbreak, n_pixels*sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_count,  d_count,  n_pixels*sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_magrms, d_magrms, n_pixels*sizeof(float),   cudaMemcpyDeviceToHost);
    cudaMemcpy(h_magb,   d_magb,   (long)a.bands*n_pixels*sizeof(float), cudaMemcpyDeviceToHost);

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

    /* --- write outputs --- */
    const char *bnames[] = {"B12","B11","B9","B8"};
    char outpath[600];

    /* create output dir */
    char mkdir_cmd[600];
    snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s", a.output_dir);
    (void)system(mkdir_cmd);

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
                 a.output_dir, (b < 4) ? bnames[b] : "Bx");
        char desc[128];
        snprintf(desc, sizeof(desc),
                 "Signed magnitude %s at first break (negative=loss)",
                 (b<4)?bnames[b]:"Bx");
        write_envi_band(outpath, h_magb + (long)b*n_pixels,
                        a.samples, a.lines, GDT_Float32,
                        geotransform, wkt_buf, -9999.0, desc);
    }

    printf("Outputs written to %s\n", a.output_dir);

    /* cleanup */
    free(h_tbreak); free(h_count); free(h_magrms); free(h_magb);
    cudaFree(d_stack); cudaFree(d_dates);
    cudaFree(d_tbreak); cudaFree(d_count);
    cudaFree(d_magrms); cudaFree(d_magb);

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
