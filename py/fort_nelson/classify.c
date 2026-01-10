/*
 * classify.c
 *
 * gcc -O3 -Wall classify.c -o classify -lgdal -lblas -llapack -lpthread -lm
 *
 * Run global_stats_to_txt.py after generating global_stats.pkl using classify_one3.py
 *
 * /

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <glob.h>

#include "gdal.h"
#include "cpl_conv.h"
#include "cblas.h"
#include "lapacke.h"

/* ---------------- constants ---------------- */

#define PATCH_SIZE 7
#define MAX_CLASSES 2

/* ---------------- global stats ---------------- */

typedef struct {
    int dim;
    double *mean;
    double *cov;
    double *inv_cov;
    int valid;
} ClassStats;

ClassStats stats[MAX_CLASSES];

/* ---------------- threading ---------------- */

typedef struct {
    int y0, y1;
    int width;
    int height;
    int bands;
    double *image;   /* padded image */
    unsigned char *out;
} ThreadJob;

int nthreads = 0;

/* ---------------- utilities ---------------- */

static inline int idx3(int y, int x, int b, int w, int bands) {
    return (y * w + x) * bands + b;
}

/* ---------------- stats loading ---------------- */

void load_global_stats(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", path);
        exit(1);
    }

    char line[8192];
    int cls = -1;

    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') continue;

        if (sscanf(line, "CLASS %d", &cls) == 1) {
            stats[cls].valid = 1;
            continue;
        }

        if (sscanf(line, "DIM %d", &stats[cls].dim) == 1) {
            int d = stats[cls].dim;
            stats[cls].mean = calloc(d, sizeof(double));
            stats[cls].cov  = calloc(d * d, sizeof(double));
            stats[cls].inv_cov = calloc(d * d, sizeof(double));
            continue;
        }

        if (strncmp(line, "MEAN", 4) == 0) {
            for (int i = 0; i < stats[cls].dim; i++)
                fscanf(f, "%lf", &stats[cls].mean[i]);
            continue;
        }

        if (strncmp(line, "COV", 3) == 0) {
            int d = stats[cls].dim;
            for (int i = 0; i < d * d; i++)
                fscanf(f, "%lf", &stats[cls].cov[i]);

            /* invert covariance */
            memcpy(stats[cls].inv_cov, stats[cls].cov,
                   d * d * sizeof(double));

            int info = LAPACKE_dpotrf(
                LAPACK_ROW_MAJOR, 'U', d, stats[cls].inv_cov, d
            );
            if (info == 0)
                LAPACKE_dpotri(
                    LAPACK_ROW_MAJOR, 'U', d, stats[cls].inv_cov, d
                );
            else
                LAPACKE_dgetrf(
                    LAPACK_ROW_MAJOR, d, d, stats[cls].inv_cov, d, NULL
                );

            continue;
        }
    }

    fclose(f);
}

/* ---------------- classification ---------------- */

void *classify_rows(void *arg) {
    ThreadJob *job = (ThreadJob *)arg;
    int pad = PATCH_SIZE / 2;
    int d = stats[0].dim;

    double *vec = malloc(sizeof(double) * d);
    double *tmp = malloc(sizeof(double) * d);

    for (int y = job->y0; y < job->y1; y++) {
        for (int x = 0; x < job->width; x++) {

            /* extract patch */
            int k = 0;
            for (int dy = 0; dy < PATCH_SIZE; dy++)
                for (int dx = 0; dx < PATCH_SIZE; dx++)
                    for (int b = 0; b < job->bands; b++)
                        vec[k++] = job->image[
                            idx3(y + dy, x + dx, b,
                                 job->width + 2*pad, job->bands)
                        ];

            double best = INFINITY;
            int best_lbl = 0;

            for (int lbl = 0; lbl < MAX_CLASSES; lbl++) {
                if (!stats[lbl].valid) continue;

                for (int i = 0; i < d; i++)
                    vec[i] -= stats[lbl].mean[i];

                cblas_dgemv(
                    CblasRowMajor, CblasNoTrans,
                    d, d, 1.0,
                    stats[lbl].inv_cov, d,
                    vec, 1,
                    0.0, tmp, 1
                );

                double score = cblas_ddot(d, vec, 1, tmp, 1);
                if (score < best) {
                    best = score;
                    best_lbl = lbl;
                }
            }

            job->out[y * job->width + x] = (unsigned char)best_lbl;
        }
    }

    free(vec);
    free(tmp);
    return NULL;
}

/* ---------------- main ---------------- */

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s global_stats.txt image.tif\n", argv[0]);
        return 1;
    }

    GDALAllRegister();

    load_global_stats(argv[1]);

    GDALDatasetH ds = GDALOpen(argv[2], GA_ReadOnly);
    if (!ds) {
        fprintf(stderr, "Failed to open image\n");
        return 1;
    }

    int w = GDALGetRasterXSize(ds);
    int h = GDALGetRasterYSize(ds);
    int bands = GDALGetRasterCount(ds);
    int pad = PATCH_SIZE / 2;

    int pw = w + 2 * pad;
    int ph = h + 2 * pad;

    double *image = calloc(pw * ph * bands, sizeof(double));

    for (int b = 0; b < bands; b++) {
        GDALRasterBandH rb = GDALGetRasterBand(ds, b + 1);
        double *tmp = malloc(sizeof(double) * w * h);
        GDALRasterIO(rb, GF_Read, 0, 0, w, h,
                     tmp, w, h, GDT_Float64, 0, 0);

        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                image[idx3(y+pad, x+pad, b, pw, bands)] =
                    tmp[y * w + x];

        free(tmp);
    }

    unsigned char *out = calloc(w * h, 1);

    nthreads = sysconf(_SC_NPROCESSORS_ONLN);
    pthread_t threads[nthreads];
    ThreadJob jobs[nthreads];

    for (int t = 0; t < nthreads; t++) {
        jobs[t].y0 = h * t / nthreads;
        jobs[t].y1 = h * (t + 1) / nthreads;
        jobs[t].width = w;
        jobs[t].height = h;
        jobs[t].bands = bands;
        jobs[t].image = image;
        jobs[t].out = out;
        pthread_create(&threads[t], NULL, classify_rows, &jobs[t]);
    }

    for (int t = 0; t < nthreads; t++)
        pthread_join(threads[t], NULL);

    GDALDriverH drv = GDALGetDriverByName("ENVI");
    GDALDatasetH ods = GDALCreate(
        drv, "classification.bin", w, h, 1, GDT_Byte, NULL
    );

    GDALSetGeoTransform(ods, GDALGetGeoTransform(ds, NULL));
    GDALSetProjection(ods, GDALGetProjectionRef(ds));

    GDALRasterIO(
        GDALGetRasterBand(ods, 1),
        GF_Write, 0, 0, w, h,
        out, w, h, GDT_Byte, 0, 0
    );

    GDALClose(ods);
    GDALClose(ds);

    free(image);
    free(out);

    return 0;
}


