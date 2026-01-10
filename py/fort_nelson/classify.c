/*
gcc -O3 -Wall classify.c -o classify \
    -I/usr/local/include \
    -L/usr/local/lib \
    -lgdal -llapacke -llapack -lcblas -lblas -lpthread -lm

 
 Gaussian patch classifier (C version)
 Uses:
  - GDAL (C API)
  - BLAS / LAPACK (CBLAS + LAPACKE)
  - pthreads
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>

#include <gdal.h>
#include <cpl_conv.h>

#include <cblas.h>
#include <lapacke.h>

/* ---------------- constants ---------------- */

#define MAX_CLASSES 2
#define PATCH_SIZE 7

/* ---------------- data structures ---------------- */

typedef struct {
    int dim;
    double *mean;
    double *cov;
    double *inv_cov;
} ClassStats;

typedef struct {
    int y0, y1;
    int width, height, bands;
    int pad;
    double *image;
    unsigned char *out;
    ClassStats *stats;
} Job;

/* ---------------- global stats ---------------- */

static ClassStats stats[MAX_CLASSES];

/* ---------------- utilities ---------------- */

static void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

/* ---------------- load global_stats.txt ---------------- */

static void load_global_stats(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f)
        die("Failed to open global_stats.txt");

    char line[256];
    int cls = -1;

    while (fgets(line, sizeof(line), f)) {

        if (sscanf(line, "CLASS %d", &cls) == 1)
            continue;

        if (cls < 0 || cls >= MAX_CLASSES)
            continue;

        if (sscanf(line, "DIM %d", &stats[cls].dim) == 1) {
            int d = stats[cls].dim;
            stats[cls].mean = calloc(d, sizeof(double));
            stats[cls].cov = calloc(d * d, sizeof(double));
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

            memcpy(stats[cls].inv_cov, stats[cls].cov,
                   sizeof(double) * d * d);

            /* Cholesky inverse if possible */
            int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U',
                                      d, stats[cls].inv_cov, d);
            if (info == 0) {
                LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'U',
                               d, stats[cls].inv_cov, d);
            } else {
                /* fallback to LU */
                int *ipiv = malloc(sizeof(int) * d);
                LAPACKE_dgetrf(LAPACK_ROW_MAJOR,
                               d, d, stats[cls].inv_cov, d, ipiv);
                LAPACKE_dgetri(LAPACK_ROW_MAJOR,
                               d, stats[cls].inv_cov, d, ipiv);
                free(ipiv);
            }
        }
    }

    fclose(f);
}

/* ---------------- classification worker ---------------- */

static void *classify_rows(void *arg) {
    Job *job = (Job *)arg;

    int d = PATCH_SIZE * PATCH_SIZE * job->bands;
    double *vec = malloc(sizeof(double) * d);
    double *tmp = malloc(sizeof(double) * d);

    for (int y = job->y0; y < job->y1; y++) {
        for (int x = 0; x < job->width; x++) {

            /* extract patch */
            int idx = 0;
            for (int dy = 0; dy < PATCH_SIZE; dy++)
                for (int dx = 0; dx < PATCH_SIZE; dx++)
                    for (int b = 0; b < job->bands; b++) {
                        int yy = y + dy;
                        int xx = x + dx;
                        vec[idx++] =
                            job->image[(yy * job->width + xx) * job->bands + b];
                    }

            int best_cls = 0;
            double best = INFINITY;

            for (int c = 0; c < MAX_CLASSES; c++) {
                if (!job->stats[c].mean)
                    continue;

                for (int i = 0; i < d; i++)
                    vec[i] -= job->stats[c].mean[i];

                cblas_dgemv(CblasRowMajor, CblasNoTrans,
                            d, d, 1.0,
                            job->stats[c].inv_cov, d,
                            vec, 1, 0.0, tmp, 1);

                double score = cblas_ddot(d, vec, 1, tmp, 1);

                if (score < best) {
                    best = score;
                    best_cls = c;
                }
            }

            job->out[y * job->width + x] = (unsigned char)best_cls;
        }
    }

    free(vec);
    free(tmp);
    return NULL;
}

/* ---------------- main ---------------- */

int main(int argc, char **argv) {

    if (argc != 3) {
        fprintf(stderr, "Usage: %s global_stats.txt image.tif\n", argv[0]);
        return 1;
    }

    GDALAllRegister();

    load_global_stats(argv[1]);

    GDALDatasetH ds = GDALOpen(argv[2], GA_ReadOnly);
    if (!ds)
        die("Failed to open image");

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

        GDALRasterIO(rb, GF_Read,
                     0, 0, w, h,
                     tmp, w, h, GDT_Float64, 0, 0);

        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                image[((y + pad) * pw + (x + pad)) * bands + b] =
                    tmp[y * w + x];

        free(tmp);
    }

    unsigned char *out = calloc(w * h, 1);

    int nthreads = sysconf(_SC_NPROCESSORS_ONLN);
    pthread_t *threads = malloc(sizeof(pthread_t) * nthreads);
    Job *jobs = malloc(sizeof(Job) * nthreads);

    int rows = h / nthreads;

    for (int t = 0; t < nthreads; t++) {
        jobs[t] = (Job){
            .y0 = t * rows,
            .y1 = (t == nthreads - 1) ? h : (t + 1) * rows,
            .width = pw,
            .height = ph,
            .bands = bands,
            .pad = pad,
            .image = image,
            .out = out,
            .stats = stats
        };
        pthread_create(&threads[t], NULL, classify_rows, &jobs[t]);
    }

    for (int t = 0; t < nthreads; t++)
        pthread_join(threads[t], NULL);

    GDALDriverH drv = GDALGetDriverByName("ENVI");
    GDALDatasetH ods =
        GDALCreate(drv, "classification.bin", w, h, 1, GDT_Byte, NULL);

    GDALRasterBandH ob = GDALGetRasterBand(ods, 1);
    GDALRasterIO(ob, GF_Write,
                 0, 0, w, h,
                 out, w, h, GDT_Byte, 0, 0);

    GDALClose(ods);
    GDALClose(ds);

    free(image);
    free(out);
    free(threads);
    free(jobs);

    return 0;
}

