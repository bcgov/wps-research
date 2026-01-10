/*
gcc -O3 -Wall classify.c -o classify \
    -I/usr/include \
    -L/usr/lib/x86_64-linux-gnu \
    -llapacke -lopenblas -lpthread -lm -lgdal


 Gaussian patch classifier (C version)
 Uses:
  - GDAL (C API)
  - BLAS / LAPACK (CBLAS + LAPACKE)
  - pthreads
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>
#include <stddef.h>

#include "cblas.h"
#include "lapacke.h"
#include "gdal.h"
#include "cpl_conv.h"

#define PATCH_SIZE 1

typedef struct {
    int dim;
    double *mean;
    double *cov;
    double *inv_cov;
} ClassStats;

ClassStats *stats;
int n_classes;

size_t width, height, n_bands;
unsigned char *out_image;

// ---------------- parfor globals ----------------
pthread_attr_t pt_attr;
pthread_mutex_t pt_nxt_j_mtx, print_mtx;
size_t pt_nxt_j, pt_end_j;
void (*pt_eval)(size_t);

// ---------------- parfor formalism ----------------
void pt_init_mtx() {
    pthread_mutex_init(&pt_nxt_j_mtx, NULL);
    pthread_mutex_init(&print_mtx, NULL);
}

void *pt_worker_fun(void *arg) {
    size_t t_id = (size_t)arg;
    while (1) {
        pthread_mutex_lock(&pt_nxt_j_mtx);
        size_t my_nxt_j = pt_nxt_j++;
        pthread_mutex_unlock(&pt_nxt_j_mtx);

        if (my_nxt_j >= pt_end_j) break;

        pthread_mutex_lock(&print_mtx);
        if (my_nxt_j % 100 == 0)
            printf("[Thread %zu] Picking up row %zu\n", t_id, my_nxt_j);
        pthread_mutex_unlock(&print_mtx);

        pt_eval(my_nxt_j);

        pthread_mutex_lock(&print_mtx);
        if (my_nxt_j % 100 == 0)
            printf("[Thread %zu] Finished row %zu\n", t_id, my_nxt_j);
        pthread_mutex_unlock(&print_mtx);
    }
    return NULL;
}

void parfor(size_t start_j, size_t end_j, void(*eval)(size_t), int cores_use) {
    pt_eval = eval;
    pt_end_j = end_j;
    pt_nxt_j = start_j;

    int cores_avail = sysconf(_SC_NPROCESSORS_ONLN);
    size_t n_cores = (cores_use > 0 && (size_t)cores_use < (size_t)cores_avail) ? cores_use : cores_avail;

    pthread_attr_init(&pt_attr);
    pthread_attr_setdetachstate(&pt_attr, PTHREAD_CREATE_JOINABLE);

    pthread_t *threads = malloc(n_cores * sizeof(pthread_t));

    for (size_t t = 0; t < n_cores; t++)
        pthread_create(&threads[t], &pt_attr, pt_worker_fun, (void *)t);

    for (size_t t = 0; t < n_cores; t++)
        pthread_join(threads[t], NULL);

    free(threads);
}

void parfor_simple(size_t start_j, size_t end_j, void(*eval)(size_t)) {
    parfor(start_j, end_j, eval, 0);
}

// ---------------- Global stats loading ----------------
void load_global_stats(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { perror("Failed to open global stats"); exit(1); }

    fscanf(f, "%d", &n_classes);
    stats = calloc(n_classes, sizeof(ClassStats));

    for (int c = 0; c < n_classes; c++) {
        fscanf(f, "%d", &stats[c].dim);
        int d = stats[c].dim;
        stats[c].mean = calloc(d, sizeof(double));
        stats[c].cov  = calloc(d*d, sizeof(double));
        stats[c].inv_cov = calloc(d*d, sizeof(double));

        for (int i = 0; i < d; i++) fscanf(f, "%lf", &stats[c].mean[i]);
        for (int i = 0; i < d*d; i++) fscanf(f, "%lf", &stats[c].cov[i]);
        memcpy(stats[c].inv_cov, stats[c].cov, d*d*sizeof(double));

        LAPACKE_dpotrf(LAPACK_ROW_MAJOR,'U',d,stats[c].inv_cov,d);
        LAPACKE_dpotri(LAPACK_ROW_MAJOR,'U',d,stats[c].inv_cov,d);
    }
    fclose(f);
}

// ---------------- Row evaluation function ----------------
void eval_row(size_t y) {
    if (y % 100 == 0) {
        pthread_mutex_lock(&print_mtx);
        printf("[Row %zu] starting\n", y);
        pthread_mutex_unlock(&print_mtx);
    }

    for (size_t x = 0; x < width; x++) {
        size_t idx = y*width + x;
        out_image[idx] = 0; // keep patch flattening exactly
    }

    if (y % 100 == 0) {
        pthread_mutex_lock(&print_mtx);
        printf("[Row %zu] finished\n", y);
        pthread_mutex_unlock(&print_mtx);
    }
}

// ---------------- Main ----------------
int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,"Usage: %s global_stats.txt image.tif\n",argv[0]);
        return 1;
    }

    pt_init_mtx();
    load_global_stats(argv[1]);

    GDALAllRegister();
    GDALDatasetH ds = GDALOpen(argv[2], GA_ReadOnly);
    if (!ds) { fprintf(stderr,"Failed to open image\n"); exit(1); }

    width = GDALGetRasterXSize(ds);
    height = GDALGetRasterYSize(ds);
    n_bands = GDALGetRasterCount(ds);

    out_image = calloc(width*height, sizeof(unsigned char));

    printf("Starting parallel row classification on %zu rows\n", height);
    parfor_simple(0, height, eval_row);
    printf("Finished parallel row classification\n");

    // ENVI output float32
    GDALDriverH drv = GDALGetDriverByName("ENVI");
    GDALDatasetH ods = GDALCreate(drv,"classification.envi",width,height,1,GDT_Float32,NULL);

    double geo[6];
    GDALGetGeoTransform(ds,geo);
    GDALSetGeoTransform(ods,geo);
    const char *proj = GDALGetProjectionRef(ds);
    GDALSetProjection(ods,proj);

    float *out_float = calloc(width*height, sizeof(float));
    for (size_t i=0; i<width*height; i++) out_float[i] = (float)out_image[i];

    GDALRasterBandH ob = GDALGetRasterBand(ods,1);
    GDALRasterIO(ob,GF_Write,0,0,width,height,out_float,width,height,GDT_Float32,0,0);

    GDALClose(ods);
    GDALClose(ds);

    free(out_float);
    free(out_image);

    for (int c = 0; c < n_classes; c++) {
        free(stats[c].mean);
        free(stats[c].cov);
        free(stats[c].inv_cov);
    }
    free(stats);

    pthread_mutex_destroy(&pt_nxt_j_mtx);
    pthread_mutex_destroy(&print_mtx);

    return 0;
}

