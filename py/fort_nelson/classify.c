/*
gcc -O3 -Wall classify.c -o classify -I/usr/local/include -L/usr/local/lib -lgdal -llapacke -llapack -lcblas -lblas -lpthread -lm

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
#include <stddef.h>
#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>

#include <cblas.h>
#include <lapacke.h>
#include "gdal.h"
#include "cpl_conv.h"

#define MAX_CLASSES 2

/* ---------------- constants ---------------- */
#define PATCH_SIZE 7
#define MIN_POLY_DIMENSION 15

/* ---------------- data structures ---------------- */
typedef struct {
    int dim;
    double *mean;
    double *cov;
    double *inv_cov;
} ClassStats;

typedef struct {
    int job_index;
    int y;
    int w;
    int h;
    double *image; /* flattened h*w*bands array */
    ClassStats *stats;
    int bands;
    unsigned char *out;
} Job;

/* ---------------- work queue ---------------- */
typedef struct Node {
    Job *job;
    struct Node *next;
} Node;

typedef struct {
    Node *head;
    Node *tail;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int count;
    int done;
} WorkQueue;

WorkQueue queue;
pthread_mutex_t print_mutex = PTHREAD_MUTEX_INITIALIZER;
int total_jobs = 0;
int jobs_done = 0;
struct timeval start_time;

/* ---------------- queue functions ---------------- */
void init_queue(WorkQueue *q) {
    q->head = q->tail = NULL;
    pthread_mutex_init(&q->mutex, NULL);
    pthread_cond_init(&q->cond, NULL);
    q->count = 0;
    q->done = 0;
}

void push_queue(WorkQueue *q, Job *job) {
    Node *n = malloc(sizeof(Node));
    n->job = job;
    n->next = NULL;
    pthread_mutex_lock(&q->mutex);
    if (!q->tail) q->head = n;
    else q->tail->next = n;
    q->tail = n;
    q->count++;
    pthread_cond_signal(&q->cond);
    pthread_mutex_unlock(&q->mutex);
}

Job *pop_queue(WorkQueue *q) {
    pthread_mutex_lock(&q->mutex);
    while (!q->head && !q->done) {
        pthread_cond_wait(&q->cond, &q->mutex);
    }
    if (q->done && !q->head) {
        pthread_mutex_unlock(&q->mutex);
        return NULL;
    }
    Node *n = q->head;
    Job *job = n->job;
    q->head = n->next;
    if (!q->head) q->tail = NULL;
    q->count--;
    free(n);
    pthread_mutex_unlock(&q->mutex);
    return job;
}

void finish_queue(WorkQueue *q) {
    pthread_mutex_lock(&q->mutex);
    q->done = 1;
    pthread_cond_broadcast(&q->cond);
    pthread_mutex_unlock(&q->mutex);
}

/* ---------------- timing ---------------- */
double elapsed_seconds() {
    struct timeval now;
    gettimeofday(&now, NULL);
    return (now.tv_sec - start_time.tv_sec) + (now.tv_usec - start_time.tv_usec)/1e6;
}

/* ---------------- global stats loading ---------------- */
ClassStats *load_global_stats(const char *path, int *bands) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", path);
        exit(1);
    }

    ClassStats *stats = calloc(MAX_CLASSES, sizeof(ClassStats));
    char line[4096];
    for (int cls=0; cls<MAX_CLASSES; cls++) {
        stats[cls].dim = 0;
        stats[cls].mean = stats[cls].cov = stats[cls].inv_cov = NULL;
    }

    while (fgets(line, sizeof(line), f)) {
        int cls;
        if (sscanf(line, "CLASS %d", &cls) == 1) {
            // read DIM next line
            fgets(line, sizeof(line), f);
            int d;
            if (sscanf(line, "DIM %d", &d) == 1) {
                stats[cls].dim = d;
                stats[cls].mean = calloc(d, sizeof(double));
                stats[cls].cov  = calloc(d*d, sizeof(double));
                stats[cls].inv_cov = calloc(d*d, sizeof(double));
            }
        } else if (strncmp(line, "MEAN", 4) == 0) {
            for (int i=0; i<stats[0].dim; i++)
                fscanf(f, "%lf", &stats[0].mean[i]);
        } else if (strncmp(line, "COV", 3) == 0) {
            for (int cls=0; cls<MAX_CLASSES; cls++) {
                for (int i=0; i<stats[cls].dim*stats[cls].dim; i++)
                    fscanf(f, "%lf", &stats[cls].cov[i]);
                memcpy(stats[cls].inv_cov, stats[cls].cov, sizeof(double)*stats[cls].dim*stats[cls].dim);
                int d = stats[cls].dim;
                int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', d, stats[cls].inv_cov, d);
                if (info==0) LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'U', d, stats[cls].inv_cov, d);
                else LAPACKE_dgetrf(LAPACK_ROW_MAJOR, d, d, stats[cls].inv_cov, d, NULL);
            }
        }
    }
    fclose(f);
    *bands = stats[0].dim / (PATCH_SIZE*PATCH_SIZE);
    return stats;
}

/* ---------------- classification ---------------- */
double classify_vector(double *v, ClassStats *stats) {
    double s0 = INFINITY, s1 = INFINITY;
    for (int cls=0; cls<MAX_CLASSES; cls++) {
        if (!stats[cls].mean) continue;
        int d = stats[cls].dim;
        double *tmp = malloc(sizeof(double)*d);
        for (int i=0;i<d;i++) tmp[i] = v[i]-stats[cls].mean[i];
        double score = cblas_ddot(d, tmp, 1, tmp, 1);
        free(tmp);
        if (cls==0) s0=score; else s1=score;
    }
    return (s1 < s0) ? 1.0 : 0.0;
}

void *worker_func(void *arg) {
    (void)arg;
    Job *job;
    while ((job = pop_queue(&queue))) {
        for (int x=0;x<job->w;x++) {
            double patch[PATCH_SIZE*PATCH_SIZE*job->bands];
            for (int by=0; by<PATCH_SIZE; by++) {
                for (int bx=0; bx<PATCH_SIZE; bx++) {
                    for (int b=0;b<job->bands;b++) {
                        int yy = job->y+by;
                        int xx = x+bx;
                        patch[by*PATCH_SIZE*job->bands + bx*job->bands + b] = job->image[(yy*job->w+xx)*job->bands + b];
                    }
                }
            }
            job->out[job->y*job->w+x] = (unsigned char)classify_vector(patch, job->stats);
        }

        pthread_mutex_lock(&print_mutex);
        jobs_done++;
        if (job->job_index % 100 == 0) {
            double frac = (double)jobs_done/total_jobs;
            double eta = elapsed_seconds() * (1-frac)/frac;
            printf("\rProcessed %d/%d rows (ETA %.1fs)", jobs_done, total_jobs, eta);
            fflush(stdout);
        }
        pthread_mutex_unlock(&print_mutex);
        free(job);
    }
    return NULL;
}

/* ---------------- main ---------------- */
int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s global_stats.txt image.tif\n", argv[0]);
        return 1;
    }

    const char *stats_path = argv[1];
    const char *image_path = argv[2];

    GDALAllRegister();
    int bands;
    ClassStats *stats = load_global_stats(stats_path, &bands);

    GDALDatasetH ds = GDALOpen(image_path, GA_ReadOnly);
    if (!ds) { fprintf(stderr, "Failed to open image\n"); return 1; }

    int w = GDALGetRasterXSize(ds);
    int h = GDALGetRasterYSize(ds);

    double *image = calloc(w*h*bands, sizeof(double));
    for (int b=0;b<bands;b++) {
        GDALRasterBandH rb = GDALGetRasterBand(ds, b+1);
        double *tmp = malloc(sizeof(double)*w*h);
        GDALRasterIO(rb, GF_Read, 0,0, w,h, tmp, w,h, GDT_Float64, 0,0);
        for (int i=0;i<w*h;i++) image[i*bands+b] = tmp[i];
        free(tmp);
    }

    unsigned char *out = calloc(w*h, sizeof(unsigned char));

    gettimeofday(&start_time, NULL);
    total_jobs = h;

    init_queue(&queue);
    int nthreads = sysconf(_SC_NPROCESSORS_ONLN);
    pthread_t threads[nthreads];

    for (int t=0;t<nthreads;t++) pthread_create(&threads[t], NULL, worker_func, NULL);
    for (int y=0;y<h;y++) {
        Job *job = malloc(sizeof(Job));
        job->job_index = y;
        job->y = y;
        job->w = w;
        job->h = h;
        job->image = image;
        job->stats = stats;
        job->bands = bands;
        job->out = out;
        push_queue(&queue, job);
    }

    finish_queue(&queue);
    for (int t=0;t<nthreads;t++) pthread_join(threads[t], NULL);
    printf("\n");

    GDALDriverH drv = GDALGetDriverByName("ENVI");
    GDALDatasetH ods = GDALCreate(drv, "classification.bin", w, h, 1, GDT_Byte, NULL);
    GDALRasterIO(GDALGetRasterBand(ods,1), GF_Write, 0,0, w,h, out, w,h, GDT_Byte,0,0);
    GDALClose(ods);
    GDALClose(ds);

    free(image);
    free(out);
    for (int cls=0;cls<MAX_CLASSES;cls++) {
        free(stats[cls].mean);
        free(stats[cls].cov);
        free(stats[cls].inv_cov);
    }
    free(stats);

    return 0;
}

