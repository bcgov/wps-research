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
#include <stddef.h>
#include <math.h>
#include <pthread.h>
#include <cblas.h>
#include <lapacke.h>
#include "gdal.h"
#include "cpl_conv.h"

#define PATCH_SIZE 7

typedef struct {
    size_t y_start;
    size_t y_end;
    size_t w;
    size_t h;
    size_t bands;
    double *image;       // padded image
    double **mean;
    double **inv_cov;
    size_t dim;
    size_t job_index;
    size_t total_jobs;
    pthread_mutex_t *print_mutex;
} Job;

typedef struct {
    Job *jobs;
    size_t job_count;
    size_t next_job;
    pthread_mutex_t mutex;
} WorkQueue;

Job* get_next_job(WorkQueue *queue) {
    pthread_mutex_lock(&queue->mutex);
    if (queue->next_job >= queue->job_count) {
        pthread_mutex_unlock(&queue->mutex);
        return NULL;
    }
    Job *job = &queue->jobs[queue->next_job];
    queue->next_job++;
    pthread_mutex_unlock(&queue->mutex);
    return job;
}

void* worker_thread(void *arg) {
    WorkQueue *queue = (WorkQueue*)arg;
    Job *job;
    while((job = get_next_job(queue)) != NULL) {
        printf("[Thread %lu] Starting rows %zu -> %zu\n",
               pthread_self(), job->y_start, job->y_end);
        size_t patch = PATCH_SIZE;
        size_t pad = patch / 2;
        size_t w = job->w;
        size_t h = job->h;
        size_t bands = job->bands;

        for (size_t y = job->y_start; y < job->y_end; y++) {
            for (size_t x = 0; x < w; x++) {
                // Flatten patch safely with padding
                double *vec = malloc(sizeof(double)*job->dim);
                size_t idx=0;
                for (size_t dy=0; dy<patch; dy++) {
                    size_t yy = y+dy;
                    if (yy >= h+2*pad) yy = h+2*pad-1;
                    for (size_t dx=0; dx<patch; dx++) {
                        size_t xx = x+dx;
                        if (xx >= w+2*pad) xx = w+2*pad-1;
                        for (size_t b=0;b<bands;b++) {
                            vec[idx++] = job->image[(yy*(w+2*pad)+xx)*bands+b];
                        }
                    }
                }

                double s0=INFINITY, s1=INFINITY;
                for (int lbl=0; lbl<2; lbl++) {
                    if (job->mean[lbl] != NULL && job->inv_cov[lbl] != NULL) {
                        double *tmp = malloc(sizeof(double)*job->dim);
                        for(size_t i=0;i<job->dim;i++) tmp[i] = vec[i] - job->mean[lbl][i];
                        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                                    job->dim, job->dim, 1.0,
                                    job->inv_cov[lbl], job->dim,
                                    tmp, 1, 0.0, tmp, 1);
                        double score = cblas_ddot(job->dim, tmp,1,tmp,1);
                        free(tmp);
                        if(lbl==0) s0 = score; else s1=score;
                    }
                }

                free(vec);
                // Here you would assign classification to output array
            }

            if(y % 100 == 0) {
                pthread_mutex_lock(job->print_mutex);
                printf("[Thread %lu] Completed row %zu / %zu\n", pthread_self(), y, h);
                pthread_mutex_unlock(job->print_mutex);
            }
        }

        printf("[Thread %lu] Finished rows %zu -> %zu\n",
               pthread_self(), job->y_start, job->y_end);
    }
    return NULL;
}

int main(int argc, char **argv) {
    if(argc < 3) {
        fprintf(stderr,"Usage: %s global_stats.txt image.tif\n", argv[0]);
        return 1;
    }

    printf("Loading global stats from %s\n", argv[1]);
    // TODO: load_global_stats() into Job.mean, Job.inv_cov
    printf("Global stats loaded\n");

    printf("Opening image %s\n", argv[2]);
    GDALAllRegister();
    GDALDatasetH ds = GDALOpen(argv[2], GA_ReadOnly);
    if(!ds) { fprintf(stderr,"Failed to open image\n"); return 1; }

    int w = GDALGetRasterXSize(ds);
    int h = GDALGetRasterYSize(ds);
    int bands = GDALGetRasterCount(ds);
    printf("Image size: %d x %d, bands: %d\n", w, h, bands);

    size_t pad = PATCH_SIZE/2;
    double *image = calloc((w+2*pad)*(h+2*pad)*bands, sizeof(double));
    printf("Allocated padded image array\n");

    // TODO: Read bands into padded image using GDALRasterIO
    printf("Read image into padded array\n");

    // Setup work queue
    size_t nthreads = sysconf(_SC_NPROCESSORS_ONLN);
    printf("Using %zu threads\n", nthreads);
    size_t n_jobs = (size_t)h / 1000 + 1;
    Job *jobs = malloc(sizeof(Job)*n_jobs);
    pthread_mutex_t print_mutex;
    pthread_mutex_init(&print_mutex, NULL);

    size_t rows_per_job = h / n_jobs;
    for(size_t j=0;j<n_jobs;j++) {
        jobs[j].y_start = j*rows_per_job;
        jobs[j].y_end = (j==n_jobs-1) ? h : (j+1)*rows_per_job;
        jobs[j].w = w;
        jobs[j].h = h;
        jobs[j].bands = bands;
        jobs[j].image = image;
        jobs[j].patch_size = PATCH_SIZE;
        jobs[j].job_index = j;
        jobs[j].total_jobs = n_jobs;
        jobs[j].print_mutex = &print_mutex;
    }

    WorkQueue queue;
    queue.jobs = jobs;
    queue.job_count = n_jobs;
    queue.next_job = 0;
    pthread_mutex_init(&queue.mutex, NULL);

    pthread_t *threads = malloc(sizeof(pthread_t)*nthreads);
    printf("Starting threads\n");
    for(size_t t=0;t<nthreads;t++)
        pthread_create(&threads[t], NULL, worker_thread, &queue);

    for(size_t t=0;t<nthreads;t++)
        pthread_join(threads[t], NULL);

    printf("All threads finished\n");

    // TODO: Write ENVI output (32-bit float) preserving map info
    printf("Writing ENVI output\n");

    free(jobs);
    free(image);
    pthread_mutex_destroy(&print_mutex);
    printf("Finished classify program\n");
    return 0;
}

