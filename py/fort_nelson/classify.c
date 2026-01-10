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
#include <pthread.h>
#include <time.h>
#include <stddef.h>

#include "cblas.h"
#include "lapacke.h"
#include "gdal.h"
#include "cpl_conv.h"

#define PATCH_SIZE 1  // keep patch flattening exactly as before

typedef struct {
    int dim;
    double *mean;
    double *cov;
    double *inv_cov;
} ClassStats;

typedef struct {
    size_t y_start;
    size_t y_end;
} Job;

typedef struct Node {
    Job job;
    struct Node *next;
} Node;

typedef struct {
    Node *head;
    Node *tail;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    size_t total_jobs;
    size_t jobs_done;
} JobQueue;

JobQueue queue;
size_t width, height, n_bands;
unsigned char *out_image;
ClassStats *stats;
int n_classes;
pthread_mutex_t print_mutex;

void enqueue(Job job) {
    Node *node = malloc(sizeof(Node));
    node->job = job;
    node->next = NULL;

    pthread_mutex_lock(&queue.mutex);
    if(queue.tail) queue.tail->next = node;
    else queue.head = node;
    queue.tail = node;
    pthread_cond_signal(&queue.cond);
    pthread_mutex_unlock(&queue.mutex);
}

int dequeue(Job *job) {
    pthread_mutex_lock(&queue.mutex);
    while(!queue.head) {
        pthread_cond_wait(&queue.cond, &queue.mutex);
    }
    Node *node = queue.head;
    *job = node->job;
    queue.head = node->next;
    if(!queue.head) queue.tail = NULL;
    free(node);
    pthread_mutex_unlock(&queue.mutex);
    return 1;
}

void load_global_stats(const char *path) {
    FILE *f = fopen(path, "r");
    if(!f) { perror("Failed to open global stats"); exit(1); }

    fscanf(f, "%d", &n_classes);
    stats = calloc(n_classes, sizeof(ClassStats));

    for(int c = 0; c < n_classes; c++) {
        fscanf(f, "%d", &stats[c].dim);
        int d = stats[c].dim;
        stats[c].mean = calloc(d, sizeof(double));
        stats[c].cov  = calloc(d*d, sizeof(double));
        stats[c].inv_cov = calloc(d*d, sizeof(double));
        for(int i = 0; i < d; i++) fscanf(f, "%lf", &stats[c].mean[i]);
        for(int i = 0; i < d*d; i++) fscanf(f, "%lf", &stats[c].cov[i]);
        memcpy(stats[c].inv_cov, stats[c].cov, d*d*sizeof(double));
        LAPACKE_dpotrf(LAPACK_ROW_MAJOR,'U',d,stats[c].inv_cov,d);
        LAPACKE_dpotri(LAPACK_ROW_MAJOR,'U',d,stats[c].inv_cov,d);
    }
    fclose(f);
}

void *worker_thread(void *arg) {
    size_t rows_done = 0;
    while(1) {
        Job job;
        if(!dequeue(&job)) break;

        pthread_mutex_lock(&print_mutex);
        printf("[Thread %lu] Starting rows %zu -> %zu\n", pthread_self(), job.y_start, job.y_end);
        pthread_mutex_unlock(&print_mutex);

        for(size_t y = job.y_start; y < job.y_end; y++) {
            // loop over row
            for(size_t x = 0; x < width; x++) {
                // here preserve patch flattening exactly
                size_t idx = y*width + x;
                out_image[idx] = 0; // dummy classification logic placeholder
            }
            rows_done++;

            if(rows_done % 100 == 0) {
                pthread_mutex_lock(&print_mutex);
                double pct = 100.0*queue.jobs_done/queue.total_jobs;
                printf("[Thread %lu] Progress: %zu/%zu rows (%.2f%%)\n", pthread_self(),
                       queue.jobs_done, queue.total_jobs, pct);
                pthread_mutex_unlock(&print_mutex);
            }
        }

        pthread_mutex_lock(&queue.mutex);
        queue.jobs_done += job.y_end - job.y_start;
        pthread_mutex_unlock(&queue.mutex);

        pthread_mutex_lock(&print_mutex);
        printf("[Thread %lu] Finished rows %zu -> %zu\n", pthread_self(), job.y_start, job.y_end);
        pthread_mutex_unlock(&print_mutex);
    }
    return NULL;
}

int main(int argc, char **argv) {
    if(argc < 3) { fprintf(stderr,"Usage: %s global_stats.txt image.tif\n",argv[0]); return 1; }

    pthread_mutex_init(&print_mutex,NULL);

    load_global_stats(argv[1]);

    GDALAllRegister();
    GDALDatasetH ds = GDALOpen(argv[2], GA_ReadOnly);
    if(!ds) { fprintf(stderr,"Failed to open image\n"); exit(1); }

    width  = GDALGetRasterXSize(ds);
    height = GDALGetRasterYSize(ds);
    n_bands = GDALGetRasterCount(ds);
    out_image = calloc(width*height,sizeof(unsigned char));

    size_t nthreads = sysconf(_SC_NPROCESSORS_ONLN);
    pthread_t threads[nthreads];

    // enqueue jobs
    queue.total_jobs = height;
    queue.jobs_done = 0;
    queue.head = queue.tail = NULL;
    pthread_mutex_init(&queue.mutex,NULL);
    pthread_cond_init(&queue.cond,NULL);

    size_t patch = PATCH_SIZE;
    for(size_t y = 0; y < height; y += patch) {
        Job j;
        j.y_start = y;
        j.y_end = (y+patch < height) ? y+patch : height;
        enqueue(j);
    }

    for(size_t t = 0; t < nthreads; t++)
        pthread_create(&threads[t],NULL,worker_thread,NULL);

    for(size_t t = 0; t < nthreads; t++)
        pthread_join(threads[t],NULL);

    // write ENVI output float32
    GDALDriverH drv = GDALGetDriverByName("ENVI");
    GDALDatasetH ods = GDALCreate(drv,"classification.envi",width,height,1,GDT_Float32,NULL);

    double geo[6];
    GDALGetGeoTransform(ds,geo);
    GDALSetGeoTransform(ods,geo);
    const char *proj = GDALGetProjectionRef(ds);
    GDALSetProjection(ods,proj);

    float *out_float = calloc(width*height,sizeof(float));
    for(size_t i=0;i<width*height;i++) out_float[i]=(float)out_image[i];

    GDALRasterBandH ob = GDALGetRasterBand(ods,1);
    GDALRasterIO(ob,GF_Write,0,0,width,height,out_float,width,height,GDT_Float32,0,0);

    GDALClose(ods);
    GDALClose(ds);
    free(out_float);
    free(out_image);

    for(int c =0;c<n_classes;c++){
        free(stats[c].mean);
        free(stats[c].cov);
        free(stats[c].inv_cov);
    }
    free(stats);

    pthread_mutex_destroy(&queue.mutex);
    pthread_cond_destroy(&queue.cond);
    pthread_mutex_destroy(&print_mutex);

    return 0;
}

