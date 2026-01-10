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
#include <sys/sysinfo.h>
#include <cblas.h>
#include <lapacke.h>
#include "gdal.h"
#include "cpl_conv.h"

#define PATCH_SIZE 7
#define MIN_POLY_DIMENSION 15

typedef struct {
    size_t dim;
    double *mean;
    double *cov;
    double *inv_cov;
} ClassStats;

typedef struct {
    size_t y_start;
    size_t y_end;
    size_t w;
    size_t h;
    size_t bands;
    double *image;
    unsigned char *out;
    size_t job_index;
    size_t total_jobs;
    pthread_mutex_t *print_mutex;
    ClassStats stats[2];
} Job;

typedef struct JobNode {
    Job *job;
    struct JobNode *next;
} JobNode;

typedef struct {
    JobNode *head;
    JobNode *tail;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} JobQueue;

// ---------------- Job Queue ----------------
void init_queue(JobQueue *q) {
    q->head = q->tail = NULL;
    pthread_mutex_init(&q->mutex,NULL);
    pthread_cond_init(&q->cond,NULL);
}

void enqueue(JobQueue *q, Job *job) {
    JobNode *node = malloc(sizeof(JobNode));
    node->job = job;
    node->next = NULL;
    pthread_mutex_lock(&q->mutex);
    if(q->tail) q->tail->next = node;
    else q->head = node;
    q->tail = node;
    pthread_cond_signal(&q->cond);
    pthread_mutex_unlock(&q->mutex);
}

Job* dequeue(JobQueue *q) {
    pthread_mutex_lock(&q->mutex);
    while(!q->head) pthread_cond_wait(&q->cond,&q->mutex);
    JobNode *node = q->head;
    q->head = node->next;
    if(!q->head) q->tail = NULL;
    pthread_mutex_unlock(&q->mutex);
    Job *job = node->job;
    free(node);
    return job;
}

// ---------------- Utility ----------------
void print_progress(Job *job, size_t done_rows) {
    if(done_rows % 100 == 0) {
        double pct = 100.0 * done_rows / job->h;
        pthread_mutex_lock(job->print_mutex);
        printf("[Job %zu/%zu] Progress: %zu/%zu rows (%.1f%%)\n",
               job->job_index+1, job->total_jobs, done_rows, job->h, pct);
        pthread_mutex_unlock(job->print_mutex);
    }
}

// ---------------- Worker ----------------
void* worker_thread(void *arg) {
    JobQueue *q = (JobQueue*)arg;
    while(1) {
        Job *job = dequeue(q);
        if(!job) break;

        pthread_mutex_lock(job->print_mutex);
        printf("[Thread %lu] Starting rows %zu -> %zu\n",
               pthread_self(), job->y_start, job->y_end);
        pthread_mutex_unlock(job->print_mutex);

        size_t pad = PATCH_SIZE/2;
        for(size_t y=job->y_start; y<job->y_end; y++) {
            for(size_t x=0; x<job->w; x++) {
                double best_score = INFINITY;
                int best_label = 0;
                for(int lbl=0; lbl<=1; lbl++) {
                    ClassStats *s = &job->stats[lbl];
                    if(!s->mean) continue;

                    size_t vec_size = s->dim;
                    double *vec = malloc(vec_size*sizeof(double));
                    // flatten patch
                    size_t idx=0;
                    for(size_t dy=0; dy<PATCH_SIZE; dy++) {
                        size_t yy = y + dy - pad;
                        if(yy >= job->h) yy = job->h-1;
                        for(size_t dx=0; dx<PATCH_SIZE; dx++) {
                            size_t xx = x + dx - pad;
                            if(xx >= job->w) xx = job->w-1;
                            for(size_t b=0;b<job->bands;b++)
                                vec[idx++] = job->image[(yy*job->w+xx)*job->bands+b];
                        }
                    }

                    double *tmp = malloc(vec_size*sizeof(double));
                    for(size_t i=0;i<vec_size;i++)
                        tmp[i] = vec[i]-s->mean[i];
                    double score;
                    cblas_ddot(vec_size,tmp,1,tmp,1); // simplified Mahalanobis
                    score = cblas_ddot(vec_size,tmp,1,tmp,1);
                    free(tmp); free(vec);

                    if(score<best_score) { best_score=score; best_label=lbl; }
                }
                job->out[y*job->w + x] = (unsigned char)best_label;
            }
            print_progress(job,y-job->y_start);
        }

        pthread_mutex_lock(job->print_mutex);
        printf("[Thread %lu] Finished rows %zu -> %zu\n",
               pthread_self(), job->y_start, job->y_end);
        pthread_mutex_unlock(job->print_mutex);
    }
    return NULL;
}

// ---------------- Global Stats ----------------
void load_global_stats(const char *path, ClassStats stats[2]) {
    FILE *f = fopen(path,"r");
    if(!f) { fprintf(stderr,"Failed to open %s\n",path); exit(1); }

    char line[4096];
    while(fgets(line,sizeof(line),f)) {
        int cls;
        if(sscanf(line,"CLASS %d",&cls)==1) continue;
        if(strncmp(line,"DIM",3)==0) {
            size_t d; sscanf(line,"DIM %zu",&d);
            stats[cls].dim=d;
            stats[cls].mean = calloc(d,sizeof(double));
            stats[cls].cov = calloc(d*d,sizeof(double));
            stats[cls].inv_cov = calloc(d*d,sizeof(double));
        } else if(strncmp(line,"MEAN",4)==0) {
            for(size_t i=0;i<stats[cls].dim;i++)
                fscanf(f,"%lf",&stats[cls].mean[i]);
        } else if(strncmp(line,"COV",3)==0) {
            for(size_t i=0;i<stats[cls].dim*stats[cls].dim;i++)
                fscanf(f,"%lf",&stats[cls].cov[i]);
            memcpy(stats[cls].inv_cov,stats[cls].cov,sizeof(double)*stats[cls].dim*stats[cls].dim);
            LAPACKE_dpotrf(LAPACK_ROW_MAJOR,'U',stats[cls].dim,stats[cls].inv_cov,stats[cls].dim);
            LAPACKE_dpotri(LAPACK_ROW_MAJOR,'U',stats[cls].dim,stats[cls].inv_cov,stats[cls].dim);
        }
    }
    fclose(f);
}

// ---------------- Main ----------------
int main(int argc,char **argv) {
    if(argc!=3) { fprintf(stderr,"Usage: %s global_stats.txt image.tif\n",argv[0]); return 1; }

    const char *stats_file = argv[1];
    const char *img_file = argv[2];

    GDALAllRegister();
    GDALDatasetH ds = GDALOpen(img_file,GA_ReadOnly);
    if(!ds) { fprintf(stderr,"Failed to open image\n"); return 1; }

    int w = GDALGetRasterXSize(ds);
    int h = GDALGetRasterYSize(ds);
    int bands = GDALGetRasterCount(ds);
    double *image = calloc((size_t)w*(size_t)h*(size_t)bands,sizeof(double));
    for(int b=0;b<bands;b++) {
        GDALRasterBandH rb = GDALGetRasterBand(ds,b+1);
        double *tmp = malloc(sizeof(double)*w*h);
        GDALRasterIO(rb,GF_Read,0,0,w,h,tmp,w,h,GDT_Float64,0,0);
        for(int y=0;y<h;y++)
            for(int x=0;x<w;x++)
                image[(y*w+x)*bands+b] = tmp[y*w+x];
        free(tmp);
    }

    ClassStats stats[2] = {0};
    printf("Loading global stats...\n");
    load_global_stats(stats_file,stats);
    printf("Global stats loaded.\n");

    size_t n_threads = get_nprocs();
    printf("Using %zu threads\n", n_threads);

    JobQueue q; init_queue(&q);

    size_t n_jobs = (size_t)h; // one job per row for simplicity
    Job *jobs = calloc(n_jobs,sizeof(Job));

    for(size_t j=0;j<n_jobs;j++) {
        jobs[j].y_start = j;
        jobs[j].y_end = j+1;
        jobs[j].w = w;
        jobs[j].h = h;
        jobs[j].bands = bands;
        jobs[j].image = image;
        jobs[j].out = calloc(w,sizeof(unsigned char));
        jobs[j].job_index = j;
        jobs[j].total_jobs = n_jobs;
        jobs[j].print_mutex = malloc(sizeof(pthread_mutex_t));
        pthread_mutex_init(jobs[j].print_mutex,NULL);
        jobs[j].stats[0] = stats[0];
        jobs[j].stats[1] = stats[1];
        enqueue(&q,&jobs[j]);
    }

    pthread_t *threads = malloc(sizeof(pthread_t)*n_threads);
    for(size_t t=0;t<n_threads;t++)
        pthread_create(&threads[t],NULL,worker_thread,&q);

    for(size_t t=0;t<n_threads;t++)
        pthread_join(threads[t],NULL);

    printf("All threads finished.\n");

    // ENVI output
    GDALDriverH drv = GDALGetDriverByName("ENVI");
    GDALDatasetH ods = GDALCreate(drv,"classification.bin",w,h,1,GDT_Float32,NULL);
    double geo[6]; GDALGetGeoTransform(ds,geo);
    GDALSetGeoTransform(ods,geo);
    GDALSetProjection(ods,GDALGetProjectionRef(ds));
    float *out_float = malloc(sizeof(float)*w*h);
    for(size_t i=0;i<w*h;i++) out_float[i] = (float)jobs[i].out[0]; // flattened
    GDALRasterBandH ob = GDALGetRasterBand(ods,1);
    GDALRasterIO(ob,GF_Write,0,0,w,h,out_float,w,h,GDT_Float32,0,0);
    GDALClose(ods);
    GDALClose(ds);

    printf("Saved classification.bin\n");

    free(image);
    free(out_float);
    for(size_t j=0;j<n_jobs;j++) {
        free(jobs[j].out);
        pthread_mutex_destroy(jobs[j].print_mutex);
        free(jobs[j].print_mutex);
    }
    free(jobs);
    return 0;
}

