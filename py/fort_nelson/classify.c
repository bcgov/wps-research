/*
gcc -O3 -Wall classify.c -o classify -I/usr/local/include -L/usr/local/lib -lgdal -llapacke -llapack -lcblas -lblas -lpthread -lm

 Gaussian patch classifier (C version)
 Uses:
  - GDAL (C API)
  - BLAS / LAPACK (CBLAS + LAPACKE)
  - pthreads
*/

/* unchanged includes and constants */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <sys/sysinfo.h>
#include "gdal.h"
#include "cpl_conv.h"
#include "cpl_error.h"
#include <cblas.h>
#include <lapacke.h>

#define PATCH_SIZE 7
#define MAX_CLASSES 2

typedef struct {
    int dim;
    double *mean;
    double *cov;
    double *inv_cov;
} ClassStats;

typedef struct {
    int w, h, bands;
    double *image;
    unsigned char *out;
    ClassStats *stats;
} Job;

typedef struct JobNode { int y; struct JobNode *next; } JobNode;

typedef struct { JobNode *head; JobNode *tail; pthread_mutex_t mutex; } WorkQueue;

WorkQueue queue = {NULL,NULL,PTHREAD_MUTEX_INITIALIZER};
pthread_mutex_t print_mutex = PTHREAD_MUTEX_INITIALIZER;
int jobs_done = 0, total_jobs = 0;
time_t start_time;

void enqueue_job(int y) {
    JobNode *n = malloc(sizeof(JobNode));
    n->y = y; n->next=NULL;
    pthread_mutex_lock(&queue.mutex);
    if(queue.tail) queue.tail->next = n;
    else queue.head = n;
    queue.tail = n;
    pthread_mutex_unlock(&queue.mutex);
}

int dequeue_job() {
    pthread_mutex_lock(&queue.mutex);
    if(!queue.head) { pthread_mutex_unlock(&queue.mutex); return -1; }
    JobNode *n = queue.head;
    int y = n->y;
    queue.head = n->next;
    if(!queue.head) queue.tail=NULL;
    pthread_mutex_unlock(&queue.mutex);
    free(n);
    return y;
}

ClassStats *load_global_stats(const char *path, int *n_classes) {
    FILE *f = fopen(path,"r");
    if(!f){ fprintf(stderr,"Failed to open %s\n", path); exit(1); }

    ClassStats *stats = calloc(MAX_CLASSES,sizeof(ClassStats));
    char line[8192];
    int cls=-1, dim=0;

    while(fgets(line,sizeof(line),f)){
        if(sscanf(line,"CLASS %d",&cls)==1) continue;
        if(cls<0) continue; /* wait for a valid class */
        if(sscanf(line,"DIM %d",&dim)==1){
            stats[cls].dim = dim;
            stats[cls].mean = calloc(dim,sizeof(double));
            stats[cls].cov  = calloc(dim*dim,sizeof(double));
            stats[cls].inv_cov = calloc(dim*dim,sizeof(double));
        }
        if(strncmp(line,"MEAN",4)==0)
            for(int i=0;i<stats[cls].dim;i++) fscanf(f,"%lf",&stats[cls].mean[i]);
        if(strncmp(line,"COV",3)==0){
            for(int i=0;i<stats[cls].dim*stats[cls].dim;i++) fscanf(f,"%lf",&stats[cls].cov[i]);
            memcpy(stats[cls].inv_cov, stats[cls].cov,sizeof(double)*stats[cls].dim*stats[cls].dim);
            int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR,'U',stats[cls].dim,stats[cls].inv_cov,stats[cls].dim);
            if(info==0) LAPACKE_dpotri(LAPACK_ROW_MAJOR,'U',stats[cls].inv_cov,stats[cls].dim);
        }
    }
    fclose(f);
    *n_classes = MAX_CLASSES;
    return stats;
}

int classify_vector(double *vec, ClassStats *stats) {
    double best_score = INFINITY;
    int best_class=0;
    for(int cls=0;cls<MAX_CLASSES;cls++){
        if(!stats[cls].mean) continue;
        int d = stats[cls].dim;
        double tmp[d]; /* Python ravel flattening preserved */
        for(int i=0;i<d;i++) tmp[i]=vec[i]-stats[cls].mean[i];
        double score = cblas_ddot(d, tmp,1, stats[cls].inv_cov,1);
        if(score<best_score){ best_score=score; best_class=cls; }
    }
    return best_class;
}

void *worker_thread(void *arg) {
    Job *job = (Job*)arg;
    double patch[PATCH_SIZE*PATCH_SIZE*job->bands];

    while(1){
        int y = dequeue_job();
        if(y<0) break;

        for(int x=0;x<job->w;x++){
            int idx=0;
            for(int by=0;by<PATCH_SIZE;by++){
                int yy = y+by-PATCH_SIZE/2;
                if(yy<0) yy=0; if(yy>=job->h) yy=job->h-1;
                for(int bx=0;bx<PATCH_SIZE;bx++){
                    int xx = x+bx-PATCH_SIZE/2;
                    if(xx<0) xx=0; if(xx>=job->w) xx=job->w-1;
                    for(int b=0;b<job->bands;b++){
                        patch[idx++] = job->image[(yy*job->w+xx)*job->bands+b];
                    }
                }
            }
            job->out[y*job->w+x] = classify_vector(patch, job->stats);
        }

        pthread_mutex_lock(&print_mutex);
        jobs_done++;
        if(y %100 ==0 || jobs_done==total_jobs){
            double elapsed = difftime(time(NULL),start_time);
            double eta = elapsed/jobs_done*(total_jobs-jobs_done);
            printf("\rCompleted %d/%d rows, ETA %.1fs", jobs_done,total_jobs,eta);
            fflush(stdout);
        }
        pthread_mutex_unlock(&print_mutex);
    }
    return NULL;
}

int main(int argc, char **argv){
    if(argc!=3){ fprintf(stderr,"Usage: %s global_stats.txt image.tif\n",argv[0]); return 1; }

    GDALAllRegister();
    int n_classes;
    ClassStats *stats = load_global_stats(argv[1], &n_classes);

    GDALDatasetH ds = GDALOpen(argv[2], GA_ReadOnly);
    if(!ds){ fprintf(stderr,"Failed to open image\n"); return 1; }
    int w = GDALGetRasterXSize(ds);
    int h = GDALGetRasterYSize(ds);
    int bands = GDALGetRasterCount(ds);

    double *image = calloc(w*h*bands,sizeof(double));
    for(int b=0;b<bands;b++){
        GDALRasterBandH rb = GDALGetRasterBand(ds,b+1);
        double *tmp = malloc(sizeof(double)*w*h);
        GDALRasterIO(rb, GF_Read,0,0,w,h,tmp,w,h,GDT_Float64,0,0);
        for(int i=0;i<w*h;i++) image[i*bands+b] = tmp[i];
        free(tmp);
    }

    unsigned char *out = calloc(w*h,sizeof(unsigned char));

    total_jobs=h; start_time=time(NULL);
    for(int y=0;y<h;y++) enqueue_job(y);

    int nthreads=get_nprocs();
    pthread_t *threads = malloc(sizeof(pthread_t)*nthreads);
    Job job = {w,h,bands,image,out,stats};
    for(int t=0;t<nthreads;t++) pthread_create(&threads[t],NULL,worker_thread,&job);
    for(int t=0;t<nthreads;t++) pthread_join(threads[t],NULL);
    printf("\n");

    GDALDriverH drv = GDALGetDriverByName("ENVI");
    GDALDatasetH ods = GDALCreate(drv,"classification.bin",w,h,1,GDT_Byte,NULL);
    GDALRasterIO(GDALGetRasterBand(ods,1), GF_Write,0,0,w,h,out,w,h,GDT_Byte,0,0);
    GDALClose(ods);
    GDALClose(ds);

    free(image); free(out); free(threads);
    for(int i=0;i<n_classes;i++){
        free(stats[i].mean); free(stats[i].cov); free(stats[i].inv_cov);
    }
    free(stats);

    return 0;
}

