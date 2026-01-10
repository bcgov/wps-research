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
#include <sys/sysinfo.h>
#include "gdal.h"
#include "cpl_conv.h"
#include "cblas.h"
#include "lapacke.h"

/* ---------------- constants ---------------- */
#define PATCH_SIZE 7
#define MIN_POLY_DIMENSION 15

typedef struct {
    int dim;
    double *mean;
    double *cov;
    double *inv_cov;
} ClassStats;

typedef struct {
    int y_start, y_end;
    int w, h, d;
    double *padded;
    float *out;
    ClassStats *stats;
} Job;

pthread_mutex_t print_mutex = PTHREAD_MUTEX_INITIALIZER;
int total_jobs = 0;
int finished_jobs = 0;

void print_progress() {
    pthread_mutex_lock(&print_mutex);
    double pct = 100.0 * finished_jobs / total_jobs;
    printf("\rClassifying: %d/%d (%.1f%%) ETA approx %.1fs",
           finished_jobs, total_jobs, pct,
           (pct > 0 ? (finished_jobs / pct - finished_jobs / 100.0) : 0.0));
    fflush(stdout);
    pthread_mutex_unlock(&print_mutex);
}

void *classify_rows(void *arg) {
    Job *job = (Job*)arg;

    pthread_mutex_lock(&print_mutex);
    printf("[Thread %lu] Starting rows %d -> %d\n", pthread_self(), job->y_start, job->y_end-1);
    pthread_mutex_unlock(&print_mutex);

    int pad = PATCH_SIZE/2;
    for(int y=job->y_start; y<job->y_end; y++) {
        for(int x=0; x<job->w; x++) {
            double patch[PATCH_SIZE*PATCH_SIZE*job->d];
            int idx=0;
            for(int dy=0; dy<PATCH_SIZE; dy++)
                for(int dx=0; dx<PATCH_SIZE; dx++)
                    for(int b=0; b<job->d; b++)
                        patch[idx++] = job->padded[((y+dy)* (job->w+2*pad) + (x+dx))*job->d + b];

            double s0 = INFINITY, s1 = INFINITY;

            if(job->stats[0].mean) {
                double tmp[job->d*job->d];
                cblas_dgemv(CblasRowMajor,CblasNoTrans,job->d,job->d,1.0,
                            job->stats[0].inv_cov,job->d,patch,1,0.0,tmp,1);
                s0 = cblas_ddot(job->d,patch,1,tmp,1);
            }

            if(job->stats[1].mean) {
                double tmp[job->d*job->d];
                cblas_dgemv(CblasRowMajor,CblasNoTrans,job->d,job->d,1.0,
                            job->stats[1].inv_cov,job->d,patch,1,0.0,tmp,1);
                s1 = cblas_ddot(job->d,patch,1,tmp,1);
            }

            job->out[y*job->w+x] = (float)(s1<s0 ? 1 : 0);
        }

        pthread_mutex_lock(&print_mutex);
        finished_jobs++;
        if(finished_jobs % 100 == 0 || finished_jobs==total_jobs) {
            print_progress();
            printf(" [Thread %lu working on row %d]\n", pthread_self(), y);
        }
        pthread_mutex_unlock(&print_mutex);
    }

    pthread_mutex_lock(&print_mutex);
    printf("[Thread %lu] Finished rows %d -> %d\n", pthread_self(), job->y_start, job->y_end-1);
    pthread_mutex_unlock(&print_mutex);

    return NULL;
}

int load_global_stats(const char *path, ClassStats stats[2]) {
    FILE *f = fopen(path,"r");
    if(!f) { fprintf(stderr,"Failed to open %s\n",path); exit(1); }

    char line[4096];
    for(int cls=0; cls<2; cls++) {
        stats[cls].dim = 0;
        stats[cls].mean = NULL;
        stats[cls].cov = NULL;
        stats[cls].inv_cov = NULL;
    }

    int cls=-1;
    while(fgets(line,sizeof(line),f)) {
        if(sscanf(line,"CLASS %d",&cls)==1) {
            printf("[Global Stats] Loading CLASS %d\n", cls);
            continue;
        }
        if(cls<0 || cls>1) continue;

        if(strncmp(line,"DIM",3)==0) {
            int d;
            if(sscanf(line,"DIM %d",&d)==1) {
                stats[cls].dim=d;
                stats[cls].mean = calloc(d,sizeof(double));
                stats[cls].cov = calloc(d*d,sizeof(double));
                stats[cls].inv_cov = calloc(d*d,sizeof(double));
                printf("[Global Stats] CLASS %d DIM %d\n",cls,d);
            }
        }
        else if(strncmp(line,"MEAN",4)==0) {
            for(int i=0;i<stats[cls].dim;i++)
                fscanf(f,"%lf",&stats[cls].mean[i]);
            printf("[Global Stats] CLASS %d MEAN loaded\n",cls);
        }
        else if(strncmp(line,"COV",3)==0) {
            for(int i=0;i<stats[cls].dim*stats[cls].dim;i++)
                fscanf(f,"%lf",&stats[cls].cov[i]);
            memcpy(stats[cls].inv_cov, stats[cls].cov, stats[cls].dim*stats[cls].dim*sizeof(double));
            int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR,'U',stats[cls].dim,stats[cls].inv_cov,stats[cls].dim);
            if(info==0) {
                LAPACKE_dpotri(LAPACK_ROW_MAJOR,'U',stats[cls].dim,stats[cls].inv_cov,stats[cls].dim);
                printf("[Global Stats] CLASS %d COV inverted\n",cls);
            }
        }
    }
    fclose(f);
    printf("[Global Stats] Finished loading stats from %s\n", path);
    return 0;
}

int main(int argc, char **argv) {
    if(argc!=3) {
        fprintf(stderr,"Usage: %s global_stats.txt image.tif\n",argv[0]);
        return 1;
    }

    GDALAllRegister();
    ClassStats stats[2];
    load_global_stats(argv[1],stats);

    printf("[GDAL] Opening TIFF %s\n",argv[2]);
    GDALDatasetH ds = GDALOpen(argv[2],GA_ReadOnly);
    if(!ds) { fprintf(stderr,"Failed to open image\n"); return 1; }

    int w = GDALGetRasterXSize(ds);
    int h = GDALGetRasterYSize(ds);
    int bands = GDALGetRasterCount(ds);
    printf("[GDAL] Image size: %dx%d, Bands: %d\n", w,h,bands);

    int pad = PATCH_SIZE/2;
    int pw = w + 2*pad;
    int ph = h + 2*pad;

    double *image = calloc(pw*ph*bands,sizeof(double));

    for(int b=0;b<bands;b++) {
        printf("[GDAL] Reading band %d/%d\n",b+1,bands);
        GDALRasterBandH rb = GDALGetRasterBand(ds,b+1);
        double *tmp = malloc(sizeof(double)*w*h);
        GDALRasterIO(rb,GF_Read,0,0,w,h,tmp,w,h,GDT_Float64,0,0);
        for(int y=0;y<h;y++)
            for(int x=0;x<w;x++)
                image[((y+pad)*pw + (x+pad))*bands + b] = tmp[y*w+x];
        free(tmp);
        printf("[GDAL] Band %d loaded\n",b+1);
    }

    float *out = calloc(w*h,sizeof(float));
    total_jobs = h;
    finished_jobs = 0;

    int nthreads = get_nprocs();
    printf("[Threads] Using %d threads\n", nthreads);
    pthread_t *threads = malloc(nthreads*sizeof(pthread_t));
    Job *jobs = malloc(nthreads*sizeof(Job));

    int rows_per_thread = (h + nthreads -1)/nthreads;
    for(int t=0;t<nthreads;t++) {
        jobs[t].y_start = t*rows_per_thread;
        jobs[t].y_end = (t+1)*rows_per_thread;
        if(jobs[t].y_end>h) jobs[t].y_end=h;
        jobs[t].w=w; jobs[t].h=h; jobs[t].d=bands;
        jobs[t].padded=image;
        jobs[t].out=out;
        jobs[t].stats=stats;
        pthread_create(&threads[t],NULL,classify_rows,&jobs[t]);
    }

    for(int t=0;t<nthreads;t++) pthread_join(threads[t],NULL);

    printf("\n[GDAL] Writing output ENVI float32 file\n");
    GDALDriverH drv = GDALGetDriverByName("ENVI");
    GDALDatasetH ods = GDALCreate(drv,"classification_float.bin",w,h,1,GDT_Float32,NULL);

    double geoTransform[6];
    if(GDALGetGeoTransform(ds,geoTransform)==CE_None) GDALSetGeoTransform(ods,geoTransform);
    const char *proj = GDALGetProjectionRef(ds);
    if(proj) GDALSetProjection(ods,proj);

    GDALRasterIO(GDALGetRasterBand(ods,1),GF_Write,0,0,w,h,out,w,h,GDT_Float32,0,0);

    GDALClose(ods);
    GDALClose(ds);
    free(image);
    free(out);
    free(threads);
    free(jobs);
    for(int cls=0;cls<2;cls++) {
        free(stats[cls].mean);
        free(stats[cls].cov);
        free(stats[cls].inv_cov);
    }

    printf("[Done] Classification written to classification_float.bin\n");
    return 0;
}

