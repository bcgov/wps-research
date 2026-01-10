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
#include <sys/time.h>
#include <sys/sysinfo.h>
#include <gdal.h>
#include <cblas.h>
#include <lapacke.h>

/* ---------------- Types ---------------- */
typedef struct {
    int dim;
    double *mean;
    double *cov;
    double *inv_cov;
} ClassStats;

typedef struct {
    int job_index;
    int y_start;
    int y_end;
    int w,h,bands,patch_size;
    double *image;     // image buffer, row-major, flattened
    ClassStats *stats;
    int n_classes;
    unsigned char *out;
} Job;

typedef struct Node {
    Job job;
    struct Node *next;
} Node;

typedef struct {
    Node *head;
    Node *tail;
    pthread_mutex_t mutex;
} Queue;

/* ---------------- Globals ---------------- */
Queue job_queue;
pthread_mutex_t print_mutex = PTHREAD_MUTEX_INITIALIZER;
int total_jobs;
int jobs_done;
struct timeval start_time;

/* ---------------- Queue ---------------- */
void queue_init(Queue *q) {
    q->head = q->tail = NULL;
    pthread_mutex_init(&q->mutex,NULL);
}

void queue_push(Queue *q, Job job) {
    Node *n = malloc(sizeof(Node));
    n->job = job;
    n->next = NULL;
    pthread_mutex_lock(&q->mutex);
    if(q->tail) q->tail->next = n;
    else q->head = n;
    q->tail = n;
    pthread_mutex_unlock(&q->mutex);
}

int queue_pop(Queue *q, Job *job) {
    pthread_mutex_lock(&q->mutex);
    Node *n = q->head;
    if(!n) { pthread_mutex_unlock(&q->mutex); return 0; }
    q->head = n->next;
    if(!q->head) q->tail=NULL;
    *job = n->job;
    free(n);
    pthread_mutex_unlock(&q->mutex);
    return 1;
}

/* ---------------- Time & Progress ---------------- */
double time_elapsed() {
    struct timeval now;
    gettimeofday(&now,NULL);
    return (now.tv_sec-start_time.tv_sec) + (now.tv_usec-start_time.tv_usec)/1e6;
}

void print_progress() {
    pthread_mutex_lock(&print_mutex);
    double pct = 100.0 * jobs_done / total_jobs;
    double eta = time_elapsed() * (total_jobs-jobs_done)/jobs_done;
    int len = 40;
    int filled = (int)(len * pct / 100.0);
    printf("\r[");
    for(int i=0;i<filled;i++) printf("█");
    for(int i=filled;i<len;i++) printf("-");
    printf("] %.1f%% ETA %.1fs", pct, eta);
    fflush(stdout);
    if(jobs_done==total_jobs) printf("\n");
    pthread_mutex_unlock(&print_mutex);
}

/* ---------------- Global Stats Loading ---------------- */
ClassStats* load_global_stats(const char *path, int *n_classes) {
    FILE *f = fopen(path,"r");
    if(!f) { fprintf(stderr,"Failed to open %s\n",path); exit(1); }

    char line[4096];
    int max_cls=-1;

    // First pass: find max class ID
    while(fgets(line,sizeof(line),f)) {
        int cls;
        if(sscanf(line,"CLASS %d",&cls)==1) {
            if(cls>max_cls) max_cls=cls;
        }
    }
    if(max_cls<0) { fprintf(stderr,"No classes found\n"); exit(1); }
    *n_classes = max_cls+1;

    ClassStats *stats = calloc(*n_classes,sizeof(ClassStats));
    rewind(f);

    int cls;
    while(fgets(line,sizeof(line),f)) {
        if(sscanf(line,"CLASS %d",&cls)==1) {
            int d=-1;
            if(fgets(line,sizeof(line),f) && sscanf(line,"DIM %d",&d)==1) {
                stats[cls].dim = d;
                stats[cls].mean = calloc(d,sizeof(double));
                stats[cls].cov  = calloc(d*d,sizeof(double));
                stats[cls].inv_cov = calloc(d*d,sizeof(double));
            }

            // Read MEAN
            if(fgets(line,sizeof(line),f) && strncmp(line,"MEAN",4)==0) {
                for(int i=0;i<stats[cls].dim;i++)
                    fscanf(f,"%lf",&stats[cls].mean[i]);
            }

            // Read COV
            if(fgets(line,sizeof(line),f) && strncmp(line,"COV",3)==0) {
                for(int i=0;i<stats[cls].dim*stats[cls].dim;i++)
                    fscanf(f,"%lf",&stats[cls].cov[i]);
            }

            // Invert cov using LAPACK
            memcpy(stats[cls].inv_cov, stats[cls].cov, sizeof(double)*stats[cls].dim*stats[cls].dim);
            int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR,'U',stats[cls].dim,stats[cls].inv_cov,stats[cls].dim);
            if(info!=0) fprintf(stderr,"Warning: dpotrf info=%d for class %d\n",info,cls);
            info = LAPACKE_dpotri(LAPACK_ROW_MAJOR,'U',stats[cls].dim,stats[cls].inv_cov,stats[cls].dim);
            if(info!=0) fprintf(stderr,"Warning: dpotri info=%d for class %d\n",info,cls);
        }
    }
    fclose(f);
    return stats;
}

/* ---------------- Classification ---------------- */
void classify_vector(Job *job, int y) {
    int pad = job->patch_size/2;
    int w=job->w, h=job->h, bands=job->bands;
    int patch = job->patch_size;
    int d = patch*patch*bands;

    double *patch_vec = malloc(sizeof(double)*d);

    for(int x=0;x<w;x++) {
        int idx=0;
        for(int yy=y-pad;yy<=y+pad;yy++)
            for(int xx=x-pad;xx<=x+pad;xx++)
                for(int b=0;b<bands;b++) {
                    int yy_clamp = yy<0?0:(yy>=h?h-1:yy);
                    int xx_clamp = xx<0?0:(xx>=w?w-1:xx);
                    patch_vec[idx++] = job->image[(yy_clamp*w+xx_clamp)*bands+b];
                }

        double best = INFINITY;
        int best_cls = 0;
        for(int cls=0;cls<job->n_classes;cls++) {
            ClassStats *s = &job->stats[cls];
            if(!s->mean) continue;

            double *tmp = malloc(d*sizeof(double));
            for(int i=0;i<d;i++) tmp[i] = patch_vec[i]-s->mean[i];
            double score = cblas_ddot(d,tmp,1, tmp,1); // Mahalanobis quadratic form
            free(tmp);

            if(score<best) { best=score; best_cls=cls; }
        }
        job->out[y*w+x] = (unsigned char)best_cls;
    }

    free(patch_vec);
}

/* ---------------- Worker ---------------- */
void* worker_thread(void *arg) {
    Job job;
    while(queue_pop(&job_queue,&job)) {
        for(int y=job.y_start;y<job.y_end;y++) {
            classify_vector(&job,y);
            __sync_fetch_and_add(&jobs_done,1);
            if(job.job_index%100==0) print_progress();
        }
    }
    return NULL;
}

/* ---------------- Main ---------------- */
int main(int argc,char **argv) {
    if(argc<3) { fprintf(stderr,"Usage: %s global_stats.txt image.tif\n",argv[0]); return 1; }

    GDALAllRegister();
    int n_classes;
    ClassStats *stats = load_global_stats(argv[1], &n_classes);

    GDALDatasetH ds = GDALOpen(argv[2], GA_ReadOnly);
    if(!ds) { fprintf(stderr,"Failed to open image\n"); return 1; }

    int w = GDALGetRasterXSize(ds);
    int h = GDALGetRasterYSize(ds);
    int bands = GDALGetRasterCount(ds);

    double *image = calloc(w*h*bands,sizeof(double));
    for(int b=0;b<bands;b++) {
        GDALRasterBandH rb = GDALGetRasterBand(ds,b+1);
        double *tmp = malloc(sizeof(double)*w*h);
        GDALRasterIO(rb,GF_Read,0,0,w,h,tmp,w,h,GDT_Float64,0,0);
        for(int i=0;i<w*h;i++) image[i*bands+b] = tmp[i];
        free(tmp);
    }

    unsigned char *out = calloc(w*h,sizeof(unsigned char));

    int patch_size=7;
    int nthreads = get_nprocs();
    pthread_t *threads = malloc(sizeof(pthread_t)*nthreads);

    jobs_done = 0;
    total_jobs = h;
    queue_init(&job_queue);
    gettimeofday(&start_time,NULL);

    // Enqueue one job per row
    for(int y=0;y<h;y++) {
        Job job = {y, y, y+1, w, h, bands, patch_size, image, stats, n_classes, out};
        job.job_index = y;
        queue_push(&job_queue,job);
    }

    for(int t=0;t<nthreads;t++) pthread_create(&threads[t],NULL,worker_thread,NULL);
    for(int t=0;t<nthreads;t++) pthread_join(threads[t],NULL);

    // Save ENVI
    GDALDriverH drv = GDALGetDriverByName("ENVI");
    GDALDatasetH ods = GDALCreate(drv,"classification.bin",w,h,1,GDT_Byte,NULL);
    GDALRasterIO(GDALGetRasterBand(ods,1),GF_Write,0,0,w,h,out,w,h,GDT_Byte,0,0);
    GDALClose(ods);
    GDALClose(ds);

    free(image);
    free(out);
    for(int i=0;i<n_classes;i++) {
        free(stats[i].mean);
        free(stats[i].cov);
        free(stats[i].inv_cov);
    }
    free(stats);
    free(threads);

    return 0;
}

