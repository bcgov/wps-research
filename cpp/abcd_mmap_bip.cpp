/* 20251020 version of abcd using mmap, and required BIP format input data. 

To convert to BIP use command like:
   gdal_translate -of ENVI -co INTERLEAVE=BIP -ot Float32 input_bsq_file.bsq output_bip_file.bip
 
*/
#include "misc.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <stdbool.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

// ----------------------------
// Globals
static size_t nr[3], nc[3], nb[3], skip_f, skip_off, np, np2;
static float *A, *B, *C, *x;
static int *bp;  // bad training pixel mask
static size_t total_iter = 0;
static time_t start_time;

// ----------------------------
// Unified bad pixel check
inline bool is_bad(float *dat, size_t n_b, size_t i) {
    bool zero = true;
    for(size_t k = 0; k < n_b; ++k) {
        float t = dat[i * n_b + k]; // BIP layout: pixel-major
        if (isnan(t) || isinf(t)) return true;
        if (t != 0) zero = false;
    }
    return (n_b > 1 && zero);
}

// ----------------------------
// Progress / ETA function
static inline void progress(size_t i, size_t total) {
    if (i % 500 == 0) {
        double elapsed = difftime(time(NULL), start_time);
        double rate = elapsed / (i + 1);
        double eta = rate * (total - (i + 1));
        fprintf(stderr,
                "Progress: %zu / %zu (%.2f%%) | Elapsed: %.1fs | ETA: %.1fs\r",
                i + 1, total, 100.0 * (i + 1) / total, elapsed, eta);
        fflush(stderr);
    }
}

// ----------------------------
// Sparse nearest-neighbor inference
void infer_px(size_t i) {
    if (is_bad(C, nb[2], i)) {
        for(size_t k = 0; k < nb[1]; ++k)
            x[i * nb[1] + k] = NAN;
        return;
    }

    float d, e, md = FLT_MAX;
    size_t j, k, mi = 0;

    // size_t n_train = (np - skip_off + skip_f - 1)/skip_f;

    for(j = skip_off; j < np; j += skip_f) {
        size_t idx = (j - skip_off)/skip_f;
        if (bp[idx]) continue; // skip bad training pixel

        d = 0;
        for(k = 0; k < nb[0]; ++k)
            d += (A[j * nb[0] + k] - C[i * nb[2] + k]) * (A[j * nb[0] + k] - C[i * nb[2] + k]);

        if(d < md) (md = d, mi = j);
    }

    for(k = 0; k < nb[1]; ++k)
        x[i * nb[1] + k] = B[mi * nb[1] + k];

    total_iter++;
    progress(total_iter, np2);
}

// ----------------------------
// mmap read with lazy paging and MADV_RANDOM
float *mmap_read_lazy(const char *fname, size_t expected_bytes) {
    int fd = open(fname, O_RDONLY);
    if(fd < 0){ perror("open"); exit(EXIT_FAILURE); }

    struct stat sb;
    if(fstat(fd, &sb) < 0){ perror("fstat"); close(fd); exit(EXIT_FAILURE); }

    if((size_t)sb.st_size < expected_bytes){
        fprintf(stderr, "File smaller than expected: %s\n", fname);
        close(fd); exit(EXIT_FAILURE);
    }

    void *data = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if(data == MAP_FAILED){ perror("mmap"); close(fd); exit(EXIT_FAILURE); }
    close(fd);

    if(madvise(data, sb.st_size, MADV_RANDOM) != 0)
        perror("madvise");

    return (float*)data;
}

// ----------------------------
void bwrite(const char *fname, float *data, size_t n) {
    FILE *f = fopen(fname, "wb");
    if(!f) err("fopen failed");
    fwrite(data, sizeof(float), n, f);
    fclose(f);
}

// ----------------------------
int main(int argc, char **argv){
    if(argc < 4) err("Usage: abcd_mmap_bip [img1] [img2] [img3] [skip_f] [skip_offset]");

    const char *A_fn = argv[1], *B_fn = argv[2], *C_fn = argv[3];
    skip_f = (argc > 4) ? atol(argv[4]) : 1;
    skip_off = (argc > 5) ? atol(argv[5]) : 0;

    size_t i;
    for(i = 0; i < 3; ++i) hread(hdr_fn(argv[1 + i]), nr[i], nc[i], nb[i]);

    if(nr[0]!=nr[1]||nc[0]!=nc[1]) err("A.shape != B.shape");
    if(nb[0]!=nb[2]) err("A.n_bands != C.n_bands");

    np = nr[0]*nc[0]; np2 = nr[2]*nc[2];
    if(skip_f >= np) err("illegal skip_f");

    size_t A_bytes = sizeof(float)*np*nb[0];
    size_t B_bytes = sizeof(float)*np*nb[1];
    size_t C_bytes = sizeof(float)*np2*nb[2];

    fprintf(stderr, "Mapping input files via mmap_lazy...\n");
    A = mmap_read_lazy(A_fn, A_bytes);
    B = mmap_read_lazy(B_fn, B_bytes);
    C = mmap_read_lazy(C_fn, C_bytes);

    fprintf(stderr, "Allocating output buffer...\n");
    x = falloc(np2*nb[1]);
    for(i=0;i<np2*nb[1];++i) x[i]=NAN;

    // Build training bad pixel mask
    size_t n_train = (np - skip_off + skip_f - 1)/skip_f;
    fprintf(stderr,"Building training bad pixel mask for %zu sampled pixels...\n",n_train);
    bp = ialloc(n_train);

    size_t j_idx = 0;
    for(size_t j = skip_off; j < np; j += skip_f, ++j_idx)
        bp[j_idx] = is_bad(A, nb[0], j) || is_bad(B, nb[1], j);

    fprintf(stderr, "Starting inference over %zu pixels...\n", np2);
    start_time = time(NULL);
    total_iter = 0;

    for(i=0;i<np2;++i)
        infer_px(i);

    fprintf(stderr,"\nDone. Total time: %.1fs\n", difftime(time(NULL), start_time));

    str pre(str("abcd_") + str(argv[1]) + "_" + str(argv[2]) + "_" + str(argv[3]) + "_" +
            to_string(skip_f) + "_" + to_string(skip_off));

    fprintf(stderr,"Writing output...\n");
    bwrite((pre+".bin").c_str(), x, np2*nb[1]);
    hwrite(pre+".hdr", nr[2], nc[2], nb[1]);

    free(x); free(bp);
    munmap(A, A_bytes); munmap(B, B_bytes); munmap(C, C_bytes);

    fprintf(stderr,"All done.\n");
    return 0;
}

