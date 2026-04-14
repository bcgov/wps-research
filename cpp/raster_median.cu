/* raster_medoid.cu

nvcc -O3 -arch=sm_89 -std=c++17 raster_medoid.cu misc.cpp -o raster_medoid -lpthread

 *
 * GPU-accelerated median/medoid computation for a stack of rasters.
 *
 * Usage
 * -----
 *  raster_medoid [options] [raster1.bin] .. [rasterN.bin] [output.bin]
 *
 *  Options:
 *    -medoid    Compute medoid (vector closest to median) instead of
 *               per-band median.  Default is median.
 *
 * Strategy (triple-buffered pipeline, low RAM)
 * ---------------------------------------------
 *  • The image is divided into chunks of CHUNK_ROWS rows.
 *  • A 3-stage asynchronous pipeline runs concurrently:
 *      Stage 1 (READ):    Parallel threaded reads via fseek into staging buf.
 *      Stage 2 (COMPUTE): H2D, kernel, D2H on GPU.
 *      Stage 3 (WRITE):   Random-access fwrite into pre-allocated output.
 *    Each stage runs in its own thread(s) with producer/consumer queues
 *    so read, compute, and write all overlap.
 *
 * Compile
 * -------
 *  nvcc -O3 -arch=sm_89 -std=c++17 raster_medoid.cu misc.cpp -o raster_medoid -lpthread
 */
#include "misc.h"

#include <cmath>
#include <algorithm>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <queue>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

// ── helpers ──────────────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s:%d  %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(_e));                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static double now_sec() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

// ── globals ──────────────────────────────────────────────────────────────────

static size_t g_nrow, g_ncol, g_nband, g_np, g_T;
static char **g_fnames;
static bool   g_use_medoid = false;   // false=median, true=medoid

static const int CHUNK_ROWS = 256;

// ── chunk descriptor ────────────────────────────────────────────────────────

struct ChunkDesc {
  float *buf;
  size_t row_start;
  size_t nrows;
  size_t chunk_pixels;
  size_t chunk_max;
  int    buf_idx;
};

// ── generic thread-safe queue ───────────────────────────────────────────────

template<typename T>
struct TSQueue {
  pthread_mutex_t mtx;
  pthread_cond_t  cond;
  std::queue<T>   q;
  bool            done;

  TSQueue() : done(false) {
    pthread_mutex_init(&mtx, nullptr);
    pthread_cond_init(&cond, nullptr);
  }
  void push(const T &item) {
    pthread_mutex_lock(&mtx);
    q.push(item);
    pthread_cond_signal(&cond);
    pthread_mutex_unlock(&mtx);
  }
  bool pop(T &item) {
    pthread_mutex_lock(&mtx);
    while (q.empty() && !done)
      pthread_cond_wait(&cond, &mtx);
    if (q.empty() && done) {
      pthread_mutex_unlock(&mtx);
      return false;
    }
    item = q.front();
    q.pop();
    pthread_mutex_unlock(&mtx);
    return true;
  }
  void finish() {
    pthread_mutex_lock(&mtx);
    done = true;
    pthread_cond_broadcast(&cond);
    pthread_mutex_unlock(&mtx);
  }
};

static TSQueue<ChunkDesc> read_to_compute;
static TSQueue<ChunkDesc> compute_to_write;
static TSQueue<int> free_buf_pool;

// ── progress tracking ───────────────────────────────────────────────────────

static pthread_mutex_t progress_mtx = PTHREAD_MUTEX_INITIALIZER;
static size_t prog_read_chunks  = 0;
static size_t prog_read_rows    = 0;
static size_t prog_comp_chunks  = 0;
static size_t prog_comp_rows    = 0;
static size_t prog_write_chunks = 0;
static size_t prog_write_rows   = 0;
static double prog_start_time   = 0;
static size_t prog_total_chunks = 0;
static size_t prog_total_rows   = 0;
static double prog_total_bytes  = 0;

static void print_progress(const char *stage, size_t stage_chunks,
                           size_t furthest_rows) {
  pthread_mutex_lock(&progress_mtx);

  double elapsed = now_sec() - prog_start_time;
  double frac = (prog_total_rows > 0)
              ? (double)prog_comp_rows / (double)prog_total_rows : 0;
  double eta = (frac > 0.001) ? elapsed / frac - elapsed : 0;

  double mb_read = (double)prog_read_rows * g_ncol * g_nband * g_T
                   * sizeof(float) / (1024.0 * 1024.0);
  double mb_written = (double)prog_write_rows * g_ncol * g_nband
                      * sizeof(float) / (1024.0 * 1024.0);
  double rate = (elapsed > 0.1) ? mb_read / elapsed : 0;

  int eta_m = (int)(eta / 60);
  int eta_s = (int)(eta) % 60;

  printf("\r\033[K"
         "R %zu/%zu  |  C %zu/%zu  |  W %zu/%zu  "
         "│  rows %zu/%zu (%.1f%%)  "
         "│  %.0f MB read  %.0f MB written  %.0f MB/s  "
         "│  ETA %dm%02ds   [%s]",
         prog_read_chunks, prog_total_chunks,
         prog_comp_chunks, prog_total_chunks,
         prog_write_chunks, prog_total_chunks,
         furthest_rows, prog_total_rows,
         100.0 * frac,
         mb_read, mb_written, rate,
         eta_m, eta_s,
         stage);

  fflush(stdout);
  pthread_mutex_unlock(&progress_mtx);
}

// ── chunk reader thread arg ─────────────────────────────────────────────────

struct ChunkReadArg {
  size_t file_idx;
  size_t band;
  size_t row_start;
  size_t nrows;
  float *dst;
};

static void *chunk_reader_thread(void *arg)
{
  ChunkReadArg *a = (ChunkReadArg *)arg;
  size_t offset_floats = a->band * g_nrow * g_ncol + a->row_start * g_ncol;
  size_t count = a->nrows * g_ncol;

  FILE *f = fopen(g_fnames[a->file_idx], "rb");
  if (!f) {
    fprintf(stderr, "\nchunk_reader: cannot open %s\n", g_fnames[a->file_idx]);
    exit(1);
  }
  fseek(f, (long)(offset_floats * sizeof(float)), SEEK_SET);
  size_t got = fread(a->dst, sizeof(float), count, f);
  if (got != count) {
    fprintf(stderr, "\nchunk_reader: short read %s band=%zu row=%zu\n",
            g_fnames[a->file_idx], a->band, a->row_start);
    exit(1);
  }
  fclose(f);
  return nullptr;
}

// ── reader stage thread ─────────────────────────────────────────────────────

struct ReaderCtx {
  float **pinned_bufs;
  size_t chunk_max;
};

static void *reader_stage(void *arg)
{
  ReaderCtx *ctx = (ReaderCtx *)arg;
  size_t total_chunks = (g_nrow + CHUNK_ROWS - 1) / CHUNK_ROWS;

  for (size_t ci = 0; ci < total_chunks; ++ci) {
    size_t row_start = ci * CHUNK_ROWS;
    size_t nrows = std::min((size_t)CHUNK_ROWS, g_nrow - row_start);
    size_t this_pixels = nrows * g_ncol;

    int buf_idx;
    free_buf_pool.pop(buf_idx);
    float *stage = ctx->pinned_bufs[buf_idx];

    size_t n_rt = g_T * g_nband;
    pthread_t *rtids = (pthread_t *)malloc(sizeof(pthread_t) * n_rt);
    ChunkReadArg *rargs = (ChunkReadArg *)malloc(sizeof(ChunkReadArg) * n_rt);

    for (size_t t = 0; t < g_T; ++t) {
      for (size_t k = 0; k < g_nband; ++k) {
        size_t idx = t * g_nband + k;
        rargs[idx].file_idx  = t;
        rargs[idx].band      = k;
        rargs[idx].row_start = row_start;
        rargs[idx].nrows     = nrows;
        rargs[idx].dst       = stage
                             + t * g_nband * ctx->chunk_max
                             + k * ctx->chunk_max;
        pthread_create(&rtids[idx], nullptr, chunk_reader_thread, &rargs[idx]);
      }
    }
    for (size_t idx = 0; idx < n_rt; ++idx)
      pthread_join(rtids[idx], nullptr);
    free(rtids);
    free(rargs);

    ChunkDesc cd;
    cd.buf          = stage;
    cd.row_start    = row_start;
    cd.nrows        = nrows;
    cd.chunk_pixels = this_pixels;
    cd.chunk_max    = ctx->chunk_max;
    cd.buf_idx      = buf_idx;

    pthread_mutex_lock(&progress_mtx);
    prog_read_chunks++;
    prog_read_rows += nrows;
    pthread_mutex_unlock(&progress_mtx);
    print_progress("read", prog_read_chunks, prog_comp_rows);

    read_to_compute.push(cd);
  }

  read_to_compute.finish();
  return nullptr;
}

// ── CUDA kernels ────────────────────────────────────────────────────────────

__device__ static float device_median(float *vals, int n)
{
  for (int i = 1; i < n; ++i) {
    float key = vals[i];
    int j = i - 1;
    while (j >= 0 && vals[j] > key) { vals[j + 1] = vals[j]; --j; }
    vals[j + 1] = key;
  }
  if (n == 0) return NAN;
  return (n % 2 == 0) ? (vals[n / 2 - 1] + vals[n / 2]) * 0.5f
                       : vals[n / 2];
}

// d_dat layout: [t * nband * chunk_pixels + band * chunk_pixels + local_j]
// d_out layout: [band * chunk_pixels + local_j]

__global__ void median_kernel(
    const float * __restrict__ d_dat,
    float       * __restrict__ d_out,
    int T, int nband, int chunk_pixels,
    float *d_workspace)
{
  int local_j = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_j >= chunk_pixels) return;

  float *scratch = d_workspace + local_j * T;

  for (int k = 0; k < nband; ++k) {
    int valid = 0;
    for (int t = 0; t < T; ++t) {
      float v = d_dat[(size_t)t * nband * chunk_pixels
                     + (size_t)k * chunk_pixels + local_j];
      if (!isnan(v)) scratch[valid++] = v;
    }
    d_out[(size_t)k * chunk_pixels + local_j] = device_median(scratch, valid);
  }
}

__global__ void medoid_kernel(
    const float * __restrict__ d_dat,
    float       * __restrict__ d_out,
    int T, int nband, int chunk_pixels,
    float *d_workspace)
{
  int local_j = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_j >= chunk_pixels) return;

  float *scratch = d_workspace + local_j * T;

  // Step 1: compute per-band median into d_out (temporary)
  for (int k = 0; k < nband; ++k) {
    int valid = 0;
    for (int t = 0; t < T; ++t) {
      float v = d_dat[(size_t)t * nband * chunk_pixels
                     + (size_t)k * chunk_pixels + local_j];
      if (!isnan(v)) scratch[valid++] = v;
    }
    d_out[(size_t)k * chunk_pixels + local_j] = device_median(scratch, valid);
  }

  // Step 2: find the timestep whose vector is closest to the median
  int   medoid_idx = -1;
  float min_dist   = 1e30f;

  for (int t = 0; t < T; ++t) {
    float sum = 0.0f;
    int   valid_count = 0;
    for (int k = 0; k < nband; ++k) {
      float v = d_dat[(size_t)t * nband * chunk_pixels
                     + (size_t)k * chunk_pixels + local_j];
      float m = d_out[(size_t)k * chunk_pixels + local_j];
      if (!isnan(v) && !isnan(m)) {
        float dd = v - m;
        sum += dd * dd;
        ++valid_count;
      }
    }
    float dist = (valid_count > 0) ? sum / valid_count : 1e30f;
    if (dist < min_dist) {
      min_dist = dist;
      medoid_idx = t;
    }
  }

  // Step 3: output the medoid vector (or NaN if none found)
  if (medoid_idx >= 0) {
    for (int k = 0; k < nband; ++k) {
      d_out[(size_t)k * chunk_pixels + local_j] =
          d_dat[(size_t)medoid_idx * nband * chunk_pixels
               + (size_t)k * chunk_pixels + local_j];
    }
  } else {
    for (int k = 0; k < nband; ++k) {
      d_out[(size_t)k * chunk_pixels + local_j] = NAN;
    }
  }
}

// ── compute stage thread ────────────────────────────────────────────────────

static void *compute_stage(void * /*arg*/)
{
  size_t chunk_max = (size_t)CHUNK_ROWS * g_ncol;

  float *d_dat, *d_out, *d_workspace;
  size_t d_dat_bytes = chunk_max * g_nband * g_T * sizeof(float);
  size_t d_out_bytes = chunk_max * g_nband       * sizeof(float);
  size_t d_wrk_bytes = chunk_max * g_T           * sizeof(float);

  CUDA_CHECK(cudaMalloc(&d_dat, d_dat_bytes));
  CUDA_CHECK(cudaMalloc(&d_out, d_out_bytes));
  CUDA_CHECK(cudaMalloc(&d_workspace, d_wrk_bytes));

  const int THREADS = 256;
  ChunkDesc cd;

  while (read_to_compute.pop(cd)) {
    size_t h2d_bytes = chunk_max * g_nband * g_T * sizeof(float);
    CUDA_CHECK(cudaMemcpy(d_dat, cd.buf, h2d_bytes, cudaMemcpyHostToDevice));

    free_buf_pool.push(cd.buf_idx);

    int blocks = (int)((cd.chunk_pixels + THREADS - 1) / THREADS);
    if (g_use_medoid) {
      medoid_kernel<<<blocks, THREADS>>>(
          d_dat, d_out,
          (int)g_T, (int)g_nband, (int)cd.chunk_pixels,
          d_workspace);
    } else {
      median_kernel<<<blocks, THREADS>>>(
          d_dat, d_out,
          (int)g_T, (int)g_nband, (int)cd.chunk_pixels,
          d_workspace);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float *wbuf = (float *)malloc(cd.chunk_pixels * g_nband * sizeof(float));
    if (!wbuf) { fprintf(stderr, "\nmalloc failed for wbuf\n"); exit(1); }

    for (size_t k = 0; k < g_nband; ++k) {
      CUDA_CHECK(cudaMemcpy(
          wbuf + k * cd.chunk_pixels,
          (char *)d_out + k * chunk_max * sizeof(float),
          cd.chunk_pixels * sizeof(float),
          cudaMemcpyDeviceToHost));
    }

    ChunkDesc wd = cd;
    wd.buf = wbuf;

    pthread_mutex_lock(&progress_mtx);
    prog_comp_chunks++;
    prog_comp_rows += cd.nrows;
    pthread_mutex_unlock(&progress_mtx);
    print_progress("compute", prog_comp_chunks, prog_comp_rows);

    compute_to_write.push(wd);
  }

  compute_to_write.finish();

  cudaFree(d_dat);
  cudaFree(d_out);
  cudaFree(d_workspace);
  return nullptr;
}

// ── writer stage thread ─────────────────────────────────────────────────────

static void *writer_stage(void *arg)
{
  FILE *outfp = (FILE *)arg;
  ChunkDesc wd;

  while (compute_to_write.pop(wd)) {
    for (size_t k = 0; k < g_nband; ++k) {
      size_t offset_floats = k * g_np + wd.row_start * g_ncol;
      fseek(outfp, (long)(offset_floats * sizeof(float)), SEEK_SET);
      size_t written = fwrite(wd.buf + k * wd.chunk_pixels,
                              sizeof(float), wd.chunk_pixels, outfp);
      if (written != wd.chunk_pixels) {
        fprintf(stderr, "\nwriter: short write band=%zu row_start=%zu\n",
                k, wd.row_start);
        exit(1);
      }
    }

    free(wd.buf);

    pthread_mutex_lock(&progress_mtx);
    prog_write_chunks++;
    prog_write_rows += wd.nrows;
    pthread_mutex_unlock(&progress_mtx);
    print_progress("write", prog_write_chunks, prog_write_rows);
  }

  fflush(outfp);
  return nullptr;
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char **argv)
{
  if (argc < 4) {
    fprintf(stderr,
      "Usage: raster_medoid [-medoid] [raster1.bin] .. [rasterN.bin] [output.bin]\n"
      "\n"
      "Computes per-band median (default) or medoid across a stack of rasters.\n"
      "\n"
      "Options:\n"
      "  -medoid   Output the actual observed vector (timestep) closest to the\n"
      "            per-band median, rather than the median itself.  This preserves\n"
      "            inter-band relationships from a real observation.\n");
    exit(1);
  }

  // ── parse options ─────────────────────────────────────────────────────────
  int arg_start = 1;
  while (arg_start < argc && argv[arg_start][0] == '-') {
    if (strcmp(argv[arg_start], "-medoid") == 0) {
      g_use_medoid = true;
      arg_start++;
    } else {
      fprintf(stderr, "Unknown option: %s\n", argv[arg_start]);
      exit(1);
    }
  }

  int n_positional = argc - arg_start;
  if (n_positional < 3) {
    fprintf(stderr, "Need at least 2 input files and 1 output file.\n");
    exit(1);
  }

  g_T = (size_t)(n_positional - 1);
  char **pos_argv = argv + arg_start;  // pos_argv[0..g_T-1] = inputs, [g_T] = output
  char *output_path = pos_argv[g_T];

  // ── filter input files ────────────────────────────────────────────────────
  {
    int valid_count = 0;
    for (size_t i = 0; i < g_T; i++) {
      str bfn(pos_argv[i]);
      if (!exists(bfn)) {
        printf("skipping %s (missing .bin)\n", pos_argv[i]);
        continue;
      }
      str hfn(hdr_fn(bfn));
      if (!exists(hfn)) {
        printf("skipping %s (missing .hdr)\n", pos_argv[i]);
        continue;
      }
      pos_argv[valid_count] = pos_argv[i];
      valid_count++;
    }
    g_T = valid_count;
    if (g_T < 1) err("no valid input files");
  }

  g_fnames = pos_argv;  // pos_argv[0..g_T-1] are now valid inputs

  // ── read headers ──────────────────────────────────────────────────────────
  {
    size_t nrow2, ncol2, nband2;
    str hfn0(hdr_fn(str(g_fnames[0])));
    hread(hfn0, g_nrow, g_ncol, g_nband);
    g_np = g_nrow * g_ncol;

    for (size_t i = 1; i < g_T; ++i) {
      str hfni(hdr_fn(str(g_fnames[i])));
      hread(hfni, nrow2, ncol2, nband2);
      if (g_nrow != nrow2 || g_ncol != ncol2 || g_nband != nband2)
        err(str("file: ") + str(g_fnames[i]) +
            str(" has different shape than ") + str(g_fnames[0]));
    }
  }

  printf("Mode: %s\n", g_use_medoid ? "medoid" : "median");
  printf("Image: %zu rows x %zu cols x %zu bands, %zu timesteps\n",
         g_nrow, g_ncol, g_nband, g_T);

  // ── progress init ─────────────────────────────────────────────────────────
  prog_total_chunks = (g_nrow + CHUNK_ROWS - 1) / CHUNK_ROWS;
  prog_total_rows   = g_nrow;
  prog_total_bytes  = (double)g_nrow * g_ncol * g_nband * g_T * sizeof(float);
  prog_start_time   = now_sec();

  printf("Total: %zu chunks of %d rows, %.2f GiB input, %.2f GiB output\n",
         prog_total_chunks, CHUNK_ROWS,
         prog_total_bytes / (1024.0*1024.0*1024.0),
         (double)g_np * g_nband * sizeof(float) / (1024.0*1024.0*1024.0));

  // ── open output file (truncate if exists) ─────────────────────────────────
  FILE *outfp = fopen(output_path, "w+b");
  if (!outfp) err("cannot open output file for writing");
  {
    size_t total_bytes = g_np * g_nband * sizeof(float);
    fseek(outfp, (long)(total_bytes - 1), SEEK_SET);
    char zero = 0;
    fwrite(&zero, 1, 1, outfp);
    fflush(outfp);
  }

  // ── start writer thread ───────────────────────────────────────────────────
  pthread_t writer_tid;
  pthread_create(&writer_tid, nullptr, writer_stage, outfp);

  // ── allocate pinned buffer pool (3 buffers for triple-buffering) ──────────
  const int NUM_BUFS = 3;
  size_t chunk_max = (size_t)CHUNK_ROWS * g_ncol;
  size_t stage_bytes = chunk_max * g_nband * g_T * sizeof(float);

  printf("Pinned staging: %d x %.2f GiB = %.2f GiB\n",
         NUM_BUFS,
         (double)stage_bytes / (1024.0*1024.0*1024.0),
         (double)NUM_BUFS * stage_bytes / (1024.0*1024.0*1024.0));

  float *pinned_bufs[NUM_BUFS];
  for (int b = 0; b < NUM_BUFS; ++b) {
    CUDA_CHECK(cudaMallocHost(&pinned_bufs[b], stage_bytes));
    free_buf_pool.push(b);
  }

  // ── launch pipeline stages ────────────────────────────────────────────────
  printf("\n");

  ReaderCtx rctx;
  rctx.pinned_bufs = pinned_bufs;
  rctx.chunk_max   = chunk_max;

  pthread_t reader_tid, compute_tid;
  pthread_create(&reader_tid,  nullptr, reader_stage,  &rctx);
  pthread_create(&compute_tid, nullptr, compute_stage, nullptr);

  pthread_join(reader_tid,  nullptr);
  pthread_join(compute_tid, nullptr);
  pthread_join(writer_tid,  nullptr);

  printf("\n\n");

  fclose(outfp);

  // ── write header ──────────────────────────────────────────────────────────
  str ofn(output_path);
  str ohfn(hdr_fn(ofn, true));
  run((str("cp -v ") + hdr_fn(str(g_fnames[0])) + str(" ") + ohfn).c_str());

  double elapsed = now_sec() - prog_start_time;
  int em = (int)(elapsed / 60);
  int es = (int)(elapsed) % 60;
  printf("Done in %dm%02ds — %s\n", em, es, output_path);

  // ── cleanup ───────────────────────────────────────────────────────────────
  for (int b = 0; b < NUM_BUFS; ++b)
    cudaFreeHost(pinned_bufs[b]);

  return 0;
}
