/* raster_medoid.cu

nvcc -O3 -arch=sm_89 -std=c++17 raster_medoid.cu misc.cpp -o raster_medoid -lpthread

 *
 * GPU-accelerated median/medoid computation for a stack of rasters.
 *
 * Strategy
 * --------
 *  • All input rasters are read in parallel using 8 POSIX threads so that
 *    disk I/O saturates available bandwidth while the CPU is not the bottleneck.
 *  • The pixel dimension (np = nrow * ncol) is split into chunks that fit
 *    inside GPU global memory.  Each chunk is processed as follows:
 *      1. Host → Device copy of the slice [chunk_start, chunk_end) for every
 *         band of every time step.
 *      2. A CUDA kernel computes the per-band median across time steps for
 *         every pixel in the chunk, writing results to a device output buffer.
 *      3. Device → Host copy of the output slice.
 *  • The final output array is written to disk via bwrite() exactly as before.
 *
 * Compile example
 * ---------------
 *
 * (sm_89 targets the L40S; adjust -arch as needed.)
 */
#include "misc.h"

#include <cmath>
#include <algorithm>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>

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

// ── globals shared with reader threads ────────────────────────────────────── 

static size_t  g_nrow, g_ncol, g_nband, g_np, g_T;
static float **g_dat;        // [T][np * nband]  (host, fully loaded)
static char  **g_fnames;     // input file names

// ── parallel reader thread ────────────────────────────────────────────────── 

struct ReadArg { size_t idx; };

static void *reader_thread(void *arg)
{
  size_t i = ((ReadArg *)arg)->idx;
  printf("+r %s\n", g_fnames[i]);
  g_dat[i] = bread(g_fnames[i], g_nrow, g_ncol, g_nband);
  if (!g_dat[i]) {
    fprintf(stderr, "Failed to read %s\n", g_fnames[i]);
    exit(1);
  }
  return nullptr;
}

// ── CUDA kernel ───────────────────────────────────────────────────────────── 
//
// Layout of d_dat:  d_dat[t * nband * chunk + band * chunk + local_j]
//   where local_j = j - chunk_start
//
// We use a flat temporary buffer d_tmp of size  T * nband * chunk_size
// arranged as  [t][band][local_j]  — same layout — just an alias to d_dat.
//
// Per pixel per band we sort T values in registers / shared memory to find
// the median.  T is typically small (e.g. ≤ 64); we use an insertion sort
// which is efficient for small N.

__device__ static float device_median(float *vals, int n)
{
  // insertion sort in place
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

// d_dat layout: [t * nband * chunk_size + band * chunk_size + local_j]
// d_out layout: [band * chunk_size + local_j]

__global__ void median_kernel(
    const float * __restrict__ d_dat,
    float       * __restrict__ d_out,
    int T, int nband, int chunk_size,
    float *d_workspace)   // T floats per thread — pre-allocated scratch
{
  int local_j = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_j >= chunk_size) return;

  // workspace: one row of T floats for this thread
  float *scratch = d_workspace + local_j * T;   // unique per thread

  for (int k = 0; k < nband; ++k) {
    int valid = 0;
    for (int t = 0; t < T; ++t) {
      float v = d_dat[(size_t)t * nband * chunk_size + (size_t)k * chunk_size + local_j];
      if (!isnan(v)) scratch[valid++] = v;
    }
    d_out[(size_t)k * chunk_size + local_j] = device_median(scratch, valid);
  }
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char **argv)
{
  if (argc < 4)
    err("raster_medoid [raster file 1] .. [raster file N] [output file]");

  g_T      = (size_t)(argc - 2);
  g_fnames = argv + 1;          // argv[1..T] are input files

  // ── read headers ──────────────────────────────────────────────────────────
  {
    size_t nrow2, ncol2, nband2;
    str hfn0(hdr_fn(str(argv[1])));
    hread(hfn0, g_nrow, g_ncol, g_nband);
    g_np = g_nrow * g_ncol;

    for (size_t i = 1; i < g_T; ++i) {
      str hfni(hdr_fn(str(argv[i + 1])));
      hread(hfni, nrow2, ncol2, nband2);
      if (g_nrow != nrow2 || g_ncol != ncol2 || g_nband != nband2)
        err(str("file: ") + str(argv[i + 1]) +
            str(" has different shape than ") + str(argv[1]));
    }
  }

  // ── allocate host dat array ───────────────────────────────────────────────
  g_dat = (float **)malloc(sizeof(float *) * g_T);
  if (!g_dat) err("malloc failed for dat array");
  memset(g_dat, 0, sizeof(float *) * g_T);

  // ── parallel read with up to 8 worker threads ─────────────────────────────
  {
    const int MAX_WORKERS = 8;
    size_t remaining = g_T;
    size_t base      = 0;

    // ReadArg pool (stack allocation is fine; we join before leaving scope)
    ReadArg args[8];
    pthread_t tids[8];

    while (remaining > 0) {
      int batch = (int)std::min(remaining, (size_t)MAX_WORKERS);
      for (int w = 0; w < batch; ++w) {
        args[w].idx = base + w;
        pthread_create(&tids[w], nullptr, reader_thread, &args[w]);
      }
      for (int w = 0; w < batch; ++w)
        pthread_join(tids[w], nullptr);
      base      += batch;
      remaining -= batch;
    }
  }
  printf("All %zu files loaded into host RAM.\n", g_T);

  // ── allocate host output ──────────────────────────────────────────────────
  float *out = falloc(g_np * g_nband);
  if (!out) err("falloc failed for output");

  // ── decide chunk size ─────────────────────────────────────────────────────
  // Budget: leave ~2 GB headroom on the L40S (48 GB).
  // GPU needs:
  //   input  : chunk_size * nband * T * 4 bytes
  //   output : chunk_size * nband     * 4 bytes
  //   scratch: chunk_size * T         * 4 bytes
  // We target at most 40 GB for safety.

  const size_t GPU_BUDGET = (size_t)40 * 1024 * 1024 * 1024;   // 40 GiB
  size_t bytes_per_pixel  = (size_t)g_nband * g_T * sizeof(float)   // input
                          + (size_t)g_nband      * sizeof(float)     // output
                          + (size_t)g_T          * sizeof(float);    // scratch
  size_t chunk_size = GPU_BUDGET / bytes_per_pixel;
  chunk_size = std::min(chunk_size, g_np);
  chunk_size = std::max(chunk_size, (size_t)1);

  printf("Chunk size: %zu pixels (of %zu total, %zu bands, %zu timesteps)\n",
         chunk_size, g_np, g_nband, g_T);

  // ── GPU allocations (sized to chunk) ─────────────────────────────────────
  float *d_dat, *d_out, *d_workspace;
  CUDA_CHECK(cudaMalloc(&d_dat,
      chunk_size * g_nband * g_T * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out,
      chunk_size * g_nband       * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_workspace,
      chunk_size * g_T           * sizeof(float)));

  // pinned staging buffer for async H2D transfers
  float *h_stage;
  CUDA_CHECK(cudaMallocHost(&h_stage,
      chunk_size * g_nband * g_T * sizeof(float)));

  // ── process chunks ────────────────────────────────────────────────────────
  const int THREADS = 256;
  size_t chunk_start = 0;

  while (chunk_start < g_np) {
    size_t this_chunk = std::min(chunk_size, g_np - chunk_start);

    // Pack the host staging buffer: [t][band][local_j]
    // g_dat[t] layout: [band * np + pixel_idx]  (BSQ / band-sequential)
    for (size_t t = 0; t < g_T; ++t) {
      for (size_t k = 0; k < g_nband; ++k) {
        const float *src = g_dat[t] + k * g_np + chunk_start;
        float       *dst = h_stage
                         + t * g_nband * chunk_size
                         + k * chunk_size;
        memcpy(dst, src, this_chunk * sizeof(float));
      }
    }

    // H2D
    CUDA_CHECK(cudaMemcpy(d_dat, h_stage,
        this_chunk * g_nband * g_T * sizeof(float),
        cudaMemcpyHostToDevice));

    // Launch kernel
    int blocks = (int)((this_chunk + THREADS - 1) / THREADS);
    median_kernel<<<blocks, THREADS>>>(
        d_dat, d_out,
        (int)g_T, (int)g_nband, (int)this_chunk,
        d_workspace);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // D2H — output layout: [band][local_j]  → unpack to out[band * np + pixel]
    float *h_out_chunk;
    CUDA_CHECK(cudaMallocHost(&h_out_chunk,
        this_chunk * g_nband * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(h_out_chunk, d_out,
        this_chunk * g_nband * sizeof(float),
        cudaMemcpyDeviceToHost));

    for (size_t k = 0; k < g_nband; ++k) {
      float *dst = out + k * g_np + chunk_start;
      float *src = h_out_chunk + k * chunk_size;
      memcpy(dst, src, this_chunk * sizeof(float));
    }
    CUDA_CHECK(cudaFreeHost(h_out_chunk));

    printf("  chunk %zu–%zu done\n", chunk_start, chunk_start + this_chunk - 1);
    chunk_start += this_chunk;
  }

  // ── write output ──────────────────────────────────────────────────────────
  str ofn(argv[argc - 1]);
  str ohfn(hdr_fn(ofn, true));
  bwrite(out, ofn, g_nrow, g_ncol, g_nband);
  run((str("cp -v ") + hdr_fn(str(argv[1])) + str(" ") + ohfn).c_str());

  printf("Output written to %s\n", argv[argc - 1]);

  // ── cleanup ───────────────────────────────────────────────────────────────
  cudaFree(d_dat);
  cudaFree(d_out);
  cudaFree(d_workspace);
  cudaFreeHost(h_stage);
  for (size_t i = 0; i < g_T; ++i) free(g_dat[i]);
  free(g_dat);
  free(out);

  return 0;
}
