/* kgc.cu -- GPU build of KGC, targeted at the NVIDIA L40S.
 *
 * This file compiles kgc.cpp with KGC_BUILD_KNN_EXTERNAL defined and
 * supplies a CUDA build_knn() in its place. Everything else -- argument
 * parsing, the ENVI I/O, the deduplication, the K sweep, the class
 * selection, the output files and every log line -- is the SAME source,
 * so the CPU and GPU builds cannot drift apart in behaviour. Only the
 * neighbour table is computed differently.
 *
 * That is deliberate: the neighbour table is O(n^2 * dim) distances and
 * n * kmax * 12 bytes of storage, which is essentially the entire cost
 * of the program. The sweep that follows is cheap by comparison and
 * reads the table sequentially, so moving it to the GPU would add
 * transfer traffic for very little gain.
 *
 * Build (the server does this automatically on first use):
 *   nvcc -O3 -std=c++14 -arch=sm_89 -Xcompiler -pthread kgc.cu -o kgc_gpu
 * sm_89 is Ada, which is the L40S. The launcher passes -arch matching
 * the detected device, falling back to sm_89.
 *
 * Numerical agreement with the CPU build: distances are computed in
 * float and square-rooted, exactly as the CPU does. Ties between equal
 * distances may be broken differently, because the sort is not the same
 * algorithm -- so a tie at the k-th neighbour can select a different but
 * equally valid neighbour. Everything downstream consumes the table as a
 * set, so this does not change the clustering.
 */

#define KGC_BUILD_KNN_EXTERNAL 1
#include "kgc.cpp"

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_OK(call)                                                    \
  do {                                                                   \
    cudaError_t _e = (call);                                             \
    if (_e != cudaSuccess) {                                             \
      std::fprintf(stderr, "[gpu] CUDA error at %s:%d -- %s\n",          \
                   __FILE__, __LINE__, cudaGetErrorString(_e));          \
      std::exit(3);                                                      \
    }                                                                    \
  } while (0)

/* Squared distances from one batch of query points to every point.
 *
 * One block per query, threads striding over the points. dim is small
 * (3-12 bands after selection), so the query's own vector is staged in
 * shared memory and re-read by every thread rather than being pulled
 * from global memory n times.
 */
__global__ void kgc_dist_kernel(const float* __restrict__ pts,
                                size_t n, size_t dim,
                                size_t q0, size_t nq,
                                float* __restrict__ out_d,
                                int* __restrict__ out_i) {
  extern __shared__ float qvec[];
  size_t qi = blockIdx.x;
  if (qi >= nq) return;
  size_t q = q0 + qi;

  for (size_t m = threadIdx.x; m < dim; m += blockDim.x)
    qvec[m] = pts[q * dim + m];
  __syncthreads();

  float* drow = out_d + qi * n;
  int*   irow = out_i + qi * n;

  for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
    float s = 0.f;
    const float* b = pts + i * dim;
#pragma unroll 4
    for (size_t m = 0; m < dim; m++) {
      float t = qvec[m] - b[m];
      s += t * t;
    }
    /* The self-match is excluded by the CPU version, so push it to the
     * end of the ordering rather than removing it -- that keeps every
     * row the same length, which the segmented sort requires. */
    drow[i] = (i == q) ? CUDART_INF_F : s;
    irow[i] = (int)i;
  }
}

/* How much device memory one batch needs, so the batch size can be
 * chosen from what the card actually has free rather than guessed. */
static size_t kgc_batch_bytes(size_t n, size_t batch) {
  /* keys in + out, values in + out */
  return batch * n * (sizeof(float) * 2 + sizeof(int) * 2);
}

void build_knn_cuda(const float* pts, size_t n, size_t dim, size_t kmax,
                    size_t threads, std::vector<float>& dd,
                    std::vector<size_t>& di) {
  (void)threads;  /* the GPU decides its own parallelism */

  dd.assign(n * kmax, 0.f);
  di.assign(n * kmax, 0);
  if (n == 0) return;

  int dev = 0;
  cudaDeviceProp prop;
  CUDA_OK(cudaGetDevice(&dev));
  CUDA_OK(cudaGetDeviceProperties(&prop, dev));
  size_t free_b = 0, total_b = 0;
  CUDA_OK(cudaMemGetInfo(&free_b, &total_b));
  std::fprintf(stderr,
               "[gpu] %s, %.1f GB free of %.1f GB, sm_%d%d\n",
               prop.name, free_b / 1e9, total_b / 1e9,
               prop.major, prop.minor);

  const size_t k = (kmax < n) ? kmax : n;

  /* Points live on the device for the whole run: n * dim * 4 bytes,
   * which is tiny next to the distance batches. */
  float* d_pts = 0;
  CUDA_OK(cudaMalloc(&d_pts, n * dim * sizeof(float)));
  CUDA_OK(cudaMemcpy(d_pts, pts, n * dim * sizeof(float),
                     cudaMemcpyHostToDevice));

  /* Choose the batch size from free memory, leaving a margin for the
   * sort's own temporary storage. Smaller batches are always correct,
   * just slower, so erring low is safe. */
  size_t budget = (size_t)(free_b * 0.60);
  size_t batch = 1;
  while (batch < 512 && kgc_batch_bytes(n, batch * 2) < budget) batch *= 2;
  if (batch * n == 0) batch = 1;
  std::fprintf(stderr,
               "[gpu] batch %zu queries, %.2f GB per batch, "
               "%zu points x %zu dims, kmax %zu\n",
               batch, kgc_batch_bytes(n, batch) / 1e9, n, dim, kmax);

  float *d_din = 0, *d_dout = 0;
  int   *d_iin = 0, *d_iout = 0;
  CUDA_OK(cudaMalloc(&d_din,  batch * n * sizeof(float)));
  CUDA_OK(cudaMalloc(&d_dout, batch * n * sizeof(float)));
  CUDA_OK(cudaMalloc(&d_iin,  batch * n * sizeof(int)));
  CUDA_OK(cudaMalloc(&d_iout, batch * n * sizeof(int)));

  /* Segment offsets: one segment per query row. */
  std::vector<int> h_off(batch + 1);
  for (size_t i = 0; i <= batch; i++) h_off[i] = (int)(i * n);
  int* d_off = 0;
  CUDA_OK(cudaMalloc(&d_off, (batch + 1) * sizeof(int)));
  CUDA_OK(cudaMemcpy(d_off, &h_off[0], (batch + 1) * sizeof(int),
                     cudaMemcpyHostToDevice));

  void*  d_tmp = 0;
  size_t tmp_bytes = 0;
  CUDA_OK(cub::DeviceSegmentedRadixSort::SortPairs(
      0, tmp_bytes, d_din, d_dout, d_iin, d_iout,
      (int)(batch * n), (int)batch, d_off, d_off + 1));
  CUDA_OK(cudaMalloc(&d_tmp, tmp_bytes));
  std::fprintf(stderr, "[gpu] sort scratch %.2f GB\n", tmp_bytes / 1e9);

  std::vector<float> h_d(batch * (size_t)k);
  std::vector<int>   h_i(batch * (size_t)k);

  const int tpb = 256;
  size_t done = 0;
  for (size_t q0 = 0; q0 < n; q0 += batch) {
    size_t nq = (q0 + batch <= n) ? batch : (n - q0);

    kgc_dist_kernel<<<(unsigned)nq, tpb, dim * sizeof(float)>>>(
        d_pts, n, dim, q0, nq, d_din, d_iin);
    CUDA_OK(cudaGetLastError());

    CUDA_OK(cub::DeviceSegmentedRadixSort::SortPairs(
        d_tmp, tmp_bytes, d_din, d_dout, d_iin, d_iout,
        (int)(nq * n), (int)nq, d_off, d_off + 1));

    /* Copy back only the first k of each row. */
    for (size_t r = 0; r < nq; r++) {
      CUDA_OK(cudaMemcpy(&h_d[r * k], d_dout + r * n, k * sizeof(float),
                         cudaMemcpyDeviceToHost));
      CUDA_OK(cudaMemcpy(&h_i[r * k], d_iout + r * n, k * sizeof(int),
                         cudaMemcpyDeviceToHost));
    }

    for (size_t r = 0; r < nq; r++) {
      size_t q = q0 + r;
      for (size_t m = 0; m < k; m++) {
        /* sqrt here, matching the CPU build exactly. */
        dd[q * kmax + m] = std::sqrt(h_d[r * k + m]);
        di[q * kmax + m] = (size_t)h_i[r * k + m];
      }
      /* Pad exactly as the CPU build does when kmax > n-1. */
      for (size_t m = k; m < kmax; m++) {
        dd[q * kmax + m] = k ? dd[q * kmax + k - 1] : 0.f;
        di[q * kmax + m] = k ? di[q * kmax + k - 1] : q;
      }
    }

    done += nq;
    /* Same progress line as the CPU build, so the server's log parsing
     * and the ETA it drives work identically for both. */
    std::fprintf(stderr, "  neighbours %5.1f%%\r", 100.0 * done / n);
    std::fflush(stderr);
  }
  std::fprintf(stderr, "  neighbours 100.0%%\n");

  cudaFree(d_tmp);
  cudaFree(d_off);
  cudaFree(d_iout);
  cudaFree(d_iin);
  cudaFree(d_dout);
  cudaFree(d_din);
  cudaFree(d_pts);
}
