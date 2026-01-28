/*
nvcc -arch=sm_89 -O3 --use_fast_math -Xcompiler -O3 -std=c++17 -o dmat_cuda.exe dmat_cuda.cu -lcudart
CUDA translation of dmat clustering program
Optimized for NVIDIA L40s (Ada Lovelace architecture)
Original: kgc2020. 20201101
CUDA port maintains same outputs as original
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <cub/cub.cuh>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <ctime>
#include <string>
#include <vector>
#include <algorithm>
#include <sys/stat.h>

using namespace std;

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// L40s optimized parameters
#define BLOCK_SIZE_DMAT 256
#define BLOCK_SIZE_ASSIGN 256
#define TILE_SIZE 32
#define WARP_SIZE 32

// Global variables matching original
size_t np, nr, nc, nb, kmax, k_use;
size_t skip_factor = 1;
size_t n_sampled = 0;
size_t n_ddup;
time_t start_time;

// Helper functions
inline bool exists(const string& fn) {
    struct stat buffer;
    return (stat(fn.c_str(), &buffer) == 0);
}

inline size_t fsize(const string& fn) {
    struct stat st;
    if (stat(fn.c_str(), &st) != 0) return 0;
    return st.st_size;
}

inline string hdr_fn(const string& fn) {
    size_t pos = fn.rfind('.');
    if (pos != string::npos) {
        return fn.substr(0, pos) + ".hdr";
    }
    return fn + ".hdr";
}

inline string zero_pad(const string& s, size_t n) {
    string result = s;
    while (result.length() < n) result = "0" + result;
    return result;
}

void err(const char* msg) {
    fprintf(stderr, "Error: %s\n", msg);
    exit(1);
}

FILE* ropen(const string& fn) {
    FILE* f = fopen(fn.c_str(), "rb");
    if (!f) { fprintf(stderr, "Error opening: %s\n", fn.c_str()); exit(1); }
    return f;
}

FILE* wopen(const string& fn) {
    FILE* f = fopen(fn.c_str(), "wb");
    if (!f) { fprintf(stderr, "Error creating: %s\n", fn.c_str()); exit(1); }
    return f;
}

void hread(const string& fn, size_t& nr, size_t& nc, size_t& nb) {
    FILE* f = fopen(fn.c_str(), "r");
    if (!f) err("failed to open header file");
    char line[1024];
    nr = nc = nb = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, "samples")) sscanf(line, "samples = %zu", &nc);
        else if (strstr(line, "lines")) sscanf(line, "lines = %zu", &nr);
        else if (strstr(line, "bands")) sscanf(line, "bands = %zu", &nb);
    }
    fclose(f);
    if (nr == 0 || nc == 0 || nb == 0) err("failed to parse header");
}

void hwrite(const string& fn, size_t nr, size_t nc, size_t nb) {
    FILE* f = wopen(fn);
    fprintf(f, "ENVI\n");
    fprintf(f, "samples = %zu\n", nc);
    fprintf(f, "lines = %zu\n", nr);
    fprintf(f, "bands = %zu\n", nb);
    fprintf(f, "header offset = 0\n");
    fprintf(f, "file type = ENVI Standard\n");
    fprintf(f, "data type = 4\n");
    fprintf(f, "interleave = bip\n");
    fprintf(f, "byte order = 0\n");
    fclose(f);
}

float* bread(const string& fn, size_t nr, size_t nc, size_t nb) {
    size_t n = nr * nc * nb;
    float* dat = (float*)malloc(n * sizeof(float));
    if (!dat) err("malloc failed");
    FILE* f = ropen(fn);
    if (fread(dat, sizeof(float), n, f) != n) err("read error");
    fclose(f);
    return dat;
}

// Structure for distance-index pair (used in sorting)
struct DistIdx {
    float dist;
    size_t idx;
};

// Comparison for sorting (ascending distance)
__host__ __device__ bool operator<(const DistIdx& a, const DistIdx& b) {
    return a.dist < b.dist;
}

// =============================================================================
// CUDA Kernels optimized for L40s
// =============================================================================

// Kernel: Compute all pairwise distances for a batch of query points
// Uses shared memory tiling for efficient memory access
__global__ void compute_distances_kernel(
    const float* __restrict__ data,      // [np x nb] data points
    float* __restrict__ distances,        // [batch_size x np] output distances
    size_t np,                            // number of points
    size_t nb,                            // number of bands
    size_t batch_start,                   // starting index of this batch
    size_t batch_size                     // number of query points in batch
) {
    extern __shared__ float shared_mem[];
    float* tile_data = shared_mem;
    float* query_data = &shared_mem[TILE_SIZE * nb];
    
    size_t query_local = blockIdx.x;  // which query in the batch
    size_t query_idx = batch_start + query_local;
    
    if (query_local >= batch_size || query_idx >= np) return;
    
    // Load query point into shared memory (cooperatively)
    for (size_t b = threadIdx.x; b < nb; b += blockDim.x) {
        query_data[b] = data[query_idx * nb + b];
    }
    __syncthreads();
    
    // Each thread computes distance to multiple target points
    for (size_t target = threadIdx.x; target < np; target += blockDim.x) {
        float dist = 0.0f;
        for (size_t b = 0; b < nb; b++) {
            float diff = query_data[b] - data[target * nb + b];
            dist += diff * diff;
        }
        distances[query_local * np + target] = sqrtf(dist);
    }
}

// Kernel: Compute distances with better memory coalescing using tiling
__global__ void compute_distances_tiled_kernel(
    const float* __restrict__ data,
    float* __restrict__ distances,
    size_t np,
    size_t nb,
    size_t query_idx
) {
    size_t target = blockIdx.x * blockDim.x + threadIdx.x;
    if (target >= np) return;
    
    float dist = 0.0f;
    for (size_t b = 0; b < nb; b++) {
        float diff = data[query_idx * nb + b] - data[target * nb + b];
        dist += diff * diff;
    }
    distances[target] = sqrtf(dist);
}

// Kernel: Find k-nearest neighbors from precomputed distances
// Uses parallel reduction and bitonic sort
__global__ void find_knn_kernel(
    const float* __restrict__ all_distances,  // [np] distances for one query
    float* __restrict__ knn_distances,         // [kmax] output distances
    size_t* __restrict__ knn_indices,          // [kmax] output indices
    size_t np,
    size_t kmax
) {
    // This kernel uses a single block to find top-k
    // More sophisticated approaches would use radix select
    extern __shared__ char shared[];
    DistIdx* candidates = (DistIdx*)shared;
    
    size_t tid = threadIdx.x;
    
    // Each thread loads multiple elements
    for (size_t i = tid; i < np; i += blockDim.x) {
        candidates[i].dist = all_distances[i];
        candidates[i].idx = i;
    }
    __syncthreads();
    
    // Simple parallel selection for small k
    // For production, use CUB radix select or thrust
    if (tid < kmax) {
        // Find tid-th smallest
        float my_dist = candidates[tid].dist;
        size_t my_idx = candidates[tid].idx;
        size_t rank = 0;
        
        for (size_t i = 0; i < np; i++) {
            if (candidates[i].dist < my_dist || 
                (candidates[i].dist == my_dist && i < tid)) {
                rank++;
            }
        }
        
        if (rank < kmax) {
            knn_distances[rank] = my_dist;
            knn_indices[rank] = my_idx;
        }
    }
}

// Kernel: Assign labels to non-sampled points (find nearest sampled point)
__global__ void assign_labels_kernel(
    const float* __restrict__ full_data,      // [n_ddup x nb] all deduplicated data
    const float* __restrict__ sampled_data,   // [n_sampled x nb] sampled data
    const size_t* __restrict__ sample_idx,    // [n_sampled] indices of sampled points
    const size_t* __restrict__ sampled_labels,// [n_sampled] labels of sampled points
    size_t* __restrict__ labels,              // [n_ddup] output labels
    size_t n_ddup,
    size_t n_sampled,
    size_t nb
) {
    size_t point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= n_ddup) return;
    
    // Skip if already labeled (it's a sampled point)
    if (labels[point_idx] != 0) return;
    
    float min_dist = FLT_MAX;
    size_t nearest_sampled = 0;
    
    // Find nearest sampled point
    for (size_t s = 0; s < n_sampled; s++) {
        float dist = 0.0f;
        for (size_t b = 0; b < nb; b++) {
            float diff = full_data[point_idx * nb + b] - sampled_data[s * nb + b];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            nearest_sampled = s;
        }
    }
    
    labels[point_idx] = sampled_labels[nearest_sampled];
}

// Optimized version using shared memory for sampled data
__global__ void assign_labels_optimized_kernel(
    const float* __restrict__ full_data,
    const float* __restrict__ sampled_data,
    const size_t* __restrict__ sample_idx,
    size_t* __restrict__ labels,
    const size_t* __restrict__ sampled_labels,
    size_t n_ddup,
    size_t n_sampled,
    size_t nb
) {
    extern __shared__ float shared_sampled[];
    
    size_t point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float min_dist = FLT_MAX;
    size_t nearest_sampled = 0;
    
    // Load point data into registers
    float point_data[64];  // Assume nb <= 64
    if (point_idx < n_ddup && labels[point_idx] == 0) {
        for (size_t b = 0; b < nb && b < 64; b++) {
            point_data[b] = full_data[point_idx * nb + b];
        }
    }
    
    // Process sampled points in tiles
    for (size_t tile_start = 0; tile_start < n_sampled; tile_start += TILE_SIZE) {
        size_t tile_end = min(tile_start + TILE_SIZE, n_sampled);
        size_t tile_size = tile_end - tile_start;
        
        // Cooperatively load sampled data tile into shared memory
        for (size_t i = threadIdx.x; i < tile_size * nb; i += blockDim.x) {
            size_t s = i / nb;
            size_t b = i % nb;
            if (tile_start + s < n_sampled) {
                shared_sampled[s * nb + b] = sampled_data[(tile_start + s) * nb + b];
            }
        }
        __syncthreads();
        
        // Compute distances to this tile
        if (point_idx < n_ddup && labels[point_idx] == 0) {
            for (size_t s = 0; s < tile_size; s++) {
                float dist = 0.0f;
                for (size_t b = 0; b < nb && b < 64; b++) {
                    float diff = point_data[b] - shared_sampled[s * nb + b];
                    dist += diff * diff;
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    nearest_sampled = tile_start + s;
                }
            }
        }
        __syncthreads();
    }
    
    if (point_idx < n_ddup && labels[point_idx] == 0) {
        labels[point_idx] = sampled_labels[nearest_sampled];
    }
}

// =============================================================================
// Host-side k-NN computation using Thrust for sorting
// =============================================================================

void compute_knn_gpu(
    float* d_data,           // Device: [np x nb]
    float* d_dmat_d,         // Device: [np x kmax] output distances
    size_t* d_dmat_i,        // Device: [np x kmax] output indices
    size_t np,
    size_t nb,
    size_t kmax
) {
    // Allocate temporary storage for distances
    float* d_distances;
    CUDA_CHECK(cudaMalloc(&d_distances, np * sizeof(float)));
    
    // Allocate for sorting
    thrust::device_vector<float> d_dist_vec(np);
    thrust::device_vector<size_t> d_idx_vec(np);
    
    start_time = time(NULL);
    
    for (size_t q = 0; q < np; q++) {
        if (q % 100 == 0) {
            float pct = 100.0f * (float)q / (float)np;
            time_t now = time(NULL);
            double elapsed = difftime(now, start_time);
            double eta = (pct > 0) ? (elapsed / pct) * (100.0 - pct) : 0;
            printf(" %.2f%% ETA: %ds query: %zu/%zu\n", pct, (int)eta, q, np);
        }
        
        // Compute distances from query q to all points
        int num_blocks = (np + BLOCK_SIZE_DMAT - 1) / BLOCK_SIZE_DMAT;
        compute_distances_tiled_kernel<<<num_blocks, BLOCK_SIZE_DMAT>>>(
            d_data, d_distances, np, nb, q
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy to thrust vectors
        CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_dist_vec.data()),
                              d_distances, np * sizeof(float), cudaMemcpyDeviceToDevice));
        thrust::sequence(d_idx_vec.begin(), d_idx_vec.end());
        
        // Sort by distance (ascending)
        thrust::sort_by_key(d_dist_vec.begin(), d_dist_vec.end(), d_idx_vec.begin());
        
        // Copy top kmax to output
        CUDA_CHECK(cudaMemcpy(d_dmat_d + q * kmax,
                              thrust::raw_pointer_cast(d_dist_vec.data()),
                              kmax * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_dmat_i + q * kmax,
                              thrust::raw_pointer_cast(d_idx_vec.data()),
                              kmax * sizeof(size_t), cudaMemcpyDeviceToDevice));
    }
    
    CUDA_CHECK(cudaFree(d_distances));
}

// Batched version for better GPU utilization
void compute_knn_gpu_batched(
    float* d_data,
    float* d_dmat_d,
    size_t* d_dmat_i,
    size_t np,
    size_t nb,
    size_t kmax
) {
    const size_t BATCH_SIZE = 256;  // Process multiple queries at once
    
    // Allocate batch storage
    float* d_batch_distances;
    CUDA_CHECK(cudaMalloc(&d_batch_distances, BATCH_SIZE * np * sizeof(float)));
    
    thrust::device_vector<float> d_dist_vec(np);
    thrust::device_vector<size_t> d_idx_vec(np);
    
    start_time = time(NULL);
    
    for (size_t batch_start = 0; batch_start < np; batch_start += BATCH_SIZE) {
        size_t batch_size = min(BATCH_SIZE, np - batch_start);
        
        float pct = 100.0f * (float)batch_start / (float)np;
        time_t now = time(NULL);
        double elapsed = difftime(now, start_time);
        double eta = (pct > 0) ? (elapsed / pct) * (100.0 - pct) : 0;
        printf(" %.2f%% ETA: %ds batch: %zu-%zu/%zu\n", 
               pct, (int)eta, batch_start, batch_start + batch_size, np);
        
        // Compute distances for entire batch
        size_t shared_size = (TILE_SIZE + 1) * nb * sizeof(float);
        compute_distances_kernel<<<batch_size, BLOCK_SIZE_DMAT, shared_size>>>(
            d_data, d_batch_distances, np, nb, batch_start, batch_size
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Sort each query's distances
        for (size_t q = 0; q < batch_size; q++) {
            CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_dist_vec.data()),
                                  d_batch_distances + q * np,
                                  np * sizeof(float), cudaMemcpyDeviceToDevice));
            thrust::sequence(d_idx_vec.begin(), d_idx_vec.end());
            thrust::sort_by_key(d_dist_vec.begin(), d_dist_vec.end(), d_idx_vec.begin());
            
            size_t out_idx = batch_start + q;
            CUDA_CHECK(cudaMemcpy(d_dmat_d + out_idx * kmax,
                                  thrust::raw_pointer_cast(d_dist_vec.data()),
                                  kmax * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_dmat_i + out_idx * kmax,
                                  thrust::raw_pointer_cast(d_idx_vec.data()),
                                  kmax * sizeof(size_t), cudaMemcpyDeviceToDevice));
        }
    }
    
    CUDA_CHECK(cudaFree(d_batch_distances));
}

// =============================================================================
// Hill climbing (sequential on CPU - maintains exact original behavior)
// =============================================================================

class TopClimber {
public:
    float* rho;
    size_t* label;
    size_t* dmat_i;
    float* dmat_d;
    size_t kmax;
    size_t k_use;
    size_t next_label;
    vector<size_t> top_i;
    
    size_t top(size_t j) {
        if (label[j] > 0) return label[j];
        
        float rho_max = rho[j];
        size_t max_i = j;
        
        for (size_t i = 0; i < k_use; i++) {
            size_t ki = j * kmax + i;
            size_t ni = dmat_i[ki];
            if (rho[ni] > rho_max) {
                rho_max = rho[ni];
                max_i = ni;
            }
        }
        
        if (max_i != j) {
            return top(max_i);
        } else {
            label[j] = next_label++;
            top_i.push_back(j);
            return label[j];
        }
    }
};

// =============================================================================
// Write mean image (same output as original)
// =============================================================================

void write_mean_image(size_t k_use, size_t nr, size_t nc, size_t nb,
                      size_t* ddup_lookup, float* dat_full, size_t n_ddup,
                      size_t* label, vector<size_t>& top_i, size_t* sample_idx,
                      size_t skip_factor) {
    size_t np_img = nr * nc;
    
    string mean_fn = "mean/" + zero_pad(to_string(k_use), 5);
    FILE* f = wopen(mean_fn + ".bin");
    
    float* mean_out = (float*)malloc(np_img * nb * sizeof(float));
    
    for (size_t i = 0; i < np_img; i++) {
        size_t ddup_idx = ddup_lookup[i];
        size_t lab = label[ddup_idx];
        size_t top_idx = top_i[lab];
        size_t exemplar_ddup;
        
        if (skip_factor > 1) {
            exemplar_ddup = sample_idx[top_idx];
        } else {
            exemplar_ddup = top_idx;
        }
        
        for (size_t k = 0; k < nb; k++) {
            mean_out[i * nb + k] = dat_full[exemplar_ddup * nb + k];
        }
    }
    
    fwrite(mean_out, np_img * nb * sizeof(float), 1, f);
    free(mean_out);
    fclose(f);
    
    hwrite(mean_fn + ".hdr", nr, nc, nb);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    kmax = 1111;
    printf("dmat_cuda.exe (CUDA optimized for L40s)\n");
    
    if (argc < 3) {
        err("dmat_cuda.exe [input file bip format] [deduplication index file _dedup] [--nskip skip_factor]");
    }
    
    // Parse --nskip argument
    for (int a = 3; a < argc; a++) {
        if (strcmp(argv[a], "--nskip") == 0 && a + 1 < argc) {
            skip_factor = (size_t)atol(argv[a + 1]);
            if (skip_factor < 1) skip_factor = 1;
            printf("skip_factor: %zu\n", skip_factor);
            a++;
        }
    }
    
    // Initialize CUDA
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Using GPU: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("  Global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0*1024.0*1024.0));
    printf("  Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    CUDA_CHECK(cudaSetDevice(device));
    
    // Read input data
    string inf(argv[1]);
    string hfn = hdr_fn(inf);
    hread(hfn, nr, nc, nb);
    np = nr * nc;
    
    printf("Image: %zu x %zu x %zu bands\n", nr, nc, nb);
    
    // Read deduplication index
    string dpf(argv[2]);
    if (!exists(dpf)) err("failed to open deduplicated data index file");
    size_t ddup_s = fsize(dpf);
    n_ddup = ddup_s / sizeof(size_t);
    printf("deduplicated data count: %zu\n", n_ddup);
    
    size_t* ddup_i = (size_t*)malloc(n_ddup * sizeof(size_t));
    FILE* f = ropen(dpf);
    if (fread(ddup_i, sizeof(size_t), n_ddup, f) != n_ddup) err("unexpected record count read");
    fclose(f);
    
    // Read original data and create deduplicated set
    float* dat_0 = bread(inf, nr, nc, nb);
    float* dat = (float*)malloc(nb * n_ddup * sizeof(float));
    memset(dat, 0, nb * n_ddup * sizeof(float));
    
    size_t m = 0;
    for (size_t i = 0; i < n_ddup; i++) {
        size_t n = nb * ddup_i[i];
        for (size_t k = 0; k < nb; k++) {
            dat[m++] = dat_0[n++];
        }
    }
    
    // Scale to [0, 1]
    float* d_min = (float*)malloc(nb * sizeof(float));
    float* d_max = (float*)malloc(nb * sizeof(float));
    for (size_t i = 0; i < nb; i++) {
        d_min[i] = FLT_MAX;
        d_max[i] = -FLT_MAX;
    }
    
    for (size_t i = 0; i < n_ddup; i++) {
        m = i * nb;
        for (size_t k = 0; k < nb; k++) {
            float d = dat[m + k];
            if (d < d_min[k]) d_min[k] = d;
            if (d > d_max[k]) d_max[k] = d;
        }
    }
    
    for (size_t i = 0; i < n_ddup; i++) {
        m = i * nb;
        for (size_t k = 0; k < nb; k++) {
            dat[m + k] -= d_min[k];
            dat[m + k] /= (d_max[k] - d_min[k]);
        }
    }
    
    printf("min ");
    for (size_t i = 0; i < nb; i++) printf(" %f", d_min[i]);
    printf("\nmax ");
    for (size_t i = 0; i < nb; i++) printf(" %f", d_max[i]);
    printf("\n");
    
    // Read dedup lookup
    string of2 = inf + "_dedup_lookup";
    size_t* ddup_lookup = (size_t*)malloc(sizeof(size_t) * nr * nc);
    f = ropen(of2);
    if (fread(ddup_lookup, sizeof(size_t), nr * nc, f) != nr * nc) err("unexpected record read count");
    fclose(f);
    
    // Compute sampled indices
    n_sampled = (n_ddup + skip_factor - 1) / skip_factor;
    size_t* sample_idx = (size_t*)malloc(n_sampled * sizeof(size_t));
    float* sampled_dat = (float*)malloc(n_sampled * nb * sizeof(float));
    
    size_t si = 0;
    for (size_t i = 0; i < n_ddup && si < n_sampled; i += skip_factor) {
        sample_idx[si] = i;
        for (size_t k = 0; k < nb; k++) {
            sampled_dat[si * nb + k] = dat[i * nb + k];
        }
        si++;
    }
    n_sampled = si;
    printf("sampled data count: %zu (skip_factor=%zu)\n", n_sampled, skip_factor);
    
    np = n_sampled;
    kmax = kmax > np ? np : kmax;
    printf("kmax %zu\n", kmax);
    printf("np %zu\n", np);
    printf("(np^2 - n) / 2 = %f\n", (((float)np * (float)np) - (float)np) / 2.0f);
    
    // Allocate dmat on host
    float* dmat_d = (float*)malloc(np * kmax * sizeof(float));
    size_t* dmat_i = (size_t*)malloc(np * kmax * sizeof(size_t));
    
    string dmatd_fn = inf + "_" + to_string(kmax) + "_skip" + to_string(skip_factor) + "_dmat.d";
    string dmati_fn = inf + "_" + to_string(kmax) + "_skip" + to_string(skip_factor) + "_dmat.i";
    
    if (fsize(dmatd_fn) != np * kmax * sizeof(float)) {
        printf("Computing k-NN matrix on GPU...\n");
        
        // Allocate GPU memory
        float* d_data;
        float* d_dmat_d;
        size_t* d_dmat_i;
        
        CUDA_CHECK(cudaMalloc(&d_data, np * nb * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dmat_d, np * kmax * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dmat_i, np * kmax * sizeof(size_t)));
        
        CUDA_CHECK(cudaMemcpy(d_data, sampled_dat, np * nb * sizeof(float), cudaMemcpyHostToDevice));
        
        // Compute k-NN
        compute_knn_gpu_batched(d_data, d_dmat_d, d_dmat_i, np, nb, kmax);
        
        // Copy back to host
        CUDA_CHECK(cudaMemcpy(dmat_d, d_dmat_d, np * kmax * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dmat_i, d_dmat_i, np * kmax * sizeof(size_t), cudaMemcpyDeviceToHost));
        
        CUDA_CHECK(cudaFree(d_data));
        CUDA_CHECK(cudaFree(d_dmat_d));
        CUDA_CHECK(cudaFree(d_dmat_i));
        
        printf("end dmat_calc()\n");
        
        // Save dmat
        f = wopen(dmatd_fn);
        fwrite(dmat_d, np * kmax * sizeof(float), 1, f);
        fclose(f);
        
        f = wopen(dmati_fn);
        fwrite(dmat_i, np * kmax * sizeof(size_t), 1, f);
        fclose(f);
    } else {
        printf("Loading cached k-NN matrix...\n");
        f = ropen(dmatd_fn);
        fread(dmat_d, np * kmax * sizeof(float), 1, f);
        fclose(f);
        
        f = ropen(dmati_fn);
        fread(dmat_i, np * kmax * sizeof(size_t), 1, f);
        fclose(f);
    }
    
    // Allocate density and labels
    float* rho = (float*)malloc(np * sizeof(float));
    size_t* label = (size_t*)malloc(n_ddup * sizeof(size_t));
    
    // GPU buffers for label assignment
    float* d_full_data = nullptr;
    float* d_sampled_data = nullptr;
    size_t* d_sample_idx = nullptr;
    size_t* d_labels = nullptr;
    size_t* d_sampled_labels = nullptr;
    
    if (skip_factor > 1) {
        CUDA_CHECK(cudaMalloc(&d_full_data, n_ddup * nb * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sampled_data, n_sampled * nb * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sample_idx, n_sampled * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_labels, n_ddup * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_sampled_labels, n_sampled * sizeof(size_t)));
        
        CUDA_CHECK(cudaMemcpy(d_full_data, dat, n_ddup * nb * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sampled_data, sampled_dat, n_sampled * nb * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sample_idx, sample_idx, n_sampled * sizeof(size_t), cudaMemcpyHostToDevice));
    }
    
    f = wopen("n_class.csv");
    fprintf(f, "n_classes,k_use,label_file,mean_file");
    fclose(f);
    
    TopClimber climber;
    climber.rho = rho;
    climber.label = label;
    climber.dmat_i = dmat_i;
    climber.dmat_d = dmat_d;
    climber.kmax = kmax;
    
    long int last_number_of_classes = -1;
    
    for (k_use = 1; k_use <= kmax; k_use += 15) {
        climber.top_i.clear();
        climber.k_use = k_use;
        climber.next_label = 1;
        climber.top_i.push_back(0);
        
        if (k_use > kmax) err("k_use > kmax");
        
        // Compute density
        for (size_t i = 0; i < np; i++) {
            float d_avg = 0.0f;
            for (size_t j = 0; j < k_use; j++) {
                d_avg += dmat_d[i * kmax + j];
            }
            rho[i] = -d_avg;
        }
        
        // Clear labels and cluster sampled points
        memset(label, 0, n_ddup * sizeof(size_t));
        for (size_t i = 0; i < np; i++) {
            label[sample_idx[i]] = climber.top(i);
        }
        
        size_t number_of_classes = climber.next_label - 1;
        printf("k_use %zu n_classes %zu\n", k_use, number_of_classes);
        
        // Assign labels to non-sampled points using GPU
        if (skip_factor > 1) {
            printf("Assigning labels to non-sampled points on GPU...\n");
            
            // Copy current labels to device
            CUDA_CHECK(cudaMemcpy(d_labels, label, n_ddup * sizeof(size_t), cudaMemcpyHostToDevice));
            
            // Extract sampled labels
            size_t* sampled_labels = (size_t*)malloc(n_sampled * sizeof(size_t));
            for (size_t i = 0; i < n_sampled; i++) {
                sampled_labels[i] = label[sample_idx[i]];
            }
            CUDA_CHECK(cudaMemcpy(d_sampled_labels, sampled_labels, n_sampled * sizeof(size_t), cudaMemcpyHostToDevice));
            
            // Launch kernel
            int num_blocks = (n_ddup + BLOCK_SIZE_ASSIGN - 1) / BLOCK_SIZE_ASSIGN;
            size_t shared_size = TILE_SIZE * nb * sizeof(float);
            
            start_time = time(NULL);
            
            if (nb <= 64 && n_sampled <= 4096) {
                // Use optimized version with shared memory
                assign_labels_optimized_kernel<<<num_blocks, BLOCK_SIZE_ASSIGN, shared_size>>>(
                    d_full_data, d_sampled_data, d_sample_idx, d_labels, d_sampled_labels,
                    n_ddup, n_sampled, nb
                );
            } else {
                // Use basic version
                assign_labels_kernel<<<num_blocks, BLOCK_SIZE_ASSIGN>>>(
                    d_full_data, d_sampled_data, d_sample_idx, d_sampled_labels, d_labels,
                    n_ddup, n_sampled, nb
                );
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Copy labels back
            CUDA_CHECK(cudaMemcpy(label, d_labels, n_ddup * sizeof(size_t), cudaMemcpyDeviceToHost));
            
            free(sampled_labels);
            printf("Label assignment complete\n");
        }
        
        // Create output directories
        system("mkdir -p label");
        system("mkdir -p nearest");
        system("mkdir -p mean");
        system("mkdir -p out");
        
        // Write outputs
        size_t np_img = nr * nc;
        
        if (number_of_classes != (size_t)last_number_of_classes) {
            string lab_fn = "label/" + zero_pad(to_string(k_use), 5);
            string mean_fn = "mean/" + zero_pad(to_string(k_use), 5);
            
            // Write label binary
            f = wopen(lab_fn + ".bin");
            float* label_float = (float*)malloc(np_img * sizeof(float));
            for (size_t i = 0; i < np_img; i++) {
                label_float[i] = (float)label[ddup_lookup[i]];
            }
            fwrite(label_float, np_img * sizeof(float), 1, f);
            free(label_float);
            fclose(f);
            hwrite(lab_fn + ".hdr", nr, nc, 1);
            
            // Write mean image
            write_mean_image(k_use, nr, nc, nb, ddup_lookup, dat, n_ddup,
                           label, climber.top_i, sample_idx, skip_factor);
            
            // Append to CSV
            f = fopen("n_class.csv", "ab");
            if (!f) err("failed to open file: n_class.csv");
            fprintf(f, "\n%zu,%zu,%s.bin,%s.bin", number_of_classes, k_use, lab_fn.c_str(), mean_fn.c_str());
            fclose(f);
        }
        
        if (number_of_classes == 1) break;
        last_number_of_classes = number_of_classes;
    }
    
    // Cleanup
    if (skip_factor > 1) {
        CUDA_CHECK(cudaFree(d_full_data));
        CUDA_CHECK(cudaFree(d_sampled_data));
        CUDA_CHECK(cudaFree(d_sample_idx));
        CUDA_CHECK(cudaFree(d_labels));
        CUDA_CHECK(cudaFree(d_sampled_labels));
    }
    
    free(ddup_i);
    free(dat_0);
    free(dat);
    free(ddup_lookup);
    free(sample_idx);
    free(sampled_dat);
    free(dmat_d);
    free(dmat_i);
    free(rho);
    free(label);
    free(d_min);
    free(d_max);
    
    printf("Done.\n");
    return 0;
}
