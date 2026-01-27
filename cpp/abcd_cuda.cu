/*
 * 20260127 abcd_cuda.cu - NVIDIA CUDA-optimized nearest neighbor spectral inference
 * 
 * Usage: abcd_cuda A.bin B.bin C.bin [skip_factor] [skip_offset]
 
Building;
nvcc -O3 -o abcd_cuda abcd_cuda.cu

Running:
./abcd_cuda A.bin B.bin C.bin [skip_factor] [skip_offset]

 * Reads ENVI format rasters (32-bit float, BSQ interleave)
 * For each pixel in C, finds nearest spectral match in A, assigns corresponding B pixel to output */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define BLOCK_SIZE 256

// Parse ENVI header file for dimensions
void parse_header(const char* hdr_path, size_t* nrow, size_t* ncol, size_t* nband) {
    FILE* f = fopen(hdr_path, "r");
    if (!f) { fprintf(stderr, "Cannot open header: %s\n", hdr_path); exit(1); }
    
    char line[512];
    *nrow = *ncol = *nband = 0;
    
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "samples = %zu", ncol) == 1) continue;
        if (sscanf(line, "lines = %zu", nrow) == 1) continue;
        if (sscanf(line, "bands = %zu", nband) == 1) continue;
    }
    fclose(f);
    
    if (*nrow == 0 || *ncol == 0 || *nband == 0) {
        fprintf(stderr, "Invalid header: %s\n", hdr_path);
        exit(1);
    }
}

// Generate header filename from binary filename
char* header_filename(const char* bin_path) {
    size_t len = strlen(bin_path);
    char* hdr = (char*)malloc(len + 5);
    strcpy(hdr, bin_path);
    
    // Replace .bin with .hdr, or append .hdr
    if (len > 4 && strcmp(hdr + len - 4, ".bin") == 0) {
        strcpy(hdr + len - 4, ".hdr");
    } else {
        strcat(hdr, ".hdr");
    }
    return hdr;
}

// Read BSQ float32 raster
float* read_bsq(const char* path, size_t nrow, size_t ncol, size_t nband) {
    size_t n = nrow * ncol * nband;
    float* data = (float*)malloc(n * sizeof(float));
    if (!data) { fprintf(stderr, "Malloc failed for %s\n", path); exit(1); }
    
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path); exit(1); }
    
    if (fread(data, sizeof(float), n, f) != n) {
        fprintf(stderr, "Read error: %s\n", path);
        exit(1);
    }
    fclose(f);
    return data;
}

// Write BSQ float32 raster
void write_bsq(const char* path, float* data, size_t nrow, size_t ncol, size_t nband) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot create: %s\n", path); exit(1); }
    fwrite(data, sizeof(float), nrow * ncol * nband, f);
    fclose(f);
}

// Write ENVI header
void write_header(const char* path, size_t nrow, size_t ncol, size_t nband) {
    FILE* f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Cannot create header: %s\n", path); exit(1); }
    fprintf(f, "ENVI\n");
    fprintf(f, "samples = %zu\n", ncol);
    fprintf(f, "lines = %zu\n", nrow);
    fprintf(f, "bands = %zu\n", nband);
    fprintf(f, "header offset = 0\n");
    fprintf(f, "file type = ENVI Standard\n");
    fprintf(f, "data type = 4\n");  // 32-bit float
    fprintf(f, "interleave = bsq\n");
    fprintf(f, "byte order = 0\n");
    fclose(f);
}

// Check if pixel is bad (NaN, Inf, or all zeros for multi-band)
__device__ bool is_bad_pixel(const float* data, size_t idx, size_t np, int nband) {
    bool all_zero = true;
    for (int b = 0; b < nband; b++) {
        float v = data[np * b + idx];
        if (isnan(v) || isinf(v)) return true;
        if (v != 0.0f) all_zero = false;
    }
    return (nband > 1 && all_zero);
}

// Kernel to compute bad pixel masks
__global__ void compute_bad_mask_AB(const float* A, const float* B, int* bp, 
                                     size_t np, int nbA, int nbB) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= np) return;
    bp[i] = is_bad_pixel(A, i, np, nbA) || is_bad_pixel(B, i, np, nbB) ? 1 : 0;
}

__global__ void compute_bad_mask_C(const float* C, int* bp2, size_t np2, int nbC) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= np2) return;
    bp2[i] = is_bad_pixel(C, i, np2, nbC) ? 1 : 0;
}

// Main inference kernel - each thread processes one output pixel
__global__ void infer_kernel(const float* A, const float* B, const float* C,
                             const int* bp, const int* bp2, float* out,
                             size_t np, size_t np2, int nbA, int nbB,
                             size_t skip_f, size_t skip_off) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= np2) return;
    
    // Skip bad pixels in C
    if (bp2[i]) return;
    
    float min_dist = FLT_MAX;
    size_t min_idx = 0;
    
    // Find nearest neighbor in A (with skip sampling)
    for (size_t j = skip_off; j < np; j += skip_f) {
        if (bp[j]) continue;  // Skip bad pixels in A/B
        
        float dist = 0.0f;
        for (int k = 0; k < nbA; k++) {
            float diff = A[np * k + j] - C[np2 * k + i];
            dist += diff * diff;
        }
        
        if (dist < min_dist) {
            min_dist = dist;
            min_idx = j;
        }
    }
    
    // Assign nearest B pixel to output
    for (int k = 0; k < nbB; k++) {
        out[np2 * k + i] = B[np * k + min_idx];
    }
}

// Optimized kernel using shared memory for small band counts
__global__ void infer_kernel_shared(const float* A, const float* B, const float* C,
                                    const int* bp, const int* bp2, float* out,
                                    size_t np, size_t np2, int nbA, int nbB,
                                    size_t skip_f, size_t skip_off) {
    extern __shared__ float shared_C[];  // Cache C pixel spectrum
    
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= np2) return;
    if (bp2[i]) return;
    
    // Load this thread's C pixel into registers
    float c_vals[64];  // Max 64 bands in registers
    int local_nbA = min(nbA, 64);
    for (int k = 0; k < local_nbA; k++) {
        c_vals[k] = C[np2 * k + i];
    }
    
    float min_dist = FLT_MAX;
    size_t min_idx = 0;
    
    for (size_t j = skip_off; j < np; j += skip_f) {
        if (bp[j]) continue;
        
        float dist = 0.0f;
        for (int k = 0; k < local_nbA; k++) {
            float diff = A[np * k + j] - c_vals[k];
            dist += diff * diff;
        }
        
        if (dist < min_dist) {
            min_dist = dist;
            min_idx = j;
        }
    }
    
    for (int k = 0; k < nbB; k++) {
        out[np2 * k + i] = B[np * k + min_idx];
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s A.bin B.bin C.bin [skip_factor] [skip_offset]\n", argv[0]);
        fprintf(stderr, "  A.bin: Reference spectra (n bands)\n");
        fprintf(stderr, "  B.bin: Values to transfer (m bands)\n");
        fprintf(stderr, "  C.bin: Target spectra (n bands)\n");
        return 1;
    }
    
    size_t skip_f = (argc > 4) ? atol(argv[4]) : 1;
    size_t skip_off = (argc > 5) ? atol(argv[5]) : 0;
    
    // Parse headers
    size_t nr[3], nc[3], nb[3];
    for (int i = 0; i < 3; i++) {
        char* hdr = header_filename(argv[i + 1]);
        parse_header(hdr, &nr[i], &nc[i], &nb[i]);
        printf("File %d: %zu x %zu x %zu bands\n", i + 1, nr[i], nc[i], nb[i]);
        free(hdr);
    }
    
    // Validate dimensions
    if (nr[0] != nr[1] || nc[0] != nc[1]) {
        fprintf(stderr, "Error: A and B must have same spatial dimensions\n");
        return 1;
    }
    if (nb[0] != nb[2]) {
        fprintf(stderr, "Error: A and C must have same number of bands\n");
        return 1;
    }
    
    size_t np = nr[0] * nc[0];   // Pixels in A, B
    size_t np2 = nr[2] * nc[2];  // Pixels in C
    
    if (skip_f >= np) {
        fprintf(stderr, "Error: skip_factor >= number of pixels\n");
        return 1;
    }
    
    printf("Processing: np=%zu, np2=%zu, skip=%zu, offset=%zu\n", np, np2, skip_f, skip_off);
    
    // Read input files
    float* h_A = read_bsq(argv[1], nr[0], nc[0], nb[0]);
    float* h_B = read_bsq(argv[2], nr[1], nc[1], nb[1]);
    float* h_C = read_bsq(argv[3], nr[2], nc[2], nb[2]);
    
    // Allocate output (initialize to NaN)
    size_t out_size = np2 * nb[1];
    float* h_out = (float*)malloc(out_size * sizeof(float));
    for (size_t i = 0; i < out_size; i++) h_out[i] = NAN;
    
    // Allocate GPU memory
    float *d_A, *d_B, *d_C, *d_out;
    int *d_bp, *d_bp2;
    
    CUDA_CHECK(cudaMalloc(&d_A, np * nb[0] * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, np * nb[1] * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, np2 * nb[2] * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bp, np * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bp2, np2 * sizeof(int)));
    
    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_A, h_A, np * nb[0] * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, np * nb[1] * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, np2 * nb[2] * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out, h_out, out_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Compute bad pixel masks
    int blocks_np = (np + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocks_np2 = (np2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    compute_bad_mask_AB<<<blocks_np, BLOCK_SIZE>>>(d_A, d_B, d_bp, np, nb[0], nb[1]);
    compute_bad_mask_C<<<blocks_np2, BLOCK_SIZE>>>(d_C, d_bp2, np2, nb[2]);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Run main inference kernel
    printf("Running inference kernel...\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    if (nb[0] <= 64) {
        // Use optimized kernel with register caching for smaller band counts
        infer_kernel_shared<<<blocks_np2, BLOCK_SIZE>>>(
            d_A, d_B, d_C, d_bp, d_bp2, d_out,
            np, np2, nb[0], nb[1], skip_f, skip_off);
    } else {
        // Fallback for large band counts
        infer_kernel<<<blocks_np2, BLOCK_SIZE>>>(
            d_A, d_B, d_C, d_bp, d_bp2, d_out,
            np, np2, nb[0], nb[1], skip_f, skip_off);
    }
    
    cudaEventRecord(stop);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel time: %.2f ms\n", ms);
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_out, d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Generate output filename
    char out_bin[512], out_hdr[512];
    snprintf(out_bin, sizeof(out_bin), "abcd_%s_%s_%s_%zu_%zu.bin",
             argv[1], argv[2], argv[3], skip_f, skip_off);
    snprintf(out_hdr, sizeof(out_hdr), "abcd_%s_%s_%s_%zu_%zu.hdr",
             argv[1], argv[2], argv[3], skip_f, skip_off);
    
    // Write output
    write_bsq(out_bin, h_out, nr[2], nc[2], nb[1]);
    write_header(out_hdr, nr[2], nc[2], nb[1]);
    printf("Output written: %s\n", out_bin);
    
    // Cleanup
    free(h_A); free(h_B); free(h_C); free(h_out);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_out); cudaFree(d_bp); cudaFree(d_bp2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
