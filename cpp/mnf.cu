/** 
 * 20260122 Minimum Noise Fraction (MNF) Transform - CUDA Implementation
 * MNF transform steps:
 * 1. Estimate noise covariance matrix from spatial differences
 * 2. Compute data covariance matrix
 * 3. Solve generalized eigenvalue problem: Σ_data * v = λ * Σ_noise * v
 * 4. Project data onto eigenvectors sorted by decreasing SNR (eigenvalue)
 * 
 * Usage: mnf_transform <input_raster> <output_raster> [n_components]
 
To compile:
nvcc -O3 -arch=sm_89 mnf_transform.cu -o mnf_transform \
    -lcublas -lcusolver \
    $(gdal-config --cflags) $(gdal-config --libs)

Usage: 
 mnf_transform <input_raster> <output_raster> [options]

Options:
  -n <num>    Number of MNF components to output (default: all)
  -i          Compute inverse transform and reconstruction RMSE
  -q          Quiet mode
 
Examples: 
./mnf_transform hyperspectral.tif mnf_output.tif
./mnf_transform aviris.img mnf.tif -n 30
./mnf_transform input.envi output.tif -n 50 -i

 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <gdal.h>
#include <gdal_priv.h>
#include <cpl_conv.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// L40S optimized parameters
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define TILE_DIM 32

// Error checking macros
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
        return -1; \
    } \
} while(0)

#define CUSOLVER_CHECK(call) do { \
    cusolverStatus_t status = call; \
    if (status != CUSOLVER_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSOLVER error at %s:%d: %d\n", __FILE__, __LINE__, status); \
        return -1; \
    } \
} while(0)

/**
 * Kernel: Compute mean per band
 * data: [bands x pixels] in row-major
 */
__global__ void compute_band_means(const float* __restrict__ data,
                                   float* __restrict__ means,
                                   int bands, int pixels) {
    extern __shared__ float sdata[];
    
    int band = blockIdx.x;
    int tid = threadIdx.x;
    
    if (band >= bands) return;
    
    // Each thread accumulates multiple pixels
    float sum = 0.0f;
    for (int p = tid; p < pixels; p += blockDim.x) {
        sum += data[band * pixels + p];
    }
    sdata[tid] = sum;
    __syncthreads();
    
    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        means[band] = sdata[0] / (float)pixels;
    }
}

/**
 * Kernel: Center data (subtract mean)
 */
__global__ void center_data(float* __restrict__ data,
                           const float* __restrict__ means,
                           int bands, int pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = bands * pixels;
    
    if (idx < total) {
        int band = idx / pixels;
        data[idx] -= means[band];
    }
}

/**
 * Kernel: Compute noise via horizontal spatial difference
 * noise[b, p] = data[b, p+1] - data[b, p] for valid pixels
 * Assumes 2D image stored row-major with known width
 */
__global__ void compute_noise_horizontal(const float* __restrict__ data,
                                         float* __restrict__ noise,
                                         int bands, int width, int height) {
    int pixels = width * height;
    int noise_pixels = (width - 1) * height;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= bands * noise_pixels) return;
    
    int band = idx / noise_pixels;
    int np = idx % noise_pixels;
    int row = np / (width - 1);
    int col = np % (width - 1);
    int p1 = row * width + col;
    int p2 = p1 + 1;
    
    noise[idx] = (data[band * pixels + p2] - data[band * pixels + p1]) * 0.5f;
}

/**
 * MNF Context - holds all GPU resources
 */
struct MNFContext {
    cublasHandle_t cublas;
    cusolverDnHandle_t cusolver;
    
    // Device memory
    float* d_data;           // [bands x pixels] input/output
    float* d_means;          // [bands]
    float* d_noise;          // [bands x noise_pixels]
    float* d_cov_data;       // [bands x bands]
    float* d_cov_noise;      // [bands x bands]
    float* d_eigenvalues;    // [bands]
    float* d_eigenvectors;   // [bands x bands]
    float* d_work;           // workspace
    int* d_info;
    
    int bands;
    int pixels;
    int width;
    int height;
    int noise_pixels;
    
    int init(int b, int w, int h) {
        bands = b;
        width = w;
        height = h;
        pixels = w * h;
        noise_pixels = (w - 1) * h;
        
        CUBLAS_CHECK(cublasCreate(&cublas));
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
        
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_data, bands * pixels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_means, bands * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_noise, bands * noise_pixels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cov_data, bands * bands * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cov_noise, bands * bands * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_eigenvalues, bands * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_eigenvectors, bands * bands * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
        
        // Query workspace size for generalized eigensolver
        int lwork;
        CUSOLVER_CHECK(cusolverDnSsygvd_bufferSize(
            cusolver, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_UPPER, bands, d_cov_data, bands,
            d_cov_noise, bands, d_eigenvalues, &lwork));
        
        CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(float)));
        
        return 0;
    }
    
    void destroy() {
        cudaFree(d_data);
        cudaFree(d_means);
        cudaFree(d_noise);
        cudaFree(d_cov_data);
        cudaFree(d_cov_noise);
        cudaFree(d_eigenvalues);
        cudaFree(d_eigenvectors);
        cudaFree(d_work);
        cudaFree(d_info);
        cublasDestroy(cublas);
        cusolverDnDestroy(cusolver);
    }
};

/**
 * Main MNF Transform Function
 * 
 * @param ctx      Initialized MNF context
 * @param h_data   Host input data [bands x pixels], modified in place
 * @param n_components  Number of MNF components to keep (0 = all)
 * @return 0 on success, -1 on error
 */
int mnf_transform(MNFContext& ctx, float* h_data, int n_components = 0) {
    if (n_components <= 0 || n_components > ctx.bands) {
        n_components = ctx.bands;
    }
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(ctx.d_data, h_data, 
                          ctx.bands * ctx.pixels * sizeof(float),
                          cudaMemcpyHostToDevice));
    
    // 1. Compute and subtract means
    int shared_mem = BLOCK_SIZE * sizeof(float);
    compute_band_means<<<ctx.bands, BLOCK_SIZE, shared_mem>>>(
        ctx.d_data, ctx.d_means, ctx.bands, ctx.pixels);
    
    int total_elements = ctx.bands * ctx.pixels;
    int grid_size = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    center_data<<<grid_size, BLOCK_SIZE>>>(
        ctx.d_data, ctx.d_means, ctx.bands, ctx.pixels);
    
    // 2. Compute noise estimate (horizontal differences)
    int noise_total = ctx.bands * ctx.noise_pixels;
    grid_size = (noise_total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_noise_horizontal<<<grid_size, BLOCK_SIZE>>>(
        ctx.d_data, ctx.d_noise, ctx.bands, ctx.width, ctx.height);
    
    // 3. Compute noise covariance: Σ_noise = (1/n) * noise * noise^T
    float scale = 1.0f / (float)ctx.noise_pixels;
    CUBLAS_CHECK(cublasSgemm(ctx.cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                             ctx.bands, ctx.bands, ctx.noise_pixels,
                             &scale, ctx.d_noise, ctx.bands,
                             ctx.d_noise, ctx.bands,
                             &beta, ctx.d_cov_noise, ctx.bands));
    
    // 4. Compute data covariance: Σ_data = (1/n) * data * data^T
    scale = 1.0f / (float)ctx.pixels;
    CUBLAS_CHECK(cublasSgemm(ctx.cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                             ctx.bands, ctx.bands, ctx.pixels,
                             &scale, ctx.d_data, ctx.bands,
                             ctx.d_data, ctx.bands,
                             &beta, ctx.d_cov_data, ctx.bands));
    
    // 5. Solve generalized eigenvalue problem: Σ_data * v = λ * Σ_noise * v
    // cusolverDnSsygvd solves A*x = λ*B*x, returns eigenvalues in ascending order
    int lwork;
    CUSOLVER_CHECK(cusolverDnSsygvd_bufferSize(
        ctx.cusolver, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER, ctx.bands, ctx.d_cov_data, ctx.bands,
        ctx.d_cov_noise, ctx.bands, ctx.d_eigenvalues, &lwork));
    
    CUSOLVER_CHECK(cusolverDnSsygvd(
        ctx.cusolver, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER, ctx.bands, ctx.d_cov_data, ctx.bands,
        ctx.d_cov_noise, ctx.bands, ctx.d_eigenvalues, ctx.d_work, lwork, ctx.d_info));
    
    // Eigenvectors are now in d_cov_data, eigenvalues in ascending order
    // Copy eigenvectors for later use (they get overwritten)
    CUDA_CHECK(cudaMemcpy(ctx.d_eigenvectors, ctx.d_cov_data,
                          ctx.bands * ctx.bands * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    
    // 6. Project data onto eigenvectors (reversed order for descending SNR)
    // MNF_data = eigenvectors^T * centered_data
    // Since eigenvalues are ascending, we want last columns first
    // Use full projection then reorder on host, or project reversed
    
    // For efficiency, project all and let user select components
    // output = V^T * data, where V columns are eigenvectors
    CUBLAS_CHECK(cublasSgemm(ctx.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             ctx.bands, ctx.pixels, ctx.bands,
                             &alpha, ctx.d_eigenvectors, ctx.bands,
                             ctx.d_data, ctx.bands,
                             &beta, ctx.d_data, ctx.bands));
    
    // Copy result back - note: bands are in ascending eigenvalue order
    // Reverse on host for descending SNR (highest SNR = first band)
    float* temp = (float*)malloc(ctx.bands * ctx.pixels * sizeof(float));
    CUDA_CHECK(cudaMemcpy(temp, ctx.d_data,
                          ctx.bands * ctx.pixels * sizeof(float),
                          cudaMemcpyDeviceToHost));
    
    // Reverse band order so highest SNR is first
    for (int b = 0; b < ctx.bands; b++) {
        int src_band = ctx.bands - 1 - b;
        memcpy(&h_data[b * ctx.pixels], &temp[src_band * ctx.pixels],
               ctx.pixels * sizeof(float));
    }
    free(temp);
    
    return 0;
}

/**
 * Get eigenvalues (SNR values) after transform
 * Returns in descending order (highest SNR first)
 */
int mnf_get_eigenvalues(MNFContext& ctx, float* h_eigenvalues) {
    float* temp = (float*)malloc(ctx.bands * sizeof(float));
    CUDA_CHECK(cudaMemcpy(temp, ctx.d_eigenvalues,
                          ctx.bands * sizeof(float),
                          cudaMemcpyDeviceToHost));
    
    // Reverse for descending order
    for (int i = 0; i < ctx.bands; i++) {
        h_eigenvalues[i] = temp[ctx.bands - 1 - i];
    }
    free(temp);
    return 0;
}

/**
 * Inverse MNF Transform
 * Projects MNF data back to original space
 */
int mnf_inverse_transform(MNFContext& ctx, float* h_data, int n_components = 0) {
    if (n_components <= 0 || n_components > ctx.bands) {
        n_components = ctx.bands;
    }
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Reverse input data band order (user gives descending SNR, we need ascending)
    float* temp = (float*)malloc(ctx.bands * ctx.pixels * sizeof(float));
    for (int b = 0; b < ctx.bands; b++) {
        int dst_band = ctx.bands - 1 - b;
        memcpy(&temp[dst_band * ctx.pixels], &h_data[b * ctx.pixels],
               ctx.pixels * sizeof(float));
    }
    
    // Zero out components beyond n_components (in ascending order, first bands)
    int zero_bands = ctx.bands - n_components;
    memset(temp, 0, zero_bands * ctx.pixels * sizeof(float));
    
    CUDA_CHECK(cudaMemcpy(ctx.d_data, temp,
                          ctx.bands * ctx.pixels * sizeof(float),
                          cudaMemcpyHostToDevice));
    free(temp);
    
    // Inverse projection: data = V * MNF_data
    CUBLAS_CHECK(cublasSgemm(ctx.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                             ctx.bands, ctx.pixels, ctx.bands,
                             &alpha, ctx.d_eigenvectors, ctx.bands,
                             ctx.d_data, ctx.bands,
                             &beta, ctx.d_data, ctx.bands));
    
    // Add means back
    int total_elements = ctx.bands * ctx.pixels;
    int grid_size = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // We need to add means (negative of center_data)
    // Quick kernel to add means
    float* d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, ctx.bands * ctx.pixels * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_temp, ctx.d_data,
                          ctx.bands * ctx.pixels * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    
    // Reuse center_data with negated means
    float* h_means = (float*)malloc(ctx.bands * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_means, ctx.d_means, ctx.bands * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < ctx.bands; i++) h_means[i] = -h_means[i];
    CUDA_CHECK(cudaMemcpy(ctx.d_means, h_means, ctx.bands * sizeof(float),
                          cudaMemcpyHostToDevice));
    
    center_data<<<grid_size, BLOCK_SIZE>>>(
        ctx.d_data, ctx.d_means, ctx.bands, ctx.pixels);
    
    // Restore original means
    for (int i = 0; i < ctx.bands; i++) h_means[i] = -h_means[i];
    CUDA_CHECK(cudaMemcpy(ctx.d_means, h_means, ctx.bands * sizeof(float),
                          cudaMemcpyHostToDevice));
    free(h_means);
    cudaFree(d_temp);
    
    CUDA_CHECK(cudaMemcpy(h_data, ctx.d_data,
                          ctx.bands * ctx.pixels * sizeof(float),
                          cudaMemcpyDeviceToHost));
    
    return 0;
}

// ============================================================================
// GDAL I/O Functions
// ============================================================================

/**
 * Read raster data using GDAL
 * Returns data in [bands x pixels] layout (band-interleaved by pixel)
 */
float* read_raster(const char* filename, int* out_bands, int* out_width, int* out_height) {
    GDALDataset* dataset = (GDALDataset*)GDALOpen(filename, GA_ReadOnly);
    if (!dataset) {
        fprintf(stderr, "Error: Cannot open %s\n", filename);
        return nullptr;
    }
    
    int width = dataset->GetRasterXSize();
    int height = dataset->GetRasterYSize();
    int bands = dataset->GetRasterCount();
    int pixels = width * height;
    
    printf("Reading: %s\n", filename);
    printf("  Dimensions: %d x %d x %d bands\n", width, height, bands);
    
    float* data = (float*)malloc(bands * pixels * sizeof(float));
    if (!data) {
        fprintf(stderr, "Error: Failed to allocate memory for raster data\n");
        GDALClose(dataset);
        return nullptr;
    }
    
    // Read each band
    for (int b = 0; b < bands; b++) {
        GDALRasterBand* band = dataset->GetRasterBand(b + 1);  // GDAL bands are 1-indexed
        CPLErr err = band->RasterIO(GF_Read, 0, 0, width, height,
                                    &data[b * pixels], width, height,
                                    GDT_Float32, 0, 0);
        if (err != CE_None) {
            fprintf(stderr, "Error: Failed to read band %d\n", b + 1);
            free(data);
            GDALClose(dataset);
            return nullptr;
        }
    }
    
    *out_bands = bands;
    *out_width = width;
    *out_height = height;
    
    GDALClose(dataset);
    return data;
}

/**
 * Write raster data using GDAL (GeoTIFF format)
 * Copies georeferencing from source dataset if provided
 */
int write_raster(const char* filename, const char* src_filename,
                 float* data, int bands, int width, int height,
                 const float* eigenvalues = nullptr) {
    
    // Open source for georeferencing info
    GDALDataset* src_dataset = nullptr;
    if (src_filename) {
        src_dataset = (GDALDataset*)GDALOpen(src_filename, GA_ReadOnly);
    }
    
    // Create output dataset
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (!driver) {
        fprintf(stderr, "Error: GTiff driver not available\n");
        if (src_dataset) GDALClose(src_dataset);
        return -1;
    }
    
    // Creation options for efficient storage
    char** options = nullptr;
    options = CSLSetNameValue(options, "COMPRESS", "LZW");
    options = CSLSetNameValue(options, "TILED", "YES");
    options = CSLSetNameValue(options, "BIGTIFF", "IF_SAFER");
    
    GDALDataset* dst_dataset = driver->Create(filename, width, height, bands,
                                               GDT_Float32, options);
    CSLDestroy(options);
    
    if (!dst_dataset) {
        fprintf(stderr, "Error: Cannot create %s\n", filename);
        if (src_dataset) GDALClose(src_dataset);
        return -1;
    }
    
    // Copy georeferencing from source
    if (src_dataset) {
        double geotransform[6];
        if (src_dataset->GetGeoTransform(geotransform) == CE_None) {
            dst_dataset->SetGeoTransform(geotransform);
        }
        
        const char* projection = src_dataset->GetProjectionRef();
        if (projection && strlen(projection) > 0) {
            dst_dataset->SetProjection(projection);
        }
    }
    
    // Write each band
    int pixels = width * height;
    for (int b = 0; b < bands; b++) {
        GDALRasterBand* band = dst_dataset->GetRasterBand(b + 1);
        
        CPLErr err = band->RasterIO(GF_Write, 0, 0, width, height,
                                    &data[b * pixels], width, height,
                                    GDT_Float32, 0, 0);
        if (err != CE_None) {
            fprintf(stderr, "Error: Failed to write band %d\n", b + 1);
            GDALClose(dst_dataset);
            if (src_dataset) GDALClose(src_dataset);
            return -1;
        }
        
        // Set band description with eigenvalue if available
        if (eigenvalues) {
            char desc[64];
            snprintf(desc, sizeof(desc), "MNF Band %d (SNR: %.4f)", b + 1, eigenvalues[b]);
            band->SetDescription(desc);
        }
    }
    
    printf("Written: %s\n", filename);
    printf("  Dimensions: %d x %d x %d bands\n", width, height, bands);
    
    GDALClose(dst_dataset);
    if (src_dataset) GDALClose(src_dataset);
    
    return 0;
}

/**
 * Write eigenvalues to a text file
 */
int write_eigenvalues(const char* base_filename, const float* eigenvalues, int bands) {
    char filename[512];
    snprintf(filename, sizeof(filename), "%s_eigenvalues.txt", base_filename);
    
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Warning: Cannot write eigenvalues to %s\n", filename);
        return -1;
    }
    
    fprintf(fp, "# MNF Eigenvalues (SNR) - Descending order\n");
    fprintf(fp, "# Band\tEigenvalue\n");
    for (int i = 0; i < bands; i++) {
        fprintf(fp, "%d\t%.6f\n", i + 1, eigenvalues[i]);
    }
    
    fclose(fp);
    printf("Eigenvalues written to: %s\n", filename);
    return 0;
}

// ============================================================================
// Main Program
// ============================================================================

void print_usage(const char* program) {
    printf("MNF (Minimum Noise Fraction) Transform - CUDA/L40S Optimized\n\n");
    printf("Usage: %s <input_raster> <output_raster> [options]\n\n", program);
    printf("Options:\n");
    printf("  -n <num>    Number of components to output (default: all)\n");
    printf("  -i          Also compute inverse transform (reconstruction)\n");
    printf("  -q          Quiet mode (minimal output)\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s input.tif mnf_output.tif\n", program);
    printf("  %s input.tif mnf_output.tif -n 20\n", program);
    printf("  %s hyperspectral.img mnf.tif -n 50\n", program);
    printf("\n");
    printf("Supported formats: Any GDAL-readable raster (GeoTIFF, ENVI, HDF, etc.)\n");
    printf("Output format: GeoTIFF with LZW compression\n");
}

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }
    
    const char* input_file = argv[1];
    const char* output_file = argv[2];
    int n_components = 0;  // 0 = all
    bool do_inverse = false;
    bool quiet = false;
    
    // Parse optional arguments
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_components = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0) {
            do_inverse = true;
        } else if (strcmp(argv[i], "-q") == 0) {
            quiet = true;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Initialize GDAL
    GDALAllRegister();
    
    // Read input raster
    int bands, width, height;
    float* data = read_raster(input_file, &bands, &width, &height);
    if (!data) {
        return 1;
    }
    
    int pixels = width * height;
    
    // Validate n_components
    if (n_components <= 0 || n_components > bands) {
        n_components = bands;
    }
    
    if (!quiet) {
        printf("\nMNF Transform Configuration:\n");
        printf("  Input bands: %d\n", bands);
        printf("  Output components: %d\n", n_components);
        printf("  Image size: %d x %d (%d pixels)\n", width, height, pixels);
        printf("  Data size: %.2f MB\n", (bands * pixels * sizeof(float)) / (1024.0 * 1024.0));
    }
    
    // Initialize CUDA context
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Initialize MNF
    MNFContext ctx;
    if (ctx.init(bands, width, height) != 0) {
        fprintf(stderr, "Error: Failed to initialize CUDA MNF context\n");
        free(data);
        return 1;
    }
    
    if (!quiet) printf("\nRunning MNF transform...\n");
    
    cudaEventRecord(start);
    
    // Run MNF transform
    if (mnf_transform(ctx, data, n_components) != 0) {
        fprintf(stderr, "Error: MNF transform failed\n");
        ctx.destroy();
        free(data);
        return 1;
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    if (!quiet) printf("MNF transform completed in %.2f ms\n", ms);
    
    // Get eigenvalues
    float* eigenvalues = (float*)malloc(bands * sizeof(float));
    mnf_get_eigenvalues(ctx, eigenvalues);
    
    if (!quiet) {
        printf("\nEigenvalues (SNR) summary:\n");
        int show_top = (bands < 10) ? bands : 10;
        for (int i = 0; i < show_top; i++) {
            printf("  Band %3d: %.4f\n", i + 1, eigenvalues[i]);
        }
        if (bands > 10) {
            printf("  ...\n");
            for (int i = bands - 3; i < bands; i++) {
                printf("  Band %3d: %.4f\n", i + 1, eigenvalues[i]);
            }
        }
    }
    
    // Write output raster (only n_components bands)
    if (write_raster(output_file, input_file, data, n_components, width, height, eigenvalues) != 0) {
        ctx.destroy();
        free(data);
        free(eigenvalues);
        return 1;
    }
    
    // Write eigenvalues to text file
    // Strip extension from output filename for eigenvalue file
    char base_name[512];
    strncpy(base_name, output_file, sizeof(base_name) - 1);
    char* dot = strrchr(base_name, '.');
    if (dot) *dot = '\0';
    write_eigenvalues(base_name, eigenvalues, bands);
    
    // Optional: inverse transform for reconstruction quality check
    if (do_inverse) {
        if (!quiet) printf("\nComputing inverse transform...\n");
        
        // Reload original data
        float* original = read_raster(input_file, &bands, &width, &height);
        
        // We need to reload the MNF data for inverse
        float* mnf_data = (float*)malloc(bands * pixels * sizeof(float));
        memcpy(mnf_data, data, n_components * pixels * sizeof(float));
        // Zero out unused components
        memset(&mnf_data[n_components * pixels], 0, (bands - n_components) * pixels * sizeof(float));
        
        if (mnf_inverse_transform(ctx, mnf_data, n_components) != 0) {
            fprintf(stderr, "Warning: Inverse transform failed\n");
        } else {
            // Compute reconstruction error
            double mse = 0.0;
            for (int i = 0; i < bands * pixels; i++) {
                double diff = original[i] - mnf_data[i];
                mse += diff * diff;
            }
            mse /= (bands * pixels);
            printf("Reconstruction RMSE (%d components): %.6f\n", n_components, sqrt(mse));
            
            // Write reconstructed image
            char recon_file[512];
            snprintf(recon_file, sizeof(recon_file), "%s_reconstructed.tif", base_name);
            write_raster(recon_file, input_file, mnf_data, bands, width, height);
        }
        
        free(original);
        free(mnf_data);
    }
    
    // Cleanup
    ctx.destroy();
    free(data);
    free(eigenvalues);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    if (!quiet) printf("\nDone.\n");
    return 0;
}



