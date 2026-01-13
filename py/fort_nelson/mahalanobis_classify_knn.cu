
/*
Compile:
nvcc -O3 -arch=sm_80 classify_mahalanobis.cu -o classify_mahalanobis \
    -I/usr/include/gdal -lgdal -lcublas -lcusolver

Usage:
./classify_mahalanobis training.bin input.tif      # Classify single image
./classify_mahalanobis training.bin input.bin      # Classify single ENVI image
./classify_mahalanobis training.bin                # Classify all .tif files in current directory

Output is automatically named input_classification.bin

Training file format (binary):
- int32: n_exemplars (number of training rectangles)
- int32: patch_dim (features per patch, e.g., 7*7*3=147)
- For each exemplar:
  - uint8: label (0 or 1)
  - float32[patch_dim]: mean vector
  - float32[patch_dim * patch_dim]: covariance matrix (row-major)
  - float32[patch_dim * patch_dim]: inverse covariance matrix (row-major)
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <gdal.h>
#include <cpl_conv.h>
#include <math.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>

#define PATCH_SIZE 7
#define BLOCK_SIZE 256
#define SKIP_FRACTION_EXEMPLARS 11
#define K_NEAREST_NEIGHBORS 7

typedef struct {
    unsigned char label;
    float* mean;
    float* inv_cov;
} Exemplar;

// CUDA kernel: classify pixels using Mahalanobis distance
__global__ void classify_mahalanobis_kernel(
    const float* __restrict__ padded_img,
    const float* __restrict__ exemplar_means,
    const float* __restrict__ exemplar_inv_covs,
    const unsigned char* __restrict__ exemplar_labels,
    unsigned char* __restrict__ output,
    int h, int w, int c,
    int n_exemplars, int patch_dim,
    int pad, long long start_pixel, long long n_pixels)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pixels) return;
    
    long long pixel_idx = start_pixel + idx;
    int y = pixel_idx / w;
    int x = pixel_idx % w;
    
    int padded_w = w + 2 * pad;
    
    // Extract patch for this pixel
    float patch[PATCH_SIZE * PATCH_SIZE * 3];
    int pi = 0;
    bool has_nan = false;
    
    for (int dy = 0; dy < PATCH_SIZE; dy++) {
        for (int dx = 0; dx < PATCH_SIZE; dx++) {
            for (int b = 0; b < c; b++) {
                int py = y + dy;
                int px = x + dx;
                float val = padded_img[b * (h + 2*pad) * padded_w + py * padded_w + px];
                patch[pi++] = val;
                if (isnan(val)) has_nan = true;
            }
        }
    }
    
    if (has_nan) {
        // Use a special float bit pattern for NAN in output
        // We'll convert this back to NAN when writing float output
        output[pixel_idx] = 254;  // Special marker for NAN
        return;
    }
    
    // Compute patch mean
    float patch_mean = 0.0f;
    for (int i = 0; i < patch_dim; i++) {
        patch_mean += patch[i];
    }
    patch_mean /= patch_dim;
    
    // Center the patch
    float centered[PATCH_SIZE * PATCH_SIZE * 3];
    for (int i = 0; i < patch_dim; i++) {
        centered[i] = patch[i] - patch_mean;
    }
    
    // Find K nearest exemplars using Mahalanobis distance
    // Use a simple fixed-size array as priority queue
    struct Neighbor {
        float dist;
        int idx;
    };
    
    Neighbor knn[K_NEAREST_NEIGHBORS];
    for (int i = 0; i < K_NEAREST_NEIGHBORS; i++) {
        knn[i].dist = INFINITY;
        knn[i].idx = -1;
    }
    
    for (int e = 0; e < n_exemplars; e++) {
        // diff = patch - exemplar_mean
        float diff[PATCH_SIZE * PATCH_SIZE * 3];
        for (int i = 0; i < patch_dim; i++) {
            diff[i] = patch[i] - exemplar_means[e * patch_dim + i];
        }
        
        // temp = inv_cov @ diff
        float temp[PATCH_SIZE * PATCH_SIZE * 3];
        for (int i = 0; i < patch_dim; i++) {
            temp[i] = 0.0f;
            for (int j = 0; j < patch_dim; j++) {
                temp[i] += exemplar_inv_covs[e * patch_dim * patch_dim + i * patch_dim + j] * diff[j];
            }
        }
        
        // dist = diff @ temp (Mahalanobis distance squared)
        float dist = 0.0f;
        for (int i = 0; i < patch_dim; i++) {
            dist += diff[i] * temp[i];
        }
        
        // Insert into K-NN list if better than worst neighbor
        // Find position of max distance (worst neighbor)
        int max_idx = 0;
        float max_dist = knn[0].dist;
        for (int k = 1; k < K_NEAREST_NEIGHBORS; k++) {
            if (knn[k].dist > max_dist) {
                max_dist = knn[k].dist;
                max_idx = k;
            }
        }
        
        // Replace worst neighbor if current is better
        if (dist < max_dist) {
            knn[max_idx].dist = dist;
            knn[max_idx].idx = e;
        }
    }
    
    // Vote: count labels among K nearest neighbors
    int votes[2] = {0, 0};  // votes for class 0 and class 1
    for (int k = 0; k < K_NEAREST_NEIGHBORS; k++) {
        if (knn[k].idx >= 0) {
            int label = exemplar_labels[knn[k].idx];
            votes[label]++;
        }
    }
    
    // Assign class with most votes
    output[pixel_idx] = (votes[1] > votes[0]) ? 1 : 0;
}

void print_progress(long long current, long long total, time_t start_time) {
    float pct = 100.0f * current / total;
    time_t now = time(NULL);
    double elapsed = difftime(now, start_time);
    double rate = current / elapsed;
    double remaining = (total - current) / rate;
    
    int bar_width = 40;
    int filled = (int)(bar_width * current / total);
    
    printf("\r[");
    for (int i = 0; i < bar_width; i++) {
        printf(i < filled ? "█" : "-");
    }
    printf("] %.1f%% [%lld/%lld] ETA: %dm %ds", 
           pct, current, total, (int)(remaining/60), (int)fmod(remaining, 60));
    fflush(stdout);
    
    if (current == total) printf("\n");
}

// Helper function to check if a filename ends with .tif
int ends_with_tif(const char* filename) {
    size_t len = strlen(filename);
    if (len < 4) return 0;
    return (strcmp(filename + len - 4, ".tif") == 0);
}

// Function to classify a single image
int classify_single_image(const char* input_file, 
                          float* all_means, float* all_inv_covs, unsigned char* all_labels,
                          int n_exemplars, int patch_dim) {
    
    printf("\n[IMAGE] Processing: %s\n", input_file);
    
    // Determine if ENVI or TIFF based on extension
    const char* ext = strrchr(input_file, '.');
    bool is_envi = (ext && strcmp(ext, ".bin") == 0);
    
    // Generate output filename: input.tif -> input_classification.bin
    char output_file[1024];
    const char* last_dot = strrchr(input_file, '.');
    if (last_dot != NULL) {
        int base_len = last_dot - input_file;
        snprintf(output_file, sizeof(output_file), "%.*s_classification.bin", base_len, input_file);
    } else {
        snprintf(output_file, sizeof(output_file), "%s_classification.bin", input_file);
    }
    
    // Load training data
    FILE* f = fopen(train_file, "rb");
    if (!f) { perror("Failed to open training file"); return 1; }
    
    int n_exemplars, patch_dim;
    fread(&n_exemplars, sizeof(int), 1, f);
    fread(&patch_dim, sizeof(int), 1, f);
    
    printf("[STATUS] Loading training data: %d exemplars, %d features\n", n_exemplars, patch_dim);
    
    // Read all exemplars and subsample by class
    float* temp_means = (float*)malloc(n_exemplars * patch_dim * sizeof(float));
    float* temp_inv_covs = (float*)malloc(n_exemplars * patch_dim * patch_dim * sizeof(float));
    unsigned char* temp_labels = (unsigned char*)malloc(n_exemplars);
    
    for (int e = 0; e < n_exemplars; e++) {
        fread(&temp_labels[e], sizeof(unsigned char), 1, f);
        fread(&temp_means[e * patch_dim], sizeof(float), patch_dim, f);
        
        // Skip covariance
        float* cov = (float*)malloc(patch_dim * patch_dim * sizeof(float));
        fread(cov, sizeof(float), patch_dim * patch_dim, f);
        free(cov);
        
        fread(&temp_inv_covs[e * patch_dim * patch_dim], sizeof(float), patch_dim * patch_dim, f);
    }
    fclose(f);
    
    // Subsample: keep every SKIP_FRACTION_EXEMPLARS-th exemplar from each class
    int* class_counts = (int*)calloc(2, sizeof(int));
    
    // Count exemplars per class and subsample
    float* all_means = (float*)malloc(n_exemplars * patch_dim * sizeof(float));
    float* all_inv_covs = (float*)malloc(n_exemplars * patch_dim * patch_dim * sizeof(float));
    unsigned char* all_labels = (unsigned char*)malloc(n_exemplars);
    
    int n_kept = 0;
    for (int e = 0; e < n_exemplars; e++) {
        int label = temp_labels[e];
        
        // Keep every SKIP_FRACTION_EXEMPLARS-th exemplar from this class
        if (class_counts[label] % SKIP_FRACTION_EXEMPLARS == 0) {
            all_labels[n_kept] = label;
            memcpy(&all_means[n_kept * patch_dim], &temp_means[e * patch_dim], patch_dim * sizeof(float));
            memcpy(&all_inv_covs[n_kept * patch_dim * patch_dim], &temp_inv_covs[e * patch_dim * patch_dim], 
                   patch_dim * patch_dim * sizeof(float));
            n_kept++;
        }
        class_counts[label]++;
    }
    
    n_exemplars = n_kept;
    printf("[STATUS] Using %d exemplars after subsampling (every %d-th from each class)\n", 
           n_exemplars, SKIP_FRACTION_EXEMPLARS);
    
    free(temp_means);
    free(temp_inv_covs);
    free(temp_labels);
    free(class_counts);
    
    Exemplar* exemplars = (Exemplar*)malloc(n_exemplars * sizeof(Exemplar));
    
    // Load image
    GDALAllRegister();
    GDALDatasetH ds;
    
    if (is_envi) {
        // Open ENVI file
        char envi_path[2048];
        snprintf(envi_path, sizeof(envi_path), "%s", input_file);
        ds = GDALOpen(envi_path, GA_ReadOnly);
    } else {
        ds = GDALOpen(input_file, GA_ReadOnly);
    }
    
    if (!ds) { fprintf(stderr, "Failed to open image\n"); return 1; }
    
    int w = GDALGetRasterXSize(ds);
    int h = GDALGetRasterYSize(ds);
    int c = GDALGetRasterCount(ds);
    
    if (c == 4) c = 3;
    
    printf("[STATUS] Image: %d x %d x %d\n", h, w, c);
    
    int pad = PATCH_SIZE / 2;
    int padded_h = h + 2 * pad;
    int padded_w = w + 2 * pad;
    
    // Load all bands
    printf("[STATUS] Loading and padding image...\n");
    float** band_data = (float**)malloc(c * sizeof(float*));
    for (int b = 0; b < c; b++) {
        GDALRasterBandH band = GDALGetRasterBand(ds, b + 1);
        band_data[b] = (float*)malloc(w * h * sizeof(float));
        GDALRasterIO(band, GF_Read, 0, 0, w, h, band_data[b], w, h, GDT_Float32, 0, 0);
    }
    
    // Allocate padded image
    float* padded_img = (float*)calloc(c * padded_h * padded_w, sizeof(float));
    
    // Copy to padded image, converting all-zero pixels to NAN
    for (int b = 0; b < c; b++) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float val = band_data[b][y * w + x];
                
                bool all_zero = true;
                for (int check_b = 0; check_b < c; check_b++) {
                    if (band_data[check_b][y * w + x] != 0.0f) {
                        all_zero = false;
                        break;
                    }
                }
                if (all_zero) val = NAN;
                
                padded_img[b * padded_h * padded_w + (y + pad) * padded_w + (x + pad)] = val;
            }
        }
    }
    
    // Reflection padding
    for (int b = 0; b < c; b++) {
        for (int y = 0; y < pad; y++) {
            for (int x = 0; x < w; x++) {
                padded_img[b * padded_h * padded_w + y * padded_w + (x + pad)] = 
                    padded_img[b * padded_h * padded_w + (2*pad - y) * padded_w + (x + pad)];
                padded_img[b * padded_h * padded_w + (h + pad + y) * padded_w + (x + pad)] = 
                    padded_img[b * padded_h * padded_w + (h + pad - 2 - y) * padded_w + (x + pad)];
            }
        }
        
        for (int y = 0; y < padded_h; y++) {
            for (int x = 0; x < pad; x++) {
                padded_img[b * padded_h * padded_w + y * padded_w + x] = 
                    padded_img[b * padded_h * padded_w + y * padded_w + (2*pad - x)];
                padded_img[b * padded_h * padded_w + y * padded_w + (w + pad + x)] = 
                    padded_img[b * padded_h * padded_w + y * padded_w + (w + pad - 2 - x)];
            }
        }
    }
    
    for (int b = 0; b < c; b++) free(band_data[b]);
    free(band_data);
    
    // Transfer to GPU
    printf("[STATUS] Transferring to GPU...\n");
    float *d_padded_img, *d_means, *d_inv_covs;
    unsigned char *d_labels, *d_output;
    
    cudaMalloc(&d_padded_img, c * padded_h * padded_w * sizeof(float));
    cudaMalloc(&d_means, n_exemplars * patch_dim * sizeof(float));
    cudaMalloc(&d_inv_covs, n_exemplars * patch_dim * patch_dim * sizeof(float));
    cudaMalloc(&d_labels, n_exemplars * sizeof(unsigned char));
    cudaMalloc(&d_output, (long long)h * w * sizeof(unsigned char));
    
    cudaMemcpy(d_padded_img, padded_img, c * padded_h * padded_w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_means, all_means, n_exemplars * patch_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inv_covs, all_inv_covs, n_exemplars * patch_dim * patch_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, all_labels, n_exemplars * sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    // Classify
    printf("[STATUS] Classifying...\n");
    long long total_pixels = (long long)h * w;
    long long batch_size = 1000000;  // 1M pixels per batch for more frequent updates
    long long n_batches = (total_pixels + batch_size - 1) / batch_size;
    
    time_t start_time = time(NULL);
    
    for (long long batch = 0; batch < n_batches; batch++) {
        long long start_pixel = batch * batch_size;
        long long end_pixel = (batch + 1) * batch_size;
        if (end_pixel > total_pixels) end_pixel = total_pixels;
        long long n_pixels = end_pixel - start_pixel;
        
        int blocks = (n_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        classify_mahalanobis_kernel<<<blocks, BLOCK_SIZE>>>(
            d_padded_img, d_means, d_inv_covs, d_labels, d_output,
            h, w, c, n_exemplars, patch_dim, pad, start_pixel, n_pixels
        );
        
        cudaDeviceSynchronize();
        print_progress(end_pixel, total_pixels, start_time);
    }
    
    // Copy results
    printf("[STATUS] Copying results back...\n");
    unsigned char* output = (unsigned char*)malloc((long long)h * w);
    cudaMemcpy(output, d_output, (long long)h * w * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
    // Save output
    printf("[STATUS] Saving output...\n");
    GDALDriverH drv = GDALGetDriverByName("ENVI");
    GDALDatasetH ods = GDALCreate(drv, output_file, w, h, 1, GDT_Float32, NULL);
    
    double geo[6];
    GDALGetGeoTransform(ds, geo);
    GDALSetGeoTransform(ods, geo);
    const char* proj = GDALGetProjectionRef(ds);
    GDALSetProjection(ods, proj);
    
    float* out_float = (float*)malloc((long long)h * w * sizeof(float));
    for (long long i = 0; i < (long long)h * w; i++) {
        if (output[i] == 254) {
            out_float[i] = NAN;
        } else {
            out_float[i] = (float)output[i];
        }
    }
    
    GDALRasterBandH ob = GDALGetRasterBand(ods, 1);
    GDALRasterIO(ob, GF_Write, 0, 0, w, h, out_float, w, h, GDT_Float32, 0, 0);
    
    GDALClose(ods);
    GDALClose(ds);
    
    printf("[STATUS] Done!\n");
    
    // Cleanup
    free(padded_img);
    free(all_means);
    free(all_inv_covs);
    free(all_labels);
    free(exemplars);
    free(output);
    free(out_float);
    
    cudaFree(d_padded_img);
    cudaFree(d_means);
    cudaFree(d_inv_covs);
    cudaFree(d_labels);
    cudaFree(d_output);
    
    printf("[IMAGE] Done: %s\n", output_file);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s training.bin [input.{tif|bin}]\n", argv[0]);
        fprintf(stderr, "  Single image: %s training.bin input.tif\n", argv[0]);
        fprintf(stderr, "  Single ENVI:  %s training.bin input.bin (assumes input.hdr exists)\n", argv[0]);
        fprintf(stderr, "  Batch mode:   %s training.bin (processes all .tif files in current directory)\n", argv[0]);
        fprintf(stderr, "\nOutput is automatically named input_classification.bin\n");
        return 1;
    }
    
    const char* train_file = argv[1];
    
    // Load training data once
    FILE* f = fopen(train_file, "rb");
    if (!f) { perror("Failed to open training file"); return 1; }
    
    int n_exemplars, patch_dim;
    fread(&n_exemplars, sizeof(int), 1, f);
    fread(&patch_dim, sizeof(int), 1, f);
    
    printf("[STATUS] Loading training data: %d exemplars, %d features\n", n_exemplars, patch_dim);
    
    // Read all exemplars and subsample by class
    float* temp_means = (float*)malloc(n_exemplars * patch_dim * sizeof(float));
    float* temp_inv_covs = (float*)malloc(n_exemplars * patch_dim * patch_dim * sizeof(float));
    unsigned char* temp_labels = (unsigned char*)malloc(n_exemplars);
    
    for (int e = 0; e < n_exemplars; e++) {
        fread(&temp_labels[e], sizeof(unsigned char), 1, f);
        fread(&temp_means[e * patch_dim], sizeof(float), patch_dim, f);
        
        // Skip covariance
        float* cov = (float*)malloc(patch_dim * patch_dim * sizeof(float));
        fread(cov, sizeof(float), patch_dim * patch_dim, f);
        free(cov);
        
        fread(&temp_inv_covs[e * patch_dim * patch_dim], sizeof(float), patch_dim * patch_dim, f);
    }
    fclose(f);
    
    // Subsample: keep every SKIP_FRACTION_EXEMPLARS-th exemplar from each class
    int* class_counts = (int*)calloc(2, sizeof(int));
    
    float* all_means = (float*)malloc(n_exemplars * patch_dim * sizeof(float));
    float* all_inv_covs = (float*)malloc(n_exemplars * patch_dim * patch_dim * sizeof(float));
    unsigned char* all_labels = (unsigned char*)malloc(n_exemplars);
    
    int n_kept = 0;
    for (int e = 0; e < n_exemplars; e++) {
        int label = temp_labels[e];
        
        if (class_counts[label] % SKIP_FRACTION_EXEMPLARS == 0) {
            all_labels[n_kept] = label;
            memcpy(&all_means[n_kept * patch_dim], &temp_means[e * patch_dim], patch_dim * sizeof(float));
            memcpy(&all_inv_covs[n_kept * patch_dim * patch_dim], &temp_inv_covs[e * patch_dim * patch_dim], 
                   patch_dim * patch_dim * sizeof(float));
            n_kept++;
        }
        class_counts[label]++;
    }
    
    n_exemplars = n_kept;
    printf("[STATUS] Using %d exemplars after subsampling (every %d-th from each class)\n", 
           n_exemplars, SKIP_FRACTION_EXEMPLARS);
    
    free(temp_means);
    free(temp_inv_covs);
    free(temp_labels);
    free(class_counts);
    
    // Determine mode: single image or batch
    if (argc >= 3) {
        // Single image mode
        const char* input_file = argv[2];
        int result = classify_single_image(input_file, all_means, all_inv_covs, all_labels, n_exemplars, patch_dim);
        
        free(all_means);
        free(all_inv_covs);
        free(all_labels);
        
        return result;
    } else {
        // Batch mode: process all .tif files in current directory
        printf("\n[BATCH MODE] Processing all .tif files in current directory\n");
        
        DIR* dir = opendir(".");
        if (!dir) {
            perror("Failed to open current directory");
            return 1;
        }
        
        struct dirent* entry;
        int file_count = 0;
        int processed = 0;
        
        // Count .tif files first
        while ((entry = readdir(dir)) != NULL) {
            if (ends_with_tif(entry->d_name)) {
                file_count++;
            }
        }
        rewinddir(dir);
        
        printf("[BATCH MODE] Found %d .tif files\n\n", file_count);
        
        // Process each .tif file
        while ((entry = readdir(dir)) != NULL) {
            if (ends_with_tif(entry->d_name)) {
                processed++;
                printf("[BATCH] File %d/%d\n", processed, file_count);
                classify_single_image(entry->d_name, all_means, all_inv_covs, all_labels, n_exemplars, patch_dim);
            }
        }
        
        closedir(dir);
        
        printf("\n[BATCH MODE] Completed: %d files processed\n", processed);
        
        free(all_means);
        free(all_inv_covs);
        free(all_labels);
        
        return 0;
    }
}

