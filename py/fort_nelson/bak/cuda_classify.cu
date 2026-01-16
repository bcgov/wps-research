

/*
Compile:
nvcc -O3 -arch=sm_80 classify_cuda.cu -o classify_cuda \
    -I/usr/include/gdal -lgdal -lcublas

Usage:
./classify_cuda training.bin input.tif output.bin

Training file format (binary):
- int32: n_train (number of training samples)
- int32: patch_dim (features per patch)
- int32: n_classes (number of classes, should be 2)
- float32[n_train * patch_dim]: training vectors
- uint8[n_train]: training labels
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <gdal.h>
#include <cpl_conv.h>
#include <math.h>
#include <time.h>

#define PATCH_SIZE 7
#define BLOCK_SIZE 256

// CUDA kernel: classify pixels using nearest neighbor
__global__ void classify_kernel(
    const float* __restrict__ padded_img,  // padded image on GPU
    const float* __restrict__ train_vecs,   // training vectors
    const unsigned char* __restrict__ train_labels, // training labels
    unsigned char* __restrict__ output,     // output classification
    int h, int w, int c,                    // image dimensions
    int n_train, int patch_dim,             // training data info
    int pad, long long start_pixel, long long n_pixels)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pixels) return;

    long long pixel_idx = start_pixel + idx;
    int y = pixel_idx / w;
    int x = pixel_idx % w;

    int padded_w = w + 2 * pad;

    // Extract patch for this pixel
    float patch[PATCH_SIZE * PATCH_SIZE * 3];  // max 147 features
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

    // If patch contains NAN, set output to 255 (NAN marker) and skip
    if (has_nan) {
        output[pixel_idx] = 255;
        return;
    }

    // Find nearest training vector
    float min_dist = INFINITY;
    int best_idx = 0;

    for (int t = 0; t < n_train; t++) {
        float dist = 0.0f;
        for (int f = 0; f < patch_dim; f++) {
            float diff = patch[f] - train_vecs[t * patch_dim + f];
            dist += diff * diff;
        }

        if (dist < min_dist) {
            min_dist = dist;
            best_idx = t;
        }
    }

    output[pixel_idx] = train_labels[best_idx];
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

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s training.bin input.tif output.bin\n", argv[0]);
        return 1;
    }

    const char* train_file = argv[1];
    const char* input_file = argv[2];
    const char* output_file = argv[3];

    // Load training data
    FILE* f = fopen(train_file, "rb");
    if (!f) { perror("Failed to open training file"); return 1; }

    int n_train, patch_dim, n_classes;
    fread(&n_train, sizeof(int), 1, f);
    fread(&patch_dim, sizeof(int), 1, f);
    fread(&n_classes, sizeof(int), 1, f);

    printf("[STATUS] Loading training data: %d samples, %d features\n", n_train, patch_dim);

    float* train_vecs = (float*)malloc(n_train * patch_dim * sizeof(float));
    unsigned char* train_labels = (unsigned char*)malloc(n_train);

    fread(train_vecs, sizeof(float), n_train * patch_dim, f);
    fread(train_labels, sizeof(unsigned char), n_train, f);
    fclose(f);

    // Load image
    GDALAllRegister();
    GDALDatasetH ds = GDALOpen(input_file, GA_ReadOnly);
    if (!ds) { fprintf(stderr, "Failed to open image\n"); return 1; }

    int w = GDALGetRasterXSize(ds);
    int h = GDALGetRasterYSize(ds);
    int c = GDALGetRasterCount(ds);

    // Drop 4th band if present
    if (c == 4) c = 3;

    printf("[STATUS] Image: %d x %d x %d\n", h, w, c);

    int pad = PATCH_SIZE / 2;
    int padded_h = h + 2 * pad;
    int padded_w = w + 2 * pad;

    // Allocate padded image
    float* padded_img = (float*)calloc(c * padded_h * padded_w, sizeof(float));

    // Load and pad image
    printf("[STATUS] Loading and padding image...\n");
    for (int b = 0; b < c; b++) {
        GDALRasterBandH band = GDALGetRasterBand(ds, b + 1);
        float* temp = (float*)malloc(w * h * sizeof(float));
        GDALRasterIO(band, GF_Read, 0, 0, w, h, temp, w, h, GDT_Float32, 0, 0);

        // Copy with reflection padding
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                padded_img[b * padded_h * padded_w + (y + pad) * padded_w + (x + pad)] = temp[y * w + x];
            }
        }

        // Pad top and bottom
        for (int y = 0; y < pad; y++) {
            for (int x = 0; x < w; x++) {
                padded_img[b * padded_h * padded_w + y * padded_w + (x + pad)] =
                    padded_img[b * padded_h * padded_w + (2*pad - y) * padded_w + (x + pad)];
                padded_img[b * padded_h * padded_w + (h + pad + y) * padded_w + (x + pad)] =
                    padded_img[b * padded_h * padded_w + (h + pad - 2 - y) * padded_w + (x + pad)];
            }
        }

        // Pad left and right
        for (int y = 0; y < padded_h; y++) {
            for (int x = 0; x < pad; x++) {
                padded_img[b * padded_h * padded_w + y * padded_w + x] =
                    padded_img[b * padded_h * padded_w + y * padded_w + (2*pad - x)];
                padded_img[b * padded_h * padded_w + y * padded_w + (w + pad + x)] =
                    padded_img[b * padded_h * padded_w + y * padded_w + (w + pad - 2 - x)];
            }
        }

        free(temp);
    }

    // Allocate GPU memory
    printf("[STATUS] Transferring to GPU...\n");
    float *d_padded_img, *d_train_vecs;
    unsigned char *d_train_labels, *d_output;

    cudaMalloc(&d_padded_img, c * padded_h * padded_w * sizeof(float));
    cudaMalloc(&d_train_vecs, n_train * patch_dim * sizeof(float));
    cudaMalloc(&d_train_labels, n_train * sizeof(unsigned char));
    cudaMalloc(&d_output, (long long)h * w * sizeof(unsigned char));

    cudaMemcpy(d_padded_img, padded_img, c * padded_h * padded_w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_train_vecs, train_vecs, n_train * patch_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_train_labels, train_labels, n_train * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Process in batches to show progress
    printf("[STATUS] Classifying...\n");
    long long total_pixels = (long long)h * w;
    long long batch_size = 10000000; // 10M pixels per batch
    long long n_batches = (total_pixels + batch_size - 1) / batch_size;

    time_t start_time = time(NULL);

    for (long long batch = 0; batch < n_batches; batch++) {
        long long start_pixel = batch * batch_size;
        long long end_pixel = (batch + 1) * batch_size;
        if (end_pixel > total_pixels) end_pixel = total_pixels;
        long long n_pixels = end_pixel - start_pixel;

        int blocks = (n_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;

        classify_kernel<<<blocks, BLOCK_SIZE>>>(
            d_padded_img, d_train_vecs, d_train_labels, d_output,
            h, w, c, n_train, patch_dim, pad, start_pixel, n_pixels
        );

        cudaDeviceSynchronize();
        print_progress(end_pixel, total_pixels, start_time);
    }

    // Copy result back
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
        // Convert 255 (NAN marker) back to NAN in output
        if (output[i] == 255) {
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
    free(train_vecs);
    free(train_labels);
    free(output);
    free(out_float);

    cudaFree(d_padded_img);
    cudaFree(d_train_vecs);
    cudaFree(d_train_labels);
    cudaFree(d_output);

    return 0;
}


