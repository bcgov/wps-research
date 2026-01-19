/* 20260119 Euclidean Nearest-Neighbor Classifier (Window-based)

Compile:
nvcc -O3 -arch=sm_80 classify_euclidean.cu -o classify_euclidean \
    -I/usr/include/gdal -lgdal

Usage:
./classify_euclidean training.bin input.bin      # Classify single ENVI image
./classify_euclidean training.bin                # Classify all stack*.bin files in current directory

Output is automatically named input_classification.bin

Validation: If annotation_labels.shp exists in current directory, computes
accuracy metrics on training areas after classification.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <gdal.h>
#include <ogr_api.h>
#include <cpl_conv.h>
#include <math.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>
#include <fnmatch.h>

// ============ GLOBAL PARAMETERS ============
#define WINDOW_SIZE 15
#define BLOCK_SIZE 256
#define SKIP_PARAMETER 100
#define K_NEAREST_NEIGHBORS 7
#define MAX_PATCH_DIM 8192  // Maximum supported patch dimension for kernel
#define ANNOTATION_SHP "annotation_labels.shp"
#define MAX_RECTANGLES 100000
// ===========================================

// Structure to hold a training rectangle
typedef struct {
    int x0, y0, x1, y1;
    int label;  // 0 = negative, 1 = positive
} TrainingRect;

// Structure to hold validation metrics
typedef struct {
    long long tp;  // True positives
    long long tn;  // True negatives
    long long fp;  // False positives
    long long fn;  // False negatives
    double prob_sum_positive;  // Sum of probabilities for positive samples
    double prob_sum_negative;  // Sum of probabilities for negative samples
    long long n_positive_samples;
    long long n_negative_samples;
} ValidationMetrics;

// CUDA kernel: classify windows using K-NN Euclidean distance
__global__ void classify_euclidean_kernel(
    const float* __restrict__ img,
    const float* __restrict__ exemplar_patches,
    const unsigned char* __restrict__ exemplar_labels,
    float* __restrict__ output,
    int h, int w, int n_bands,
    int n_exemplars, int patch_dim,
    int n_windows_x, int n_windows_y,
    long long start_window, long long n_windows)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_windows) return;
    
    long long window_idx = start_window + idx;
    int wy = window_idx / n_windows_x;
    int wx = window_idx % n_windows_x;
    
    // Top-left corner of this window in image coordinates
    int win_y = wy * WINDOW_SIZE;
    int win_x = wx * WINDOW_SIZE;
    
    // Extract window for classification
    // Use dynamic allocation workaround - patch stored in local memory
    float patch[MAX_PATCH_DIM];
    int pi = 0;
    bool has_nan = false;
    
    for (int dy = 0; dy < WINDOW_SIZE && !has_nan; dy++) {
        for (int dx = 0; dx < WINDOW_SIZE && !has_nan; dx++) {
            int py = win_y + dy;
            int px = win_x + dx;
            
            // Bounds check
            if (py >= h || px >= w) {
                has_nan = true;
                break;
            }
            
            for (int b = 0; b < n_bands; b++) {
                float val = img[b * h * w + py * w + px];
                if (pi < MAX_PATCH_DIM) {
                    patch[pi++] = val;
                }
                if (isnan(val)) has_nan = true;
            }
        }
    }
    
    // Determine output value for this window
    float result;
    
    if (has_nan) {
        result = NAN;
    } else {
        // Find K nearest exemplars using Euclidean distance
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
            // Compute squared Euclidean distance
            float dist = 0.0f;
            for (int i = 0; i < patch_dim; i++) {
                float diff = patch[i] - exemplar_patches[e * patch_dim + i];
                dist += diff * diff;
            }
            
            // Insert into K-NN list if better than worst neighbor
            int max_idx = 0;
            float max_dist = knn[0].dist;
            for (int k = 1; k < K_NEAREST_NEIGHBORS; k++) {
                if (knn[k].dist > max_dist) {
                    max_dist = knn[k].dist;
                    max_idx = k;
                }
            }
            
            if (dist < max_dist) {
                knn[max_idx].dist = dist;
                knn[max_idx].idx = e;
            }
        }
        
        // Vote: count labels among K nearest neighbors
        int votes[2] = {0, 0};
        int total_votes = 0;
        for (int k = 0; k < K_NEAREST_NEIGHBORS; k++) {
            if (knn[k].idx >= 0) {
                int label = exemplar_labels[knn[k].idx];
                votes[label]++;
                total_votes++;
            }
        }
        
        // Output probability of positive class
        if (total_votes > 0) {
            result = (float)votes[1] / (float)total_votes;
        } else {
            result = 0.0f;
        }
    }
    
    // Write result to all pixels in this window
    for (int dy = 0; dy < WINDOW_SIZE; dy++) {
        for (int dx = 0; dx < WINDOW_SIZE; dx++) {
            int py = win_y + dy;
            int px = win_x + dx;
            if (py < h && px < w) {
                output[py * w + px] = result;
            }
        }
    }
}

void print_status(const char* msg) {
    time_t now = time(NULL);
    struct tm* t = localtime(&now);
    printf("[%02d:%02d:%02d] %s\n", t->tm_hour, t->tm_min, t->tm_sec, msg);
    fflush(stdout);
}

void print_progress(const char* stage, long long current, long long total, time_t start_time) {
    float pct = 100.0f * current / total;
    time_t now = time(NULL);
    double elapsed = difftime(now, start_time);
    
    int bar_width = 30;
    int filled = (int)(bar_width * current / total);
    
    printf("\r  %-12s [", stage);
    for (int i = 0; i < bar_width; i++) {
        printf(i < filled ? "=" : "-");
    }
    
    if (elapsed > 0.5 && current > 0) {
        double rate = current / elapsed;
        double remaining = (total - current) / rate;
        printf("] %5.1f%% | %.1f win/s | ETA %dm %02ds   ", 
               pct, rate, (int)(remaining/60), (int)fmod(remaining, 60));
    } else {
        printf("] %5.1f%%                              ", pct);
    }
    
    fflush(stdout);
    
    if (current == total) printf("\n");
}

int is_stack_bin(const char* filename) {
    // Check if filename matches stack*.bin pattern
    return (fnmatch("stack*.bin", filename, 0) == 0);
}

// Parse COORDS_IMG field to get rectangle bounds
int parse_coords_img(const char* s, int* x0, int* y0, int* x1, int* y1) {
    if (!s || strlen(s) == 0) return 0;
    
    // Check for "NO_RASTER"
    if (strstr(s, "NO_RASTER") != NULL || strstr(s, "no_raster") != NULL) return 0;
    
    // Parse format: "x1,y1;x2,y2;x3,y3;x4,y4"
    int min_x = INT_MAX, min_y = INT_MAX;
    int max_x = INT_MIN, max_y = INT_MIN;
    
    char* copy = strdup(s);
    char* token = strtok(copy, ";");
    int found_any = 0;
    
    while (token != NULL) {
        float fx, fy;
        if (sscanf(token, "%f,%f", &fx, &fy) == 2) {
            int ix = (int)roundf(fx);
            int iy = (int)roundf(fy);
            if (ix < min_x) min_x = ix;
            if (ix > max_x) max_x = ix;
            if (iy < min_y) min_y = iy;
            if (iy > max_y) max_y = iy;
            found_any = 1;
        }
        token = strtok(NULL, ";");
    }
    
    free(copy);
    
    if (!found_any || min_x == INT_MAX) return 0;
    
    *x0 = min_x;
    *y0 = min_y;
    *x1 = max_x;
    *y1 = max_y;
    return 1;
}

// Load training rectangles from shapefile
int load_training_rectangles(const char* shp_path, TrainingRect* rects, int max_rects) {
    OGRRegisterAll();
    
    OGRDataSourceH ds = OGROpen(shp_path, 0, NULL);
    if (ds == NULL) {
        return 0;
    }
    
    OGRLayerH layer = OGR_DS_GetLayer(ds, 0);
    if (layer == NULL) {
        OGR_DS_Destroy(ds);
        return 0;
    }
    
    OGR_L_ResetReading(layer);
    
    int n_rects = 0;
    OGRFeatureH feature;
    
    while ((feature = OGR_L_GetNextFeature(layer)) != NULL && n_rects < max_rects) {
        // Get CLASS field
        int class_idx = OGR_F_GetFieldIndex(feature, "CLASS");
        int coords_idx = OGR_F_GetFieldIndex(feature, "COORDS_IMG");
        
        if (class_idx < 0 || coords_idx < 0) {
            OGR_F_Destroy(feature);
            continue;
        }
        
        const char* class_str = OGR_F_GetFieldAsString(feature, class_idx);
        const char* coords_str = OGR_F_GetFieldAsString(feature, coords_idx);
        
        int x0, y0, x1, y1;
        if (!parse_coords_img(coords_str, &x0, &y0, &x1, &y1)) {
            OGR_F_Destroy(feature);
            continue;
        }
        
        // Determine label
        int label = 0;
        if (class_str) {
            // Check if POSITIVE (case insensitive)
            char upper[64];
            strncpy(upper, class_str, sizeof(upper) - 1);
            upper[sizeof(upper) - 1] = '\0';
            for (int i = 0; upper[i]; i++) {
                if (upper[i] >= 'a' && upper[i] <= 'z') {
                    upper[i] -= 32;
                }
            }
            // Trim whitespace
            char* p = upper;
            while (*p == ' ') p++;
            if (strncmp(p, "POSITIVE", 8) == 0) {
                label = 1;
            }
        }
        
        rects[n_rects].x0 = x0;
        rects[n_rects].y0 = y0;
        rects[n_rects].x1 = x1;
        rects[n_rects].y1 = y1;
        rects[n_rects].label = label;
        n_rects++;
        
        OGR_F_Destroy(feature);
    }
    
    OGR_DS_Destroy(ds);
    return n_rects;
}

// Compute validation metrics on training areas
void compute_validation_metrics(float* output, int w, int h, 
                                 TrainingRect* rects, int n_rects,
                                 ValidationMetrics* metrics) {
    memset(metrics, 0, sizeof(ValidationMetrics));
    
    for (int r = 0; r < n_rects; r++) {
        TrainingRect* rect = &rects[r];
        
        // Clamp to image bounds
        int x0 = rect->x0 < 0 ? 0 : rect->x0;
        int y0 = rect->y0 < 0 ? 0 : rect->y0;
        int x1 = rect->x1 > w ? w : rect->x1;
        int y1 = rect->y1 > h ? h : rect->y1;
        
        // Skip if too small
        if ((x1 - x0) < WINDOW_SIZE || (y1 - y0) < WINDOW_SIZE) {
            continue;
        }
        
        int true_label = rect->label;
        
        // Sample windows within the rectangle (same tiling as training)
        int n_windows_x = (x1 - x0) / WINDOW_SIZE;
        int n_windows_y = (y1 - y0) / WINDOW_SIZE;
        
        for (int wy = 0; wy < n_windows_y; wy++) {
            for (int wx = 0; wx < n_windows_x; wx++) {
                // Get center pixel of window
                int win_y = y0 + wy * WINDOW_SIZE + WINDOW_SIZE / 2;
                int win_x = x0 + wx * WINDOW_SIZE + WINDOW_SIZE / 2;
                
                if (win_y >= h || win_x >= w) continue;
                
                float prob = output[win_y * w + win_x];
                
                if (isnan(prob)) continue;
                
                int predicted = (prob > 0.5f) ? 1 : 0;
                
                if (true_label == 1) {
                    metrics->prob_sum_positive += prob;
                    metrics->n_positive_samples++;
                    if (predicted == 1) {
                        metrics->tp++;
                    } else {
                        metrics->fn++;
                    }
                } else {
                    metrics->prob_sum_negative += prob;
                    metrics->n_negative_samples++;
                    if (predicted == 0) {
                        metrics->tn++;
                    } else {
                        metrics->fp++;
                    }
                }
            }
        }
    }
}

// Print validation report
void print_validation_report(ValidationMetrics* m) {
    printf("\n");
    printf("  ╔════════════════════════════════════════════════════════════════╗\n");
    printf("  ║              VALIDATION ON TRAINING AREAS                      ║\n");
    printf("  ╠════════════════════════════════════════════════════════════════╣\n");
    
    long long total = m->tp + m->tn + m->fp + m->fn;
    
    if (total == 0) {
        printf("  ║  No valid samples found in training areas                      ║\n");
        printf("  ╚════════════════════════════════════════════════════════════════╝\n");
        return;
    }
    
    // Confusion matrix
    printf("  ║  CONFUSION MATRIX:                                             ║\n");
    printf("  ║                        Predicted                               ║\n");
    printf("  ║                    Negative    Positive                        ║\n");
    printf("  ║  Actual Negative   %8lld    %8lld   (TN, FP)              ║\n", m->tn, m->fp);
    printf("  ║  Actual Positive   %8lld    %8lld   (FN, TP)              ║\n", m->fn, m->tp);
    printf("  ╠════════════════════════════════════════════════════════════════╣\n");
    
    // Calculate metrics
    double accuracy = (double)(m->tp + m->tn) / total;
    
    double precision = (m->tp + m->fp > 0) ? (double)m->tp / (m->tp + m->fp) : 0.0;
    double recall = (m->tp + m->fn > 0) ? (double)m->tp / (m->tp + m->fn) : 0.0;
    double f1 = (precision + recall > 0) ? 2.0 * precision * recall / (precision + recall) : 0.0;
    
    double specificity = (m->tn + m->fp > 0) ? (double)m->tn / (m->tn + m->fp) : 0.0;
    double npv = (m->tn + m->fn > 0) ? (double)m->tn / (m->tn + m->fn) : 0.0;
    
    // Balanced accuracy
    double balanced_acc = (recall + specificity) / 2.0;
    
    // Matthews Correlation Coefficient
    double mcc_num = (double)m->tp * m->tn - (double)m->fp * m->fn;
    double mcc_den = sqrt((double)(m->tp + m->fp) * (m->tp + m->fn) * (m->tn + m->fp) * (m->tn + m->fn));
    double mcc = (mcc_den > 0) ? mcc_num / mcc_den : 0.0;
    
    // Average probabilities
    double avg_prob_pos = (m->n_positive_samples > 0) ? m->prob_sum_positive / m->n_positive_samples : 0.0;
    double avg_prob_neg = (m->n_negative_samples > 0) ? m->prob_sum_negative / m->n_negative_samples : 0.0;
    
    printf("  ║  CLASSIFICATION METRICS:                                       ║\n");
    printf("  ║    Accuracy:              %6.2f%%                              ║\n", accuracy * 100);
    printf("  ║    Balanced Accuracy:     %6.2f%%                              ║\n", balanced_acc * 100);
    printf("  ║    Precision (PPV):       %6.2f%%                              ║\n", precision * 100);
    printf("  ║    Recall (Sensitivity):  %6.2f%%                              ║\n", recall * 100);
    printf("  ║    Specificity (TNR):     %6.2f%%                              ║\n", specificity * 100);
    printf("  ║    F1 Score:              %6.4f                               ║\n", f1);
    printf("  ║    NPV:                   %6.2f%%                              ║\n", npv * 100);
    printf("  ║    MCC:                   %+6.4f                               ║\n", mcc);
    printf("  ╠════════════════════════════════════════════════════════════════╣\n");
    printf("  ║  PROBABILITY CALIBRATION:                                      ║\n");
    printf("  ║    Avg prob on POSITIVE:  %6.4f  (ideal: 1.0)                 ║\n", avg_prob_pos);
    printf("  ║    Avg prob on NEGATIVE:  %6.4f  (ideal: 0.0)                 ║\n", avg_prob_neg);
    printf("  ║    Separation:            %6.4f  (ideal: 1.0)                 ║\n", avg_prob_pos - avg_prob_neg);
    printf("  ╠════════════════════════════════════════════════════════════════╣\n");
    printf("  ║  SAMPLE COUNTS:                                                ║\n");
    printf("  ║    Total windows:         %8lld                             ║\n", total);
    printf("  ║    Positive samples:      %8lld                             ║\n", m->n_positive_samples);
    printf("  ║    Negative samples:      %8lld                             ║\n", m->n_negative_samples);
    printf("  ╚════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

int classify_single_image(const char* input_file, 
                          float* all_patches, unsigned char* all_labels,
                          int n_exemplars, int patch_dim, int n_bands_training,
                          int image_num, int total_images,
                          TrainingRect* train_rects, int n_train_rects) {
    
    char status_buf[2048];
    
    printf("\n");
    printf("======================================================================\n");
    if (total_images > 1) {
        printf("  IMAGE %d / %d\n", image_num, total_images);
    }
    printf("  %s\n", input_file);
    printf("======================================================================\n");
    
    char output_file[1024];
    const char* last_dot = strrchr(input_file, '.');
    if (last_dot != NULL) {
        int base_len = last_dot - input_file;
        snprintf(output_file, sizeof(output_file), "%.*s_classification.bin", base_len, input_file);
    } else {
        snprintf(output_file, sizeof(output_file), "%s_classification.bin", input_file);
    }
    
    print_status("Opening image file...");
    GDALAllRegister();
    GDALDatasetH ds = GDALOpen(input_file, GA_ReadOnly);
    
    if (!ds) { 
        fprintf(stderr, "  ERROR: Failed to open image\n"); 
        return 1; 
    }
    
    int w = GDALGetRasterXSize(ds);
    int h = GDALGetRasterYSize(ds);
    int n_bands = GDALGetRasterCount(ds);  // Use ALL bands
    
    // Verify band count matches training data
    if (n_bands != n_bands_training) {
        fprintf(stderr, "  ERROR: Band count mismatch. Image has %d bands, training has %d\n", 
                n_bands, n_bands_training);
        GDALClose(ds);
        return 1;
    }
    
    long long total_pixels = (long long)h * w;
    double megapixels = total_pixels / 1e6;
    
    // Calculate number of windows
    int n_windows_x = (w + WINDOW_SIZE - 1) / WINDOW_SIZE;
    int n_windows_y = (h + WINDOW_SIZE - 1) / WINDOW_SIZE;
    long long total_windows = (long long)n_windows_x * n_windows_y;
    
    snprintf(status_buf, sizeof(status_buf), 
             "Image size: %d x %d x %d (%.2f Mpx)", w, h, n_bands, megapixels);
    print_status(status_buf);
    
    snprintf(status_buf, sizeof(status_buf), 
             "Window grid: %d x %d = %lld windows (%dx%d each)", 
             n_windows_x, n_windows_y, total_windows, WINDOW_SIZE, WINDOW_SIZE);
    print_status(status_buf);
    
    print_status("Loading image bands...");
    time_t load_start = time(NULL);
    
    // Load all bands into contiguous memory (band-interleaved by plane)
    float* img_data = (float*)malloc((size_t)n_bands * h * w * sizeof(float));
    if (!img_data) {
        fprintf(stderr, "  ERROR: Failed to allocate image memory\n");
        GDALClose(ds);
        return 1;
    }
    
    for (int b = 0; b < n_bands; b++) {
        GDALRasterBandH band = GDALGetRasterBand(ds, b + 1);
        CPLErr err = GDALRasterIO(band, GF_Read, 0, 0, w, h, 
                     img_data + (size_t)b * h * w, w, h, GDT_Float32, 0, 0);
        if (err != CE_None) {
            fprintf(stderr, "  ERROR: Failed to read band %d\n", b + 1);
            free(img_data);
            GDALClose(ds);
            return 1;
        }
        print_progress("Loading", b + 1, n_bands, load_start);
    }
    
    // Mark all-zero pixels as NaN
    print_status("Marking nodata pixels...");
    time_t mark_start = time(NULL);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            bool all_zero = true;
            for (int b = 0; b < n_bands; b++) {
                if (img_data[b * h * w + y * w + x] != 0.0f) {
                    all_zero = false;
                    break;
                }
            }
            if (all_zero) {
                for (int b = 0; b < n_bands; b++) {
                    img_data[b * h * w + y * w + x] = NAN;
                }
            }
        }
        if ((y + 1) % 1000 == 0 || y == h - 1) {
            print_progress("Marking", y + 1, h, mark_start);
        }
    }
    
    print_status("Transferring data to GPU...");
    
    float *d_img, *d_patches;
    unsigned char *d_labels;
    float *d_output;
    
    size_t img_size = (size_t)n_bands * h * w * sizeof(float);
    size_t patches_size = (size_t)n_exemplars * patch_dim * sizeof(float);
    size_t labels_size = n_exemplars * sizeof(unsigned char);
    size_t output_size = (size_t)h * w * sizeof(float);
    
    snprintf(status_buf, sizeof(status_buf),
             "GPU memory: img=%.1fMB patches=%.1fMB output=%.1fMB",
             img_size/1e6, patches_size/1e6, output_size/1e6);
    print_status(status_buf);
    
    cudaMalloc(&d_img, img_size);
    cudaMalloc(&d_patches, patches_size);
    cudaMalloc(&d_labels, labels_size);
    cudaMalloc(&d_output, output_size);
    
    cudaMemcpy(d_img, img_data, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_patches, all_patches, patches_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, all_labels, labels_size, cudaMemcpyHostToDevice);
    
    // Initialize output to NaN
    float* nan_buf = (float*)malloc(output_size);
    for (long long i = 0; i < (long long)h * w; i++) nan_buf[i] = NAN;
    cudaMemcpy(d_output, nan_buf, output_size, cudaMemcpyHostToDevice);
    free(nan_buf);
    
    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "  ERROR: CUDA error: %s\n", cudaGetErrorString(cuda_err));
        free(img_data);
        GDALClose(ds);
        return 1;
    }
    
    snprintf(status_buf, sizeof(status_buf),
             "Classifying %lld windows with %d exemplars (K=%d)...",
             total_windows, n_exemplars, K_NEAREST_NEIGHBORS);
    print_status(status_buf);
    
    long long batch_size = 10000;  // Windows per batch
    long long n_batches = (total_windows + batch_size - 1) / batch_size;
    
    time_t classify_start = time(NULL);
    
    for (long long batch = 0; batch < n_batches; batch++) {
        long long start_window = batch * batch_size;
        long long end_window = (batch + 1) * batch_size;
        if (end_window > total_windows) end_window = total_windows;
        long long n_windows_batch = end_window - start_window;
        
        int blocks = (n_windows_batch + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        classify_euclidean_kernel<<<blocks, BLOCK_SIZE>>>(
            d_img, d_patches, d_labels, d_output,
            h, w, n_bands, n_exemplars, patch_dim,
            n_windows_x, n_windows_y, start_window, n_windows_batch
        );
        
        cudaDeviceSynchronize();
        print_progress("Classifying", end_window, total_windows, classify_start);
    }
    
    print_status("Copying results from GPU...");
    float* output = (float*)malloc(output_size);
    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);
    
    long long n_positive = 0, n_negative = 0, n_nan = 0;
    double prob_sum = 0.0;
    for (long long i = 0; i < total_pixels; i++) {
        if (isnan(output[i])) n_nan++;
        else {
            prob_sum += output[i];
            if (output[i] > 0.5f) n_positive++;
            else n_negative++;
        }
    }
    double avg_prob = (n_positive + n_negative > 0) ? prob_sum / (n_positive + n_negative) : 0.0;
    
    snprintf(status_buf, sizeof(status_buf),
             "Results: %lld positive (>0.5), %lld negative, %lld nodata, avg_prob=%.3f",
             n_positive, n_negative, n_nan, avg_prob);
    print_status(status_buf);
    
    // Compute validation metrics if training rectangles are available
    if (n_train_rects > 0) {
        print_status("Computing validation metrics on training areas...");
        ValidationMetrics metrics;
        compute_validation_metrics(output, w, h, train_rects, n_train_rects, &metrics);
        print_validation_report(&metrics);
    }
    
    print_status("Saving output file...");
    GDALDriverH drv = GDALGetDriverByName("ENVI");
    GDALDatasetH ods = GDALCreate(drv, output_file, w, h, 1, GDT_Float32, NULL);
    
    double geo[6];
    GDALGetGeoTransform(ds, geo);
    GDALSetGeoTransform(ods, geo);
    const char* proj = GDALGetProjectionRef(ds);
    GDALSetProjection(ods, proj);
    
    GDALRasterBandH ob = GDALGetRasterBand(ods, 1);
    CPLErr write_err = GDALRasterIO(ob, GF_Write, 0, 0, w, h, output, w, h, GDT_Float32, 0, 0);
    if (write_err != CE_None) {
        fprintf(stderr, "  WARNING: Error writing output raster\n");
    }
    
    GDALClose(ods);
    GDALClose(ds);
    
    free(img_data);
    free(output);
    
    cudaFree(d_img);
    cudaFree(d_patches);
    cudaFree(d_labels);
    cudaFree(d_output);
    
    time_t end_time = time(NULL);
    double total_time = difftime(end_time, load_start);
    
    snprintf(status_buf, sizeof(status_buf),
             "Complete: %.1f sec, %.2f Mpx/s",
             total_time, megapixels/total_time);
    print_status(status_buf);
    
    return 0;
}

int main(int argc, char** argv) {
    printf("\n");
    printf("========================================================================\n");
    printf("  EUCLIDEAN K-NN CLASSIFIER (Window-based)\n");
    printf("========================================================================\n");
    
    if (argc < 2) {
        fprintf(stderr, "\nUsage: %s training.bin [input.bin]\n", argv[0]);
        fprintf(stderr, "  Single ENVI:  %s training.bin input.bin\n", argv[0]);
        fprintf(stderr, "  Batch mode:   %s training.bin\n", argv[0]);
        fprintf(stderr, "\nBatch mode processes all stack*.bin files in current directory\n");
        fprintf(stderr, "\nIf %s exists, validation metrics are computed.\n", ANNOTATION_SHP);
        return 1;
    }
    
    const char* train_file = argv[1];
    char status_buf[512];
    
    print_status("Loading training data...");
    printf("  File: %s\n", train_file);
    printf("  WINDOW_SIZE = %d\n", WINDOW_SIZE);
    printf("  SKIP_PARAMETER = %d\n", SKIP_PARAMETER);
    
    FILE* f = fopen(train_file, "rb");
    if (!f) { perror("  ERROR: Failed to open training file"); return 1; }
    
    int n_exemplars, patch_dim, n_bands;
    if (fread(&n_exemplars, sizeof(int), 1, f) != 1 ||
        fread(&patch_dim, sizeof(int), 1, f) != 1 ||
        fread(&n_bands, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "  ERROR: Failed to read training file header\n");
        fclose(f);
        return 1;
    }
    
    snprintf(status_buf, sizeof(status_buf),
             "Training file: %d exemplars, %d features per patch, %d bands", 
             n_exemplars, patch_dim, n_bands);
    print_status(status_buf);
    
    // Verify patch_dim matches expected
    int expected_patch_dim = WINDOW_SIZE * WINDOW_SIZE * n_bands;
    if (patch_dim != expected_patch_dim) {
        fprintf(stderr, "  WARNING: patch_dim=%d but expected %d for %dx%d windows with %d bands\n",
                patch_dim, expected_patch_dim, WINDOW_SIZE, WINDOW_SIZE, n_bands);
    }
    
    if (patch_dim > MAX_PATCH_DIM) {
        fprintf(stderr, "  ERROR: patch_dim=%d exceeds MAX_PATCH_DIM=%d\n", patch_dim, MAX_PATCH_DIM);
        fclose(f);
        return 1;
    }
    
    // Read exemplars (skipping based on SKIP_PARAMETER)
    print_status("Reading exemplars...");
    time_t read_start = time(NULL);
    
    int n_to_read = (n_exemplars + SKIP_PARAMETER - 1) / SKIP_PARAMETER;
    
    float* temp_patches = (float*)malloc((size_t)n_to_read * patch_dim * sizeof(float));
    unsigned char* temp_labels = (unsigned char*)malloc(n_to_read);
    
    float* skip_buf = (float*)malloc(patch_dim * sizeof(float));
    
    int n_read = 0;
    for (int e = 0; e < n_exemplars; e++) {
        if (e % SKIP_PARAMETER == 0) {
            if (fread(&temp_labels[n_read], sizeof(unsigned char), 1, f) != 1 ||
                fread(&temp_patches[(size_t)n_read * patch_dim], sizeof(float), patch_dim, f) != (size_t)patch_dim) {
                fprintf(stderr, "  ERROR: Failed to read exemplar %d\n", e);
                break;
            }
            n_read++;
        } else {
            unsigned char dummy_label;
            if (fread(&dummy_label, sizeof(unsigned char), 1, f) != 1 ||
                fread(skip_buf, sizeof(float), patch_dim, f) != (size_t)patch_dim) {
                fprintf(stderr, "  ERROR: Failed to skip exemplar %d\n", e);
                break;
            }
        }
        
        if ((e + 1) % 10000 == 0 || e == n_exemplars - 1) {
            print_progress("Reading", e + 1, n_exemplars, read_start);
        }
    }
    fclose(f);
    free(skip_buf);
    
    snprintf(status_buf, sizeof(status_buf),
             "Read %d exemplars (1/%d of %d in file)", n_read, SKIP_PARAMETER, n_exemplars);
    print_status(status_buf);
    
    n_exemplars = n_read;
    
    // Count classes
    int n_pos = 0, n_neg = 0;
    for (int e = 0; e < n_exemplars; e++) {
        if (temp_labels[e] == 1) n_pos++;
        else n_neg++;
    }
    
    snprintf(status_buf, sizeof(status_buf),
             "Class distribution: %d positive, %d negative", n_pos, n_neg);
    print_status(status_buf);
    
    // Try to load training rectangles for validation
    TrainingRect* train_rects = (TrainingRect*)malloc(MAX_RECTANGLES * sizeof(TrainingRect));
    int n_train_rects = 0;
    
    struct stat st;
    if (stat(ANNOTATION_SHP, &st) == 0) {
        print_status("Loading annotation shapefile for validation...");
        n_train_rects = load_training_rectangles(ANNOTATION_SHP, train_rects, MAX_RECTANGLES);
        
        if (n_train_rects > 0) {
            int n_pos_rects = 0, n_neg_rects = 0;
            for (int i = 0; i < n_train_rects; i++) {
                if (train_rects[i].label == 1) n_pos_rects++;
                else n_neg_rects++;
            }
            snprintf(status_buf, sizeof(status_buf),
                     "Loaded %d training rectangles (%d positive, %d negative)",
                     n_train_rects, n_pos_rects, n_neg_rects);
            print_status(status_buf);
        } else {
            print_status("WARNING: Could not load training rectangles from shapefile");
        }
    } else {
        snprintf(status_buf, sizeof(status_buf),
                 "No %s found - skipping validation metrics", ANNOTATION_SHP);
        print_status(status_buf);
    }
    
    // Single or batch mode
    if (argc >= 3) {
        int result = classify_single_image(argv[2], temp_patches, temp_labels, 
                                           n_exemplars, patch_dim, n_bands, 1, 1,
                                           train_rects, n_train_rects);
        free(temp_patches);
        free(temp_labels);
        free(train_rects);
        
        printf("\n========================================================================\n");
        printf("  DONE\n");
        printf("========================================================================\n\n");
        
        return result;
    } else {
        print_status("BATCH MODE: Scanning for stack*.bin files...");
        
        DIR* dir = opendir(".");
        if (!dir) { perror("  ERROR: Failed to open directory"); return 1; }
        
        struct dirent* entry;
        int file_count = 0, processed = 0;
        
        while ((entry = readdir(dir)) != NULL) {
            if (is_stack_bin(entry->d_name)) file_count++;
        }
        rewinddir(dir);
        
        snprintf(status_buf, sizeof(status_buf), "Found %d stack*.bin files to process", file_count);
        print_status(status_buf);
        
        if (file_count == 0) {
            print_status("No stack*.bin files found in current directory");
            closedir(dir);
            free(temp_patches);
            free(temp_labels);
            free(train_rects);
            return 0;
        }
        
        time_t batch_start = time(NULL);
        
        while ((entry = readdir(dir)) != NULL) {
            if (is_stack_bin(entry->d_name)) {
                processed++;
                classify_single_image(entry->d_name, temp_patches, temp_labels, 
                                      n_exemplars, patch_dim, n_bands, processed, file_count,
                                      train_rects, n_train_rects);
            }
        }
        
        closedir(dir);
        
        time_t batch_end = time(NULL);
        double total_batch_time = difftime(batch_end, batch_start);
        
        printf("\n========================================================================\n");
        printf("  BATCH COMPLETE\n");
        printf("========================================================================\n");
        snprintf(status_buf, sizeof(status_buf),
                 "Processed %d files in %.1f seconds (%.1f sec/file)",
                 processed, total_batch_time, processed > 0 ? total_batch_time/processed : 0);
        print_status(status_buf);
        printf("\n");
        
        free(temp_patches);
        free(temp_labels);
        free(train_rects);
        return 0;
    }
}


