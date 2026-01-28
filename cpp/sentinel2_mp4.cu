/**
 * CUDA-accelerated ENVI .bin to MP4 video generator with wipe transitions
 * Optimized for NVIDIA L40S GPU (48GB VRAM, sm_89)
 * 
 * Features:
 * - Multi-threaded file loading (32 workers)
 * - GPU memory prefetching (8 images ahead)
 * - Continuous wipe transitions between frames
 * - Hardware-accelerated NVENC encoding
 * - ENVI format 32-bit float input with histogram trimming
 * - Averaged min/max bounds for consistent visualization
 * 
 * Compile with:
 *   nvcc -O3 -arch=sm_89 bin_to_video.cu -o bin_to_video \
 *        -lpthread -lavformat -lavcodec -lavutil -lswscale
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/stat.h>
#include <vector>
#include <string>
#include <algorithm>
#include <queue>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <map>
#include <set>
#include <cmath>
#include <thread>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

// ============================================================================
// Configuration constants optimized for L40S
// ============================================================================
#define NUM_LOADER_THREADS 32
#define GPU_PREFETCH_BUFFER 8
#define OUTPUT_WIDTH 1080
#define OUTPUT_HEIGHT 1080
#define OUTPUT_FPS 60
#define TRANSITION_DURATION_SEC 1.0f  // 1 second per transition (doubled)
#define FRAMES_PER_TRANSITION ((int)(OUTPUT_FPS * TRANSITION_DURATION_SEC))
#define HISTOGRAM_TRIM_PERCENT 2.0f
#define MAX_BANDS_USED 3
#define HISTOGRAM_BINS 65536  // 16-bit precision for histogram binning
#define VIDEO_BITRATE 24000000  // 24 Mbps (3x original 8 Mbps)
#define FONT_CHAR_WIDTH 8
#define FONT_CHAR_HEIGHT 16
#define FONT_SCALE 2
#define TEXT_MARGIN 20

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ============================================================================
// Data Structures
// ============================================================================

// ENVI header information
struct ENVIHeader {
    int samples;      // width
    int lines;        // height
    int bands;
    int data_type;    // should be 4 for 32-bit float
    std::string interleave; // BSQ, BIL, or BIP
    int header_offset;
    std::string byte_order;
};

// Image file information
struct ImageFile {
    std::string filepath;
    std::string filename;
    std::string timestamp;  // 3rd field from filename
    int index;              // sorted index
};

// Per-band statistics for histogram trimming
struct BandStats {
    float min_val;
    float max_val;
};

// Image data ready for GPU
struct ImageData {
    int index;
    float* host_data;       // Interleaved RGB float data
    size_t data_size;
    int width;
    int height;
    bool valid;
};

// GPU ring buffer entry
struct GPUBufferEntry {
    float* d_float_data;    // GPU float RGB data
    uint8_t* d_rgb_data;    // GPU 8-bit RGB data (scaled)
    int image_index;
    bool ready;
    bool processed;
};

// ============================================================================
// Global State
// ============================================================================

// Thread synchronization
std::mutex g_queue_mutex;
std::condition_variable g_producer_cv;
std::condition_variable g_consumer_cv;
std::atomic<bool> g_loading_complete(false);
std::atomic<int> g_next_load_index(0);

// Image queue for GPU processing
std::queue<ImageData> g_image_queue;
const size_t MAX_QUEUE_SIZE = GPU_PREFETCH_BUFFER;

// Global image parameters (determined from first image)
int g_image_width = 0;
int g_image_height = 0;
int g_image_bands = 0;

// Global scaling bounds (averaged from all images)
BandStats g_global_bounds[MAX_BANDS_USED];

// File list
std::vector<ImageFile> g_image_files;

// Globals

std::map<int, ImageData> g_reorder_buffer;
std::atomic<int> g_next_process_index(0);



// ============================================================================
// ENVI Header Parser
// ============================================================================

bool parseENVIHeader(const std::string& hdr_path, ENVIHeader& header) {
    FILE* f = fopen(hdr_path.c_str(), "r");
    if (!f) return false;
    
    // Set defaults
    header.samples = 0;
    header.lines = 0;
    header.bands = 1;
    header.data_type = 4;
    header.interleave = "bsq";
    header.header_offset = 0;
    header.byte_order = "0";
    
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        char* eq = strchr(line, '=');
        if (!eq) continue;
        
        *eq = '\0';
        char* key = line;
        char* value = eq + 1;
        
        // Trim whitespace
        while (*key == ' ' || *key == '\t') key++;
        char* end = key + strlen(key) - 1;
        while (end > key && (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r')) *end-- = '\0';
        
        while (*value == ' ' || *value == '\t') value++;
        end = value + strlen(value) - 1;
        while (end > value && (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r')) *end-- = '\0';
        
        if (strcasecmp(key, "samples") == 0) header.samples = atoi(value);
        else if (strcasecmp(key, "lines") == 0) header.lines = atoi(value);
        else if (strcasecmp(key, "bands") == 0) header.bands = atoi(value);
        else if (strcasecmp(key, "data type") == 0) header.data_type = atoi(value);
        else if (strcasecmp(key, "interleave") == 0) header.interleave = value;
        else if (strcasecmp(key, "header offset") == 0) header.header_offset = atoi(value);
        else if (strcasecmp(key, "byte order") == 0) header.byte_order = value;
    }
    
    fclose(f);
    return (header.samples > 0 && header.lines > 0);
}

// ============================================================================
// Filename parsing and sorting
// ============================================================================

std::string extractTimestamp(const std::string& filename) {
    // Filename format: S2C_MSIL2A_20251014T192401_N0511_R099_T09UYU_...
    // Fields separated by underscore, timestamp is 3rd field (index 2)
    std::vector<std::string> fields;
    size_t start = 0;
    size_t end = filename.find('_');
    
    while (end != std::string::npos) {
        fields.push_back(filename.substr(start, end - start));
        start = end + 1;
        end = filename.find('_', start);
    }
    fields.push_back(filename.substr(start));
    
    if (fields.size() >= 3) {
        return fields[2];  // e.g., "20251014T192401"
    }
    return "";
}

std::vector<ImageFile> findAndSortBinFiles(const std::string& directory) {
    std::vector<ImageFile> files;
    std::set<std::string> seen_timestamps;
    
    DIR* dir = opendir(directory.c_str());
    if (!dir) {
        fprintf(stderr, "Error: Cannot open directory %s\n", directory.c_str());
        return files;
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        std::string name = entry->d_name;
        if (name.length() > 4 && name.substr(name.length() - 4) == ".bin") {
            std::string timestamp = extractTimestamp(name);
            
            if (timestamp.empty()) {
                fprintf(stderr, "Warning: Could not extract timestamp from %s, skipping\n", name.c_str());
                continue;
            }
            
            if (seen_timestamps.count(timestamp)) {
                fprintf(stderr, "Warning: Duplicate timestamp %s in %s, skipping\n", 
                        timestamp.c_str(), name.c_str());
                continue;
            }
            seen_timestamps.insert(timestamp);
            
            ImageFile img;
            img.filepath = directory + "/" + name;
            img.filename = name;
            img.timestamp = timestamp;
            files.push_back(img);
        }
    }
    closedir(dir);
    
    // Sort by timestamp
    std::sort(files.begin(), files.end(), 
              [](const ImageFile& a, const ImageFile& b) {
                  return a.timestamp < b.timestamp;
              });
    
    // Assign sorted indices
    for (size_t i = 0; i < files.size(); i++) {
        files[i].index = (int)i;
    }
    
    return files;
}

// ============================================================================
// ENVI File Reading
// ============================================================================

float* readENVIFile(const ImageFile& img, ENVIHeader& header) {
    // Try to find header file
    std::string hdr_path = img.filepath;
    size_t dot_pos = hdr_path.rfind('.');
    if (dot_pos != std::string::npos) {
        hdr_path = hdr_path.substr(0, dot_pos) + ".hdr";
    } else {
        hdr_path += ".hdr";
    }
    
    // Also try .bin.hdr
    if (!parseENVIHeader(hdr_path, header)) {
        hdr_path = img.filepath + ".hdr";
        if (!parseENVIHeader(hdr_path, header)) {
            fprintf(stderr, "Warning: No header found for %s, trying to detect from file size\n", 
                    img.filename.c_str());
            return nullptr;
        }
    }
    
    if (header.data_type != 4) {
        fprintf(stderr, "Warning: Expected 32-bit float (type 4), got type %d for %s\n",
                header.data_type, img.filename.c_str());
    }
    
    int bands_to_use = std::min(header.bands, MAX_BANDS_USED);
    size_t pixels = (size_t)header.samples * header.lines;
    size_t band_size = pixels * sizeof(float);
    
    FILE* f = fopen(img.filepath.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open %s\n", img.filepath.c_str());
        return nullptr;
    }
    
    fseek(f, header.header_offset, SEEK_SET);
    
    // Allocate output as interleaved RGB
    float* output = (float*)malloc(pixels * bands_to_use * sizeof(float));
    if (!output) {
        fclose(f);
        return nullptr;
    }
    
    // Read based on interleave format
    std::string interleave = header.interleave;
    std::transform(interleave.begin(), interleave.end(), interleave.begin(), ::tolower);
    
    if (interleave == "bsq") {
        // Band Sequential: all of band 1, then all of band 2, etc.
        float* temp_band = (float*)malloc(band_size);
        for (int b = 0; b < bands_to_use; b++) {
            fseek(f, header.header_offset + b * band_size, SEEK_SET);
            if (fread(temp_band, sizeof(float), pixels, f) != pixels) {
                fprintf(stderr, "Error reading band %d from %s\n", b, img.filename.c_str());
                free(temp_band);
                free(output);
                fclose(f);
                return nullptr;
            }
            // Interleave into output
            for (size_t p = 0; p < pixels; p++) {
                output[p * bands_to_use + b] = temp_band[p];
            }
        }
        free(temp_band);
    } else if (interleave == "bip") {
        // Band Interleaved by Pixel: RGB RGB RGB ...
        float* temp = (float*)malloc(pixels * header.bands * sizeof(float));
        if (fread(temp, sizeof(float), pixels * header.bands, f) != pixels * header.bands) {
            fprintf(stderr, "Error reading %s\n", img.filename.c_str());
            free(temp);
            free(output);
            fclose(f);
            return nullptr;
        }
        for (size_t p = 0; p < pixels; p++) {
            for (int b = 0; b < bands_to_use; b++) {
                output[p * bands_to_use + b] = temp[p * header.bands + b];
            }
        }
        free(temp);
    } else if (interleave == "bil") {
        // Band Interleaved by Line
        float* temp_line = (float*)malloc(header.samples * header.bands * sizeof(float));
        for (int line = 0; line < header.lines; line++) {
            if (fread(temp_line, sizeof(float), header.samples * header.bands, f) 
                != (size_t)header.samples * header.bands) {
                fprintf(stderr, "Error reading line %d from %s\n", line, img.filename.c_str());
                free(temp_line);
                free(output);
                fclose(f);
                return nullptr;
            }
            for (int s = 0; s < header.samples; s++) {
                for (int b = 0; b < bands_to_use; b++) {
                    size_t out_idx = ((size_t)line * header.samples + s) * bands_to_use + b;
                    output[out_idx] = temp_line[b * header.samples + s];
                }
            }
        }
        free(temp_line);
    } else {
        fprintf(stderr, "Unknown interleave format: %s\n", interleave.c_str());
        free(output);
        fclose(f);
        return nullptr;
    }
    
    fclose(f);
    return output;
}

// ============================================================================
// GPU-Accelerated Histogram Computation (Two-Pass Binning)
// Pass 1: Find min/max range per band
// Pass 2: Build histogram and find percentile bounds
// ============================================================================

// Kernel: Find min/max values per band (reduction)
__global__ void findMinMaxKernel(const float* data, int num_pixels, int bands,
                                  float* block_mins, float* block_maxs) {
    extern __shared__ float shared[];
    float* s_min = shared;
    float* s_max = shared + blockDim.x * bands;
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    for (int b = 0; b < bands; b++) {
        s_min[tid * bands + b] = INFINITY;
        s_max[tid * bands + b] = -INFINITY;
    }
    
    // Each thread processes multiple pixels (grid-stride loop)
    for (int i = gid; i < num_pixels; i += blockDim.x * gridDim.x) {
        for (int b = 0; b < bands; b++) {
            float val = data[i * bands + b];
            if (isfinite(val)) {
                s_min[tid * bands + b] = fminf(s_min[tid * bands + b], val);
                s_max[tid * bands + b] = fmaxf(s_max[tid * bands + b], val);
            }
        }
    }
    __syncthreads();
    
    // Reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int b = 0; b < bands; b++) {
                s_min[tid * bands + b] = fminf(s_min[tid * bands + b], s_min[(tid + stride) * bands + b]);
                s_max[tid * bands + b] = fmaxf(s_max[tid * bands + b], s_max[(tid + stride) * bands + b]);
            }
        }
        __syncthreads();
    }
    
    // Write block result
    if (tid == 0) {
        for (int b = 0; b < bands; b++) {
            block_mins[blockIdx.x * bands + b] = s_min[b];
            block_maxs[blockIdx.x * bands + b] = s_max[b];
        }
    }
}

// Kernel: Build histogram for each band
__global__ void buildHistogramKernel(const float* data, int num_pixels, int bands,
                                      const float* mins, const float* maxs,
                                      unsigned int* histograms) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid >= num_pixels) return;
    
    for (int b = 0; b < bands; b++) {
        float val = data[gid * bands + b];
        if (isfinite(val)) {
            float range = maxs[b] - mins[b];
            if (range > 0) {
                float normalized = (val - mins[b]) / range;
                normalized = fminf(fmaxf(normalized, 0.0f), 1.0f);
                int bin = (int)(normalized * (HISTOGRAM_BINS - 1));
                bin = min(max(bin, 0), HISTOGRAM_BINS - 1);
                atomicAdd(&histograms[b * HISTOGRAM_BINS + bin], 1);
            }
        }
    }
}

// Host function: GPU-accelerated histogram bounds computation
void computeHistogramBoundsGPU(const float* d_data, int width, int height, int bands,
                                BandStats* stats, float trim_percent, cudaStream_t stream) {
    int num_pixels = width * height;
    
    // Pass 1: Find global min/max per band
    int block_size = 256;
    int num_blocks = min((num_pixels + block_size - 1) / block_size, 1024);
    size_t shared_mem = 2 * block_size * bands * sizeof(float);
    
    float *d_block_mins, *d_block_maxs;
    CUDA_CHECK(cudaMalloc(&d_block_mins, num_blocks * bands * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_block_maxs, num_blocks * bands * sizeof(float)));
    
    findMinMaxKernel<<<num_blocks, block_size, shared_mem, stream>>>(
        d_data, num_pixels, bands, d_block_mins, d_block_maxs);
    
    // Reduce block results on CPU (small array)
    std::vector<float> h_block_mins(num_blocks * bands);
    std::vector<float> h_block_maxs(num_blocks * bands);
    CUDA_CHECK(cudaMemcpyAsync(h_block_mins.data(), d_block_mins, 
                                num_blocks * bands * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_block_maxs.data(), d_block_maxs,
                                num_blocks * bands * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    float h_mins[MAX_BANDS_USED], h_maxs[MAX_BANDS_USED];
    for (int b = 0; b < bands; b++) {
        h_mins[b] = INFINITY;
        h_maxs[b] = -INFINITY;
        for (int i = 0; i < num_blocks; i++) {
            h_mins[b] = fminf(h_mins[b], h_block_mins[i * bands + b]);
            h_maxs[b] = fmaxf(h_maxs[b], h_block_maxs[i * bands + b]);
        }
        // Handle edge cases
        if (!std::isfinite(h_mins[b])) h_mins[b] = 0.0f;
        if (!std::isfinite(h_maxs[b])) h_maxs[b] = 1.0f;
        if (h_maxs[b] <= h_mins[b]) h_maxs[b] = h_mins[b] + 1.0f;
    }
    
    CUDA_CHECK(cudaFree(d_block_mins));
    CUDA_CHECK(cudaFree(d_block_maxs));
    
    // Pass 2: Build histograms
    float *d_mins, *d_maxs;
    unsigned int* d_histograms;
    CUDA_CHECK(cudaMalloc(&d_mins, bands * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_maxs, bands * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_histograms, bands * HISTOGRAM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemsetAsync(d_histograms, 0, bands * HISTOGRAM_BINS * sizeof(unsigned int), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_mins, h_mins, bands * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_maxs, h_maxs, bands * sizeof(float), cudaMemcpyHostToDevice, stream));
    
    int hist_blocks = (num_pixels + block_size - 1) / block_size;
    buildHistogramKernel<<<hist_blocks, block_size, 0, stream>>>(
        d_data, num_pixels, bands, d_mins, d_maxs, d_histograms);
    
    // Copy histograms back to CPU for percentile computation
    std::vector<unsigned int> h_histograms(bands * HISTOGRAM_BINS);
    CUDA_CHECK(cudaMemcpyAsync(h_histograms.data(), d_histograms,
                                bands * HISTOGRAM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    CUDA_CHECK(cudaFree(d_mins));
    CUDA_CHECK(cudaFree(d_maxs));
    CUDA_CHECK(cudaFree(d_histograms));
    
    // Find percentile bounds from histograms
    for (int b = 0; b < bands; b++) {
        unsigned int* hist = &h_histograms[b * HISTOGRAM_BINS];
        
        // Count total valid pixels
        unsigned long total = 0;
        for (int i = 0; i < HISTOGRAM_BINS; i++) {
            total += hist[i];
        }
        
        if (total == 0) {
            stats[b].min_val = h_mins[b];
            stats[b].max_val = h_maxs[b];
            continue;
        }
        
        unsigned long trim_count = (unsigned long)(total * trim_percent / 100.0f);
        float range = h_maxs[b] - h_mins[b];
        
        // Find lower percentile
        unsigned long cumsum = 0;
        int lower_bin = 0;
        for (int i = 0; i < HISTOGRAM_BINS; i++) {
            cumsum += hist[i];
            if (cumsum >= trim_count) {
                lower_bin = i;
                break;
            }
        }
        
        // Find upper percentile
        cumsum = 0;
        int upper_bin = HISTOGRAM_BINS - 1;
        for (int i = HISTOGRAM_BINS - 1; i >= 0; i--) {
            cumsum += hist[i];
            if (cumsum >= trim_count) {
                upper_bin = i;
                break;
            }
        }
        
        // Convert bins back to values
        stats[b].min_val = h_mins[b] + (lower_bin / (float)(HISTOGRAM_BINS - 1)) * range;
        stats[b].max_val = h_mins[b] + (upper_bin / (float)(HISTOGRAM_BINS - 1)) * range;
        
        // Ensure valid range
        if (stats[b].max_val <= stats[b].min_val) {
            stats[b].max_val = stats[b].min_val + 0.001f;
        }
    }
}

// ============================================================================
// CPU fallback for histogram (used only if needed)
// ============================================================================

void computeHistogramBounds(const float* data, int width, int height, int bands,
                           BandStats* stats, float trim_percent) {
    size_t pixels = (size_t)width * height;
    
    for (int b = 0; b < bands; b++) {
        // Extract band values
        std::vector<float> values;
        values.reserve(pixels);
        
        for (size_t p = 0; p < pixels; p++) {
            float val = data[p * bands + b];
            if (std::isfinite(val)) {
                values.push_back(val);
            }
        }
        
        if (values.empty()) {
            stats[b].min_val = 0.0f;
            stats[b].max_val = 1.0f;
            continue;
        }
        
        std::sort(values.begin(), values.end());
        
        size_t trim_count = (size_t)(values.size() * trim_percent / 100.0f);
        size_t min_idx = trim_count;
        size_t max_idx = values.size() - 1 - trim_count;
        
        if (min_idx >= max_idx) {
            min_idx = 0;
            max_idx = values.size() - 1;
        }
        
        stats[b].min_val = values[min_idx];
        stats[b].max_val = values[max_idx];
        
        // Avoid division by zero
        if (stats[b].max_val <= stats[b].min_val) {
            stats[b].max_val = stats[b].min_val + 1.0f;
        }
    }
}

// ============================================================================
// CUDA Kernels
// ============================================================================

// Kernel: Apply scaling and convert float RGB to uint8 RGB
__global__ void scaleAndConvertKernel(const float* input, uint8_t* output,
                                       int width, int height, int bands,
                                       float min_r, float max_r,
                                       float min_g, float max_g,
                                       float min_b, float max_b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    
    if (idx >= total_pixels) return;
    
    float scale_r = 255.0f / (max_r - min_r);
    float scale_g = 255.0f / (max_g - min_g);
    float scale_b = 255.0f / (max_b - min_b);
    
    float r = input[idx * bands + 0];
    float g = (bands >= 2) ? input[idx * bands + 1] : r;
    float b = (bands >= 3) ? input[idx * bands + 2] : r;
    
    // Scale to 0-255
    r = (r - min_r) * scale_r;
    g = (g - min_g) * scale_g;
    b = (b - min_b) * scale_b;
    
    // Clamp
    output[idx * 3 + 0] = (uint8_t)fminf(fmaxf(r, 0.0f), 255.0f);
    output[idx * 3 + 1] = (uint8_t)fminf(fmaxf(g, 0.0f), 255.0f);
    output[idx * 3 + 2] = (uint8_t)fminf(fmaxf(b, 0.0f), 255.0f);
}

// Kernel: Horizontal wipe transition between two images
__global__ void wipeTransitionKernel(const uint8_t* img1, const uint8_t* img2,
                                      uint8_t* output, int width, int height,
                                      float progress) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = (y * width + x) * 3;
    
    // Wipe position with slight feathering
    float wipe_x = progress * (width + 50) - 25;  // Add 50px transition zone
    float blend = 0.0f;
    
    if ((float)x < wipe_x - 25.0f) {
        blend = 1.0f;  // Fully img2
    } else if ((float)x > wipe_x + 25.0f) {
        blend = 0.0f;  // Fully img1
    } else {
        // Smooth transition in the 50px zone
        blend = ((float)x - (wipe_x - 25.0f)) / 50.0f;
        blend = 1.0f - blend;
        blend = blend * blend * (3.0f - 2.0f * blend);  // Smoothstep
        blend = 1.0f - blend;
    }
    
    output[idx + 0] = (uint8_t)(img1[idx + 0] * (1.0f - blend) + img2[idx + 0] * blend);
    output[idx + 1] = (uint8_t)(img1[idx + 1] * (1.0f - blend) + img2[idx + 1] * blend);
    output[idx + 2] = (uint8_t)(img1[idx + 2] * (1.0f - blend) + img2[idx + 2] * blend);
}

// Kernel: Resize image using bilinear interpolation
__global__ void resizeKernel(const uint8_t* input, uint8_t* output,
                              int in_width, int in_height,
                              int out_width, int out_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= out_width || y >= out_height) return;
    
    float scale_x = (float)in_width / out_width;
    float scale_y = (float)in_height / out_height;
    
    float src_x = x * scale_x;
    float src_y = y * scale_y;
    
    int x0 = (int)src_x;
    int y0 = (int)src_y;
    int x1 = min(x0 + 1, in_width - 1);
    int y1 = min(y0 + 1, in_height - 1);
    
    float fx = src_x - x0;
    float fy = src_y - y0;
    
    for (int c = 0; c < 3; c++) {
        float v00 = input[(y0 * in_width + x0) * 3 + c];
        float v10 = input[(y0 * in_width + x1) * 3 + c];
        float v01 = input[(y1 * in_width + x0) * 3 + c];
        float v11 = input[(y1 * in_width + x1) * 3 + c];
        
        float v = v00 * (1-fx) * (1-fy) + v10 * fx * (1-fy) +
                  v01 * (1-fx) * fy + v11 * fx * fy;
        
        output[(y * out_width + x) * 3 + c] = (uint8_t)fminf(fmaxf(v, 0.0f), 255.0f);
    }
}

// ============================================================================
// Bitmap Font Data (8x16 basic ASCII font)
// Covers printable ASCII characters 32-126
// ============================================================================

// Compact font - each byte is one row of 8 pixels
__constant__ unsigned char d_font_8x16[95][16] = {
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // 32 space
    {0x00,0x00,0x18,0x3C,0x3C,0x3C,0x18,0x18,0x18,0x00,0x18,0x18,0x00,0x00,0x00,0x00}, // 33 !
    {0x00,0x66,0x66,0x66,0x24,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // 34 "
    {0x00,0x00,0x00,0x6C,0x6C,0xFE,0x6C,0x6C,0x6C,0xFE,0x6C,0x6C,0x00,0x00,0x00,0x00}, // 35 #
    {0x18,0x18,0x7C,0xC6,0xC2,0xC0,0x7C,0x06,0x86,0xC6,0x7C,0x18,0x18,0x00,0x00,0x00}, // 36 $
    {0x00,0x00,0x00,0x00,0xC2,0xC6,0x0C,0x18,0x30,0x60,0xC6,0x86,0x00,0x00,0x00,0x00}, // 37 %
    {0x00,0x00,0x38,0x6C,0x6C,0x38,0x76,0xDC,0xCC,0xCC,0xCC,0x76,0x00,0x00,0x00,0x00}, // 38 &
    {0x00,0x30,0x30,0x30,0x60,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // 39 '
    {0x00,0x00,0x0C,0x18,0x30,0x30,0x30,0x30,0x30,0x30,0x18,0x0C,0x00,0x00,0x00,0x00}, // 40 (
    {0x00,0x00,0x30,0x18,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x18,0x30,0x00,0x00,0x00,0x00}, // 41 )
    {0x00,0x00,0x00,0x00,0x00,0x66,0x3C,0xFF,0x3C,0x66,0x00,0x00,0x00,0x00,0x00,0x00}, // 42 *
    {0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x7E,0x18,0x18,0x00,0x00,0x00,0x00,0x00,0x00}, // 43 +
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x18,0x30,0x00,0x00,0x00}, // 44 ,
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xFE,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // 45 -
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x00,0x00,0x00,0x00}, // 46 .
    {0x00,0x00,0x00,0x00,0x02,0x06,0x0C,0x18,0x30,0x60,0xC0,0x80,0x00,0x00,0x00,0x00}, // 47 /
    {0x00,0x00,0x38,0x6C,0xC6,0xC6,0xD6,0xD6,0xC6,0xC6,0x6C,0x38,0x00,0x00,0x00,0x00}, // 48 0
    {0x00,0x00,0x18,0x38,0x78,0x18,0x18,0x18,0x18,0x18,0x18,0x7E,0x00,0x00,0x00,0x00}, // 49 1
    {0x00,0x00,0x7C,0xC6,0x06,0x0C,0x18,0x30,0x60,0xC0,0xC6,0xFE,0x00,0x00,0x00,0x00}, // 50 2
    {0x00,0x00,0x7C,0xC6,0x06,0x06,0x3C,0x06,0x06,0x06,0xC6,0x7C,0x00,0x00,0x00,0x00}, // 51 3
    {0x00,0x00,0x0C,0x1C,0x3C,0x6C,0xCC,0xFE,0x0C,0x0C,0x0C,0x1E,0x00,0x00,0x00,0x00}, // 52 4
    {0x00,0x00,0xFE,0xC0,0xC0,0xC0,0xFC,0x06,0x06,0x06,0xC6,0x7C,0x00,0x00,0x00,0x00}, // 53 5
    {0x00,0x00,0x38,0x60,0xC0,0xC0,0xFC,0xC6,0xC6,0xC6,0xC6,0x7C,0x00,0x00,0x00,0x00}, // 54 6
    {0x00,0x00,0xFE,0xC6,0x06,0x06,0x0C,0x18,0x30,0x30,0x30,0x30,0x00,0x00,0x00,0x00}, // 55 7
    {0x00,0x00,0x7C,0xC6,0xC6,0xC6,0x7C,0xC6,0xC6,0xC6,0xC6,0x7C,0x00,0x00,0x00,0x00}, // 56 8
    {0x00,0x00,0x7C,0xC6,0xC6,0xC6,0x7E,0x06,0x06,0x06,0x0C,0x78,0x00,0x00,0x00,0x00}, // 57 9
    {0x00,0x00,0x00,0x00,0x18,0x18,0x00,0x00,0x00,0x18,0x18,0x00,0x00,0x00,0x00,0x00}, // 58 :
    {0x00,0x00,0x00,0x00,0x18,0x18,0x00,0x00,0x00,0x18,0x18,0x30,0x00,0x00,0x00,0x00}, // 59 ;
    {0x00,0x00,0x00,0x06,0x0C,0x18,0x30,0x60,0x30,0x18,0x0C,0x06,0x00,0x00,0x00,0x00}, // 60 <
    {0x00,0x00,0x00,0x00,0x00,0x7E,0x00,0x00,0x7E,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // 61 =
    {0x00,0x00,0x00,0x60,0x30,0x18,0x0C,0x06,0x0C,0x18,0x30,0x60,0x00,0x00,0x00,0x00}, // 62 >
    {0x00,0x00,0x7C,0xC6,0xC6,0x0C,0x18,0x18,0x18,0x00,0x18,0x18,0x00,0x00,0x00,0x00}, // 63 ?
    {0x00,0x00,0x00,0x7C,0xC6,0xC6,0xDE,0xDE,0xDE,0xDC,0xC0,0x7C,0x00,0x00,0x00,0x00}, // 64 @
    {0x00,0x00,0x10,0x38,0x6C,0xC6,0xC6,0xFE,0xC6,0xC6,0xC6,0xC6,0x00,0x00,0x00,0x00}, // 65 A
    {0x00,0x00,0xFC,0x66,0x66,0x66,0x7C,0x66,0x66,0x66,0x66,0xFC,0x00,0x00,0x00,0x00}, // 66 B
    {0x00,0x00,0x3C,0x66,0xC2,0xC0,0xC0,0xC0,0xC0,0xC2,0x66,0x3C,0x00,0x00,0x00,0x00}, // 67 C
    {0x00,0x00,0xF8,0x6C,0x66,0x66,0x66,0x66,0x66,0x66,0x6C,0xF8,0x00,0x00,0x00,0x00}, // 68 D
    {0x00,0x00,0xFE,0x66,0x62,0x68,0x78,0x68,0x60,0x62,0x66,0xFE,0x00,0x00,0x00,0x00}, // 69 E
    {0x00,0x00,0xFE,0x66,0x62,0x68,0x78,0x68,0x60,0x60,0x60,0xF0,0x00,0x00,0x00,0x00}, // 70 F
    {0x00,0x00,0x3C,0x66,0xC2,0xC0,0xC0,0xDE,0xC6,0xC6,0x66,0x3A,0x00,0x00,0x00,0x00}, // 71 G
    {0x00,0x00,0xC6,0xC6,0xC6,0xC6,0xFE,0xC6,0xC6,0xC6,0xC6,0xC6,0x00,0x00,0x00,0x00}, // 72 H
    {0x00,0x00,0x3C,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x3C,0x00,0x00,0x00,0x00}, // 73 I
    {0x00,0x00,0x1E,0x0C,0x0C,0x0C,0x0C,0x0C,0xCC,0xCC,0xCC,0x78,0x00,0x00,0x00,0x00}, // 74 J
    {0x00,0x00,0xE6,0x66,0x66,0x6C,0x78,0x78,0x6C,0x66,0x66,0xE6,0x00,0x00,0x00,0x00}, // 75 K
    {0x00,0x00,0xF0,0x60,0x60,0x60,0x60,0x60,0x60,0x62,0x66,0xFE,0x00,0x00,0x00,0x00}, // 76 L
    {0x00,0x00,0xC6,0xEE,0xFE,0xFE,0xD6,0xC6,0xC6,0xC6,0xC6,0xC6,0x00,0x00,0x00,0x00}, // 77 M
    {0x00,0x00,0xC6,0xE6,0xF6,0xFE,0xDE,0xCE,0xC6,0xC6,0xC6,0xC6,0x00,0x00,0x00,0x00}, // 78 N
    {0x00,0x00,0x7C,0xC6,0xC6,0xC6,0xC6,0xC6,0xC6,0xC6,0xC6,0x7C,0x00,0x00,0x00,0x00}, // 79 O
    {0x00,0x00,0xFC,0x66,0x66,0x66,0x7C,0x60,0x60,0x60,0x60,0xF0,0x00,0x00,0x00,0x00}, // 80 P
    {0x00,0x00,0x7C,0xC6,0xC6,0xC6,0xC6,0xC6,0xC6,0xD6,0xDE,0x7C,0x0C,0x0E,0x00,0x00}, // 81 Q
    {0x00,0x00,0xFC,0x66,0x66,0x66,0x7C,0x6C,0x66,0x66,0x66,0xE6,0x00,0x00,0x00,0x00}, // 82 R
    {0x00,0x00,0x7C,0xC6,0xC6,0x60,0x38,0x0C,0x06,0xC6,0xC6,0x7C,0x00,0x00,0x00,0x00}, // 83 S
    {0x00,0x00,0x7E,0x7E,0x5A,0x18,0x18,0x18,0x18,0x18,0x18,0x3C,0x00,0x00,0x00,0x00}, // 84 T
    {0x00,0x00,0xC6,0xC6,0xC6,0xC6,0xC6,0xC6,0xC6,0xC6,0xC6,0x7C,0x00,0x00,0x00,0x00}, // 85 U
    {0x00,0x00,0xC6,0xC6,0xC6,0xC6,0xC6,0xC6,0xC6,0x6C,0x38,0x10,0x00,0x00,0x00,0x00}, // 86 V
    {0x00,0x00,0xC6,0xC6,0xC6,0xC6,0xD6,0xD6,0xD6,0xFE,0xEE,0x6C,0x00,0x00,0x00,0x00}, // 87 W
    {0x00,0x00,0xC6,0xC6,0x6C,0x7C,0x38,0x38,0x7C,0x6C,0xC6,0xC6,0x00,0x00,0x00,0x00}, // 88 X
    {0x00,0x00,0x66,0x66,0x66,0x66,0x3C,0x18,0x18,0x18,0x18,0x3C,0x00,0x00,0x00,0x00}, // 89 Y
    {0x00,0x00,0xFE,0xC6,0x86,0x0C,0x18,0x30,0x60,0xC2,0xC6,0xFE,0x00,0x00,0x00,0x00}, // 90 Z
    {0x00,0x00,0x3C,0x30,0x30,0x30,0x30,0x30,0x30,0x30,0x30,0x3C,0x00,0x00,0x00,0x00}, // 91 [
    {0x00,0x00,0x00,0x80,0xC0,0xE0,0x70,0x38,0x1C,0x0E,0x06,0x02,0x00,0x00,0x00,0x00}, // 92 backslash
    {0x00,0x00,0x3C,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x3C,0x00,0x00,0x00,0x00}, // 93 ]
    {0x10,0x38,0x6C,0xC6,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // 94 ^
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xFF,0x00,0x00}, // 95 _
    {0x30,0x30,0x18,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // 96 `
    {0x00,0x00,0x00,0x00,0x00,0x78,0x0C,0x7C,0xCC,0xCC,0xCC,0x76,0x00,0x00,0x00,0x00}, // 97 a
    {0x00,0x00,0xE0,0x60,0x60,0x78,0x6C,0x66,0x66,0x66,0x66,0x7C,0x00,0x00,0x00,0x00}, // 98 b
    {0x00,0x00,0x00,0x00,0x00,0x7C,0xC6,0xC0,0xC0,0xC0,0xC6,0x7C,0x00,0x00,0x00,0x00}, // 99 c
    {0x00,0x00,0x1C,0x0C,0x0C,0x3C,0x6C,0xCC,0xCC,0xCC,0xCC,0x76,0x00,0x00,0x00,0x00}, // 100 d
    {0x00,0x00,0x00,0x00,0x00,0x7C,0xC6,0xFE,0xC0,0xC0,0xC6,0x7C,0x00,0x00,0x00,0x00}, // 101 e
    {0x00,0x00,0x38,0x6C,0x64,0x60,0xF0,0x60,0x60,0x60,0x60,0xF0,0x00,0x00,0x00,0x00}, // 102 f
    {0x00,0x00,0x00,0x00,0x00,0x76,0xCC,0xCC,0xCC,0xCC,0xCC,0x7C,0x0C,0xCC,0x78,0x00}, // 103 g
    {0x00,0x00,0xE0,0x60,0x60,0x6C,0x76,0x66,0x66,0x66,0x66,0xE6,0x00,0x00,0x00,0x00}, // 104 h
    {0x00,0x00,0x18,0x18,0x00,0x38,0x18,0x18,0x18,0x18,0x18,0x3C,0x00,0x00,0x00,0x00}, // 105 i
    {0x00,0x00,0x06,0x06,0x00,0x0E,0x06,0x06,0x06,0x06,0x06,0x06,0x66,0x66,0x3C,0x00}, // 106 j
    {0x00,0x00,0xE0,0x60,0x60,0x66,0x6C,0x78,0x78,0x6C,0x66,0xE6,0x00,0x00,0x00,0x00}, // 107 k
    {0x00,0x00,0x38,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x3C,0x00,0x00,0x00,0x00}, // 108 l
    {0x00,0x00,0x00,0x00,0x00,0xEC,0xFE,0xD6,0xD6,0xD6,0xD6,0xC6,0x00,0x00,0x00,0x00}, // 109 m
    {0x00,0x00,0x00,0x00,0x00,0xDC,0x66,0x66,0x66,0x66,0x66,0x66,0x00,0x00,0x00,0x00}, // 110 n
    {0x00,0x00,0x00,0x00,0x00,0x7C,0xC6,0xC6,0xC6,0xC6,0xC6,0x7C,0x00,0x00,0x00,0x00}, // 111 o
    {0x00,0x00,0x00,0x00,0x00,0xDC,0x66,0x66,0x66,0x66,0x66,0x7C,0x60,0x60,0xF0,0x00}, // 112 p
    {0x00,0x00,0x00,0x00,0x00,0x76,0xCC,0xCC,0xCC,0xCC,0xCC,0x7C,0x0C,0x0C,0x1E,0x00}, // 113 q
    {0x00,0x00,0x00,0x00,0x00,0xDC,0x76,0x66,0x60,0x60,0x60,0xF0,0x00,0x00,0x00,0x00}, // 114 r
    {0x00,0x00,0x00,0x00,0x00,0x7C,0xC6,0x60,0x38,0x0C,0xC6,0x7C,0x00,0x00,0x00,0x00}, // 115 s
    {0x00,0x00,0x10,0x30,0x30,0xFC,0x30,0x30,0x30,0x30,0x36,0x1C,0x00,0x00,0x00,0x00}, // 116 t
    {0x00,0x00,0x00,0x00,0x00,0xCC,0xCC,0xCC,0xCC,0xCC,0xCC,0x76,0x00,0x00,0x00,0x00}, // 117 u
    {0x00,0x00,0x00,0x00,0x00,0x66,0x66,0x66,0x66,0x66,0x3C,0x18,0x00,0x00,0x00,0x00}, // 118 v
    {0x00,0x00,0x00,0x00,0x00,0xC6,0xC6,0xD6,0xD6,0xD6,0xFE,0x6C,0x00,0x00,0x00,0x00}, // 119 w
    {0x00,0x00,0x00,0x00,0x00,0xC6,0x6C,0x38,0x38,0x38,0x6C,0xC6,0x00,0x00,0x00,0x00}, // 120 x
    {0x00,0x00,0x00,0x00,0x00,0xC6,0xC6,0xC6,0xC6,0xC6,0xC6,0x7E,0x06,0x0C,0xF8,0x00}, // 121 y
    {0x00,0x00,0x00,0x00,0x00,0xFE,0xCC,0x18,0x30,0x60,0xC6,0xFE,0x00,0x00,0x00,0x00}, // 122 z
    {0x00,0x00,0x0E,0x18,0x18,0x18,0x70,0x18,0x18,0x18,0x18,0x0E,0x00,0x00,0x00,0x00}, // 123 {
    {0x00,0x00,0x18,0x18,0x18,0x18,0x00,0x18,0x18,0x18,0x18,0x18,0x00,0x00,0x00,0x00}, // 124 |
    {0x00,0x00,0x70,0x18,0x18,0x18,0x0E,0x18,0x18,0x18,0x18,0x70,0x00,0x00,0x00,0x00}, // 125 }
    {0x00,0x00,0x76,0xDC,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // 126 ~
};

// Kernel: Render text overlay on image (with shadow for visibility)
__global__ void renderTextKernel(uint8_t* image, int img_width, int img_height,
                                  const char* text, int text_len,
                                  int start_x, int start_y, int scale) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread handles one character
    if (tid >= text_len) return;
    
    char c = text[tid];
    if (c < 32 || c > 126) return;  // Skip non-printable
    
    int char_idx = c - 32;
    int char_x = start_x + tid * FONT_CHAR_WIDTH * scale;
    
    // Render each pixel of the character
    for (int row = 0; row < FONT_CHAR_HEIGHT; row++) {
        unsigned char font_row = d_font_8x16[char_idx][row];
        for (int col = 0; col < FONT_CHAR_WIDTH; col++) {
            if (font_row & (0x80 >> col)) {
                // Draw scaled pixel
                for (int sy = 0; sy < scale; sy++) {
                    for (int sx = 0; sx < scale; sx++) {
                        int px = char_x + col * scale + sx;
                        int py = start_y + row * scale + sy;
                        
                        if (px >= 0 && px < img_width && py >= 0 && py < img_height) {
                            int idx = (py * img_width + px) * 3;
                            
                            // Draw shadow (offset by 2 pixels)
                            int shadow_px = px + 2;
                            int shadow_py = py + 2;
                            if (shadow_px < img_width && shadow_py < img_height) {
                                int shadow_idx = (shadow_py * img_width + shadow_px) * 3;
                                image[shadow_idx + 0] = 0;
                                image[shadow_idx + 1] = 0;
                                image[shadow_idx + 2] = 0;
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Second pass: draw white text on top
    for (int row = 0; row < FONT_CHAR_HEIGHT; row++) {
        unsigned char font_row = d_font_8x16[char_idx][row];
        for (int col = 0; col < FONT_CHAR_WIDTH; col++) {
            if (font_row & (0x80 >> col)) {
                for (int sy = 0; sy < scale; sy++) {
                    for (int sx = 0; sx < scale; sx++) {
                        int px = char_x + col * scale + sx;
                        int py = start_y + row * scale + sy;
                        
                        if (px >= 0 && px < img_width && py >= 0 && py < img_height) {
                            int idx = (py * img_width + px) * 3;
                            image[idx + 0] = 255;  // White text
                            image[idx + 1] = 255;
                            image[idx + 2] = 255;
                        }
                    }
                }
            }
        }
    }
}

// Host function to render filename on frame
void renderFilenameOverlay(uint8_t* d_image, int width, int height,
                           const char* filename, cudaStream_t stream) {
    // Copy filename to device
    int text_len = strlen(filename);
    if (text_len > 256) text_len = 256;  // Limit length
    
    char* d_text;
    CUDA_CHECK(cudaMalloc(&d_text, text_len + 1));
    CUDA_CHECK(cudaMemcpyAsync(d_text, filename, text_len + 1, cudaMemcpyHostToDevice, stream));
    
    // Position: lower-left corner with margin
    int text_y = height - TEXT_MARGIN - FONT_CHAR_HEIGHT * FONT_SCALE;
    int text_x = TEXT_MARGIN;
    
    // Launch kernel (one thread per character)
    int block_size = 128;
    int num_blocks = (text_len + block_size - 1) / block_size;
    
    renderTextKernel<<<num_blocks, block_size, 0, stream>>>(
        d_image, width, height, d_text, text_len, text_x, text_y, FONT_SCALE);
    
    CUDA_CHECK(cudaFree(d_text));
}

// ============================================================================
// Multi-threaded File Loader
// ============================================================================

struct LoaderThreadData {
    int thread_id;
};

void* loaderThreadFunc(void* arg) {
    LoaderThreadData* data = (LoaderThreadData*)arg;
    
    while (true) {
        int load_index = g_next_load_index.fetch_add(1);
        
        if (load_index >= (int)g_image_files.size()) {
            break;
        }
        
        const ImageFile& img = g_image_files[load_index];
        ENVIHeader header;
        
        printf("[Thread %2d] Loading image %d: %s\n", data->thread_id, load_index, img.filename.c_str());
        
        float* raw_data = readENVIFile(img, header);
        
        ImageData img_data;
        img_data.index = load_index;
        img_data.valid = (raw_data != nullptr);
        
        if (raw_data) {
            img_data.host_data = raw_data;
            img_data.width = header.samples;
            img_data.height = header.lines;
            img_data.data_size = (size_t)header.samples * header.lines * 
                                 std::min(header.bands, MAX_BANDS_USED) * sizeof(float);
            
            // Set global dimensions from first image
            if (load_index == 0) {
                g_image_width = header.samples;
                g_image_height = header.lines;
                g_image_bands = std::min(header.bands, MAX_BANDS_USED);
                printf("Image dimensions: %d x %d, %d bands\n", 
                       g_image_width, g_image_height, g_image_bands);
            }
        } else {
            img_data.host_data = nullptr;
            img_data.width = 0;
            img_data.height = 0;
            img_data.data_size = 0;
        }
        
        // Add to queue with flow control
        {
            std::unique_lock<std::mutex> lock(g_queue_mutex);
            
            // Wait if queue is full
            g_producer_cv.wait(lock, []{ 
                return g_image_queue.size() < MAX_QUEUE_SIZE; 
            });
            
            g_image_queue.push(img_data);
            g_consumer_cv.notify_one();
        }
    }
    
    return nullptr;
}

// ============================================================================
// Video Encoder using FFmpeg
// ============================================================================

struct VideoEncoder {
    AVFormatContext* format_ctx;
    AVCodecContext* codec_ctx;
    AVStream* stream;
    AVFrame* frame;
    AVPacket* packet;
    struct SwsContext* sws_ctx;
    int64_t pts;
    
    bool init(const char* filename, int width, int height, int fps) {
        pts = 0;
        
        // Allocate format context
        avformat_alloc_output_context2(&format_ctx, NULL, NULL, filename);
        if (!format_ctx) {
            fprintf(stderr, "Could not create output context\n");
            return false;
        }
        
        // Find encoder - try NVENC first, fall back to libx264
        const AVCodec* codec = avcodec_find_encoder_by_name("h264_nvenc");
        if (!codec) {
            printf("NVENC not available, falling back to libx264\n");
            codec = avcodec_find_encoder_by_name("libx264");
        }
        if (!codec) {
            codec = avcodec_find_encoder(AV_CODEC_ID_H264);
        }
        if (!codec) {
            fprintf(stderr, "H264 encoder not found\n");
            return false;
        }
        printf("Using encoder: %s\n", codec->name);
        
        // Create stream
        stream = avformat_new_stream(format_ctx, NULL);
        if (!stream) {
            fprintf(stderr, "Could not create stream\n");
            return false;
        }
        
        // Allocate codec context
        codec_ctx = avcodec_alloc_context3(codec);
        if (!codec_ctx) {
            fprintf(stderr, "Could not allocate codec context\n");
            return false;
        }
        
        // Configure codec
        codec_ctx->width = width;
        codec_ctx->height = height;
        codec_ctx->time_base = (AVRational){1, fps};
        codec_ctx->framerate = (AVRational){fps, 1};
        codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
        codec_ctx->gop_size = 12;
        codec_ctx->max_b_frames = 2;
        
        // NVENC specific options
        if (strcmp(codec->name, "h264_nvenc") == 0) {
            av_opt_set(codec_ctx->priv_data, "preset", "p4", 0);  // Balanced preset
            av_opt_set(codec_ctx->priv_data, "tune", "hq", 0);
            av_opt_set(codec_ctx->priv_data, "rc", "vbr", 0);
            codec_ctx->bit_rate = VIDEO_BITRATE;  // 24 Mbps
        } else {
            av_opt_set(codec_ctx->priv_data, "preset", "medium", 0);
            av_opt_set(codec_ctx->priv_data, "crf", "18", 0);  // Higher quality (lower CRF)
        }
        
        if (format_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
            codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        }
        
        // Open codec
        if (avcodec_open2(codec_ctx, codec, NULL) < 0) {
            fprintf(stderr, "Could not open codec\n");
            return false;
        }
        
        // Copy codec params to stream
        avcodec_parameters_from_context(stream->codecpar, codec_ctx);
        stream->time_base = codec_ctx->time_base;
        
        // Open output file
        if (!(format_ctx->oformat->flags & AVFMT_NOFILE)) {
            if (avio_open(&format_ctx->pb, filename, AVIO_FLAG_WRITE) < 0) {
                fprintf(stderr, "Could not open output file '%s'\n", filename);
                return false;
            }
        }
        
        // Write header
        if (avformat_write_header(format_ctx, NULL) < 0) {
            fprintf(stderr, "Could not write header\n");
            return false;
        }
        
        // Allocate frame
        frame = av_frame_alloc();
        frame->format = codec_ctx->pix_fmt;
        frame->width = width;
        frame->height = height;
        av_frame_get_buffer(frame, 0);
        
        // Allocate packet
        packet = av_packet_alloc();
        
        // Create scaler for RGB to YUV conversion
        sws_ctx = sws_getContext(width, height, AV_PIX_FMT_RGB24,
                                  width, height, AV_PIX_FMT_YUV420P,
                                  SWS_BILINEAR, NULL, NULL, NULL);
        
        return true;
    }
    
    bool writeFrame(uint8_t* rgb_data) {
        av_frame_make_writable(frame);
        
        // Convert RGB to YUV
        uint8_t* src_data[1] = { rgb_data };
        int src_linesize[1] = { codec_ctx->width * 3 };
        sws_scale(sws_ctx, src_data, src_linesize, 0, codec_ctx->height,
                  frame->data, frame->linesize);
        
        frame->pts = pts++;
        
        // Encode frame
        if (avcodec_send_frame(codec_ctx, frame) < 0) {
            fprintf(stderr, "Error sending frame\n");
            return false;
        }
        
        while (true) {
            int ret = avcodec_receive_packet(codec_ctx, packet);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
            if (ret < 0) {
                fprintf(stderr, "Error receiving packet\n");
                return false;
            }
            
            av_packet_rescale_ts(packet, codec_ctx->time_base, stream->time_base);
            packet->stream_index = stream->index;
            
            av_interleaved_write_frame(format_ctx, packet);
            av_packet_unref(packet);
        }
        
        return true;
    }
    
    void finish() {
        // Flush encoder
        avcodec_send_frame(codec_ctx, NULL);
        while (true) {
            int ret = avcodec_receive_packet(codec_ctx, packet);
            if (ret == AVERROR_EOF) break;
            if (ret < 0) break;
            
            av_packet_rescale_ts(packet, codec_ctx->time_base, stream->time_base);
            packet->stream_index = stream->index;
            av_interleaved_write_frame(format_ctx, packet);
            av_packet_unref(packet);
        }
        
        av_write_trailer(format_ctx);
        
        // Cleanup
        sws_freeContext(sws_ctx);
        av_frame_free(&frame);
        av_packet_free(&packet);
        avcodec_free_context(&codec_ctx);
        if (!(format_ctx->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&format_ctx->pb);
        }
        avformat_free_context(format_ctx);
    }
};

// ============================================================================
// Main Processing Pipeline
// ============================================================================

int main(int argc, char** argv) {
    const char* input_dir = ".";
    const char* output_file = "output.mp4";
    
    if (argc >= 2) input_dir = argv[1];
    if (argc >= 3) output_file = argv[2];
    
    printf("=== CUDA ENVI to Video Converter ===\n");
    printf("Input directory: %s\n", input_dir);
    printf("Output file: %s\n\n", output_file);
    
    // Find and sort .bin files
    g_image_files = findAndSortBinFiles(input_dir);
    
    if (g_image_files.empty()) {
        fprintf(stderr, "No valid .bin files found in %s\n", input_dir);
        return 1;
    }
    
    printf("Found %zu images:\n", g_image_files.size());
    for (const auto& img : g_image_files) {
        printf("  [%d] %s (timestamp: %s)\n", img.index, img.filename.c_str(), img.timestamp.c_str());
    }
    printf("\n");
    
    // ========================================================================
    // Phase 1: Setup GPU and compute histogram bounds (GPU-accelerated)
    // ========================================================================
    printf("=== Phase 1: Initializing GPU ===\n");
    
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Using GPU: %s\n", prop.name);
    printf("Total VRAM: %.2f GB\n", prop.totalGlobalMem / (1024.0*1024.0*1024.0));
    
    // Read first image to get dimensions
    {
        ENVIHeader header;
        float* data = readENVIFile(g_image_files[0], header);
        if (!data) {
            fprintf(stderr, "Failed to read first image\n");
            return 1;
        }
        g_image_width = header.samples;
        g_image_height = header.lines;
        g_image_bands = std::min(header.bands, MAX_BANDS_USED);
        free(data);
    }
    
    printf("Image dimensions: %d x %d, %d bands\n", g_image_width, g_image_height, g_image_bands);
    
    // Calculate memory requirements
    size_t pixels = (size_t)g_image_width * g_image_height;
    size_t float_image_size = pixels * g_image_bands * sizeof(float);
    size_t rgb_image_size = pixels * 3;
    size_t output_rgb_size = OUTPUT_WIDTH * OUTPUT_HEIGHT * 3;
    
    printf("Per-image GPU memory: %.2f MB (float) + %.2f MB (RGB8)\n",
           float_image_size / (1024.0*1024.0), rgb_image_size / (1024.0*1024.0));
    
    printf("\n=== Phase 2: Computing histogram bounds (GPU-accelerated, multi-threaded) ===\n");
    
    std::vector<BandStats> all_stats(g_image_files.size() * MAX_BANDS_USED);
    
    // Histogram phase structures
    struct HistImageData {
        int index;
        float* host_data;
        ENVIHeader header;
        bool valid;
    };
    
    std::queue<HistImageData> hist_queue;
    std::mutex hist_queue_mutex;
    std::condition_variable hist_producer_cv;
    std::condition_variable hist_consumer_cv;
    std::atomic<int> hist_next_load_index(0);
    std::atomic<bool> hist_loading_complete(false);
    const size_t HIST_QUEUE_SIZE = GPU_PREFETCH_BUFFER;
    
    // Histogram loader thread function
    auto histLoaderFunc = [&](int thread_id) {
        while (true) {
            int load_index = hist_next_load_index.fetch_add(1);
            
            if (load_index >= (int)g_image_files.size()) {
                break;
            }
            
            const ImageFile& img = g_image_files[load_index];
            ENVIHeader header;
            
            float* raw_data = readENVIFile(img, header);
            
            HistImageData img_data;
            img_data.index = load_index;
            img_data.host_data = raw_data;
            img_data.header = header;
            img_data.valid = (raw_data != nullptr);
            
            // Add to queue with flow control
            {
                std::unique_lock<std::mutex> lock(hist_queue_mutex);
                hist_producer_cv.wait(lock, [&]{ 
                    return hist_queue.size() < HIST_QUEUE_SIZE; 
                });
                hist_queue.push(img_data);
                hist_consumer_cv.notify_one();
            }
        }
    };
    
    // Start histogram loader threads
    std::vector<std::thread> hist_loader_threads;
    for (int i = 0; i < NUM_LOADER_THREADS; i++) {
        hist_loader_threads.emplace_back(histLoaderFunc, i);
    }
    
    // Allocate GPU buffers for histogram computation
    float* d_hist_buffer;
    CUDA_CHECK(cudaMalloc(&d_hist_buffer, float_image_size));
    
    cudaStream_t hist_stream;
    CUDA_CHECK(cudaStreamCreate(&hist_stream));
    
    // Process images as they arrive from loader threads
    int hist_images_processed = 0;
    while (hist_images_processed < (int)g_image_files.size()) {
        HistImageData img_data;
        {
            std::unique_lock<std::mutex> lock(hist_queue_mutex);
            hist_consumer_cv.wait(lock, [&]{ return !hist_queue.empty(); });
            img_data = hist_queue.front();
            hist_queue.pop();
            hist_producer_cv.notify_one();
        }
        
        if (img_data.valid && img_data.host_data) {
            int bands = std::min(img_data.header.bands, MAX_BANDS_USED);
            
            // Upload to GPU
            CUDA_CHECK(cudaMemcpyAsync(d_hist_buffer, img_data.host_data, float_image_size,
                                        cudaMemcpyHostToDevice, hist_stream));
            
            // Compute histogram bounds on GPU
            computeHistogramBoundsGPU(d_hist_buffer, img_data.header.samples, img_data.header.lines, bands,
                                      &all_stats[img_data.index * MAX_BANDS_USED], HISTOGRAM_TRIM_PERCENT, hist_stream);
            
            printf("  Image %d bounds: R[%.4f, %.4f] G[%.4f, %.4f] B[%.4f, %.4f]\n",
                   img_data.index, 
                   all_stats[img_data.index*MAX_BANDS_USED].min_val, 
                   all_stats[img_data.index*MAX_BANDS_USED].max_val,
                   bands >= 2 ? all_stats[img_data.index*MAX_BANDS_USED+1].min_val : 0,
                   bands >= 2 ? all_stats[img_data.index*MAX_BANDS_USED+1].max_val : 0,
                   bands >= 3 ? all_stats[img_data.index*MAX_BANDS_USED+2].min_val : 0,
                   bands >= 3 ? all_stats[img_data.index*MAX_BANDS_USED+2].max_val : 0);
            
            free(img_data.host_data);
        }
        
        hist_images_processed++;
    }
    
    // Wait for all histogram loader threads to finish
    for (auto& t : hist_loader_threads) {
        t.join();
    }
    
    CUDA_CHECK(cudaFree(d_hist_buffer));
    CUDA_CHECK(cudaStreamDestroy(hist_stream));
    
    // Average the bounds
    for (int b = 0; b < MAX_BANDS_USED; b++) {
        float sum_min = 0, sum_max = 0;
        int count = 0;
        for (size_t i = 0; i < g_image_files.size(); i++) {
            sum_min += all_stats[i * MAX_BANDS_USED + b].min_val;
            sum_max += all_stats[i * MAX_BANDS_USED + b].max_val;
            count++;
        }
        g_global_bounds[b].min_val = sum_min / count;
        g_global_bounds[b].max_val = sum_max / count;
    }
    
    printf("\nAveraged global bounds:\n");
    printf("  R: [%.4f, %.4f]\n", g_global_bounds[0].min_val, g_global_bounds[0].max_val);
    printf("  G: [%.4f, %.4f]\n", g_global_bounds[1].min_val, g_global_bounds[1].max_val);
    printf("  B: [%.4f, %.4f]\n\n", g_global_bounds[2].min_val, g_global_bounds[2].max_val);
    
    // ========================================================================
    // Phase 3: Allocate GPU resources for video generation
    // ========================================================================
    printf("=== Phase 3: Allocating GPU resources ===\n");
    
    // Allocate GPU buffers for ring buffer
    std::vector<GPUBufferEntry> gpu_buffers(GPU_PREFETCH_BUFFER);
    for (int i = 0; i < GPU_PREFETCH_BUFFER; i++) {
        CUDA_CHECK(cudaMalloc(&gpu_buffers[i].d_float_data, float_image_size));
        CUDA_CHECK(cudaMalloc(&gpu_buffers[i].d_rgb_data, rgb_image_size));
        gpu_buffers[i].image_index = -1;
        gpu_buffers[i].ready = false;
        gpu_buffers[i].processed = false;
    }
    
    // Allocate output buffers
    uint8_t *d_resized1, *d_resized2, *d_output;
    CUDA_CHECK(cudaMalloc(&d_resized1, output_rgb_size));
    CUDA_CHECK(cudaMalloc(&d_resized2, output_rgb_size));
    CUDA_CHECK(cudaMalloc(&d_output, output_rgb_size));
    
    uint8_t* h_output = (uint8_t*)malloc(output_rgb_size);
    
    // Create CUDA stream for async operations
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // ========================================================================
    // Phase 4: Start loader threads
    // ========================================================================
    printf("\n=== Phase 4: Starting video generation ===\n");
    
    pthread_t loader_threads[NUM_LOADER_THREADS];
    LoaderThreadData thread_data[NUM_LOADER_THREADS];
    
    for (int i = 0; i < NUM_LOADER_THREADS; i++) {
        thread_data[i].thread_id = i;
        pthread_create(&loader_threads[i], NULL, loaderThreadFunc, &thread_data[i]);
    }
    
    // ========================================================================
    // Phase 5: Initialize video encoder
    // ========================================================================
    VideoEncoder encoder;
    if (!encoder.init(output_file, OUTPUT_WIDTH, OUTPUT_HEIGHT, OUTPUT_FPS)) {
        fprintf(stderr, "Failed to initialize video encoder\n");
        return 1;
    }
    
    // ========================================================================
    // Phase 6: Main processing loop
    // ========================================================================
    
    // CUDA kernel configuration
    dim3 block_1d(256);
    dim3 grid_1d((pixels + block_1d.x - 1) / block_1d.x);
    
    dim3 block_2d(16, 16);
    dim3 grid_2d_input((g_image_width + 15) / 16, (g_image_height + 15) / 16);
    dim3 grid_2d_output((OUTPUT_WIDTH + 15) / 16, (OUTPUT_HEIGHT + 15) / 16);
    
    int current_buffer_idx = 0;
    int prev_buffer_idx = -1;
    int images_processed = 0;
    int total_frames_written = 0;
    
    // Map to track which buffer holds which image
    std::map<int, int> image_to_buffer;
    
    while (images_processed < (int)g_image_files.size()) {
        // Get next image from queue
        /* ImageData img_data;
        {
            std::unique_lock<std::mutex> lock(g_queue_mutex);
            g_consumer_cv.wait(lock, []{ return !g_image_queue.empty(); });
            img_data = g_image_queue.front();
            g_image_queue.pop();
            g_producer_cv.notify_one();
        } */

        ImageData img_data;
bool have_image = false;

while (!have_image) {
    {
        std::unique_lock<std::mutex> lock(g_queue_mutex);
        g_consumer_cv.wait(lock, []{
            return !g_image_queue.empty() || g_loading_complete.load();
        });

        while (!g_image_queue.empty()) {
            ImageData tmp = g_image_queue.front();
            g_image_queue.pop();
            g_reorder_buffer[tmp.index] = tmp;
            g_producer_cv.notify_one();
        }
    }

    auto it = g_reorder_buffer.find(g_next_process_index.load());
    if (it != g_reorder_buffer.end()) {
        img_data = it->second;
        g_reorder_buffer.erase(it);
        g_next_process_index++;
        have_image = true;
    }
}

        
        if (!img_data.valid || !img_data.host_data) {
            fprintf(stderr, "Skipping invalid image %d\n", img_data.index);
            images_processed++;
            continue;
        }
        
        printf("Processing image %d/%zu: %s\n", 
               img_data.index + 1, g_image_files.size(), 
               g_image_files[img_data.index].filename.c_str());
        
        // Find a free buffer slot
        int buf_idx = current_buffer_idx % GPU_PREFETCH_BUFFER;
        
        // Upload float data to GPU
        CUDA_CHECK(cudaMemcpyAsync(gpu_buffers[buf_idx].d_float_data, img_data.host_data,
                                    float_image_size, cudaMemcpyHostToDevice, stream));
        
        // Scale and convert to RGB8
        scaleAndConvertKernel<<<grid_1d, block_1d, 0, stream>>>(
            gpu_buffers[buf_idx].d_float_data,
            gpu_buffers[buf_idx].d_rgb_data,
            g_image_width, g_image_height, g_image_bands,
            g_global_bounds[0].min_val, g_global_bounds[0].max_val,
            g_global_bounds[1].min_val, g_global_bounds[1].max_val,
            g_global_bounds[2].min_val, g_global_bounds[2].max_val
        );
        
        // Resize to output resolution
        resizeKernel<<<grid_2d_output, block_2d, 0, stream>>>(
            gpu_buffers[buf_idx].d_rgb_data, d_resized2,
            g_image_width, g_image_height,
            OUTPUT_WIDTH, OUTPUT_HEIGHT
        );
        
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        gpu_buffers[buf_idx].image_index = img_data.index;
        gpu_buffers[buf_idx].ready = true;
        image_to_buffer[img_data.index] = buf_idx;
        
        // Generate transition frames if we have a previous image
        if (prev_buffer_idx >= 0) {
            printf("  Generating %d transition frames...\n", FRAMES_PER_TRANSITION);
            
            // Get current filename for overlay
            const char* current_filename = g_image_files[img_data.index].filename.c_str();
            
            for (int f = 0; f < FRAMES_PER_TRANSITION; f++) {
                float progress = (float)(f + 1) / FRAMES_PER_TRANSITION;
                
                // Wipe transition
                wipeTransitionKernel<<<grid_2d_output, block_2d, 0, stream>>>(
                    d_resized1, d_resized2, d_output,
                    OUTPUT_WIDTH, OUTPUT_HEIGHT, progress
                );
                
                // Render filename overlay on the output frame
                renderFilenameOverlay(d_output, OUTPUT_WIDTH, OUTPUT_HEIGHT, 
                                     current_filename, stream);
                
                // Download and encode
                CUDA_CHECK(cudaMemcpyAsync(h_output, d_output, output_rgb_size,
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                
                encoder.writeFrame(h_output);
                total_frames_written++;
            }
        } else {
            // First image - render with filename overlay
            const char* first_filename = g_image_files[img_data.index].filename.c_str();
            
            // Copy to output buffer for overlay
            CUDA_CHECK(cudaMemcpy(d_output, d_resized2, output_rgb_size, cudaMemcpyDeviceToDevice));
            renderFilenameOverlay(d_output, OUTPUT_WIDTH, OUTPUT_HEIGHT,
                                 first_filename, stream);
            
            CUDA_CHECK(cudaMemcpy(h_output, d_output, output_rgb_size, cudaMemcpyDeviceToHost));
            encoder.writeFrame(h_output);
            total_frames_written++;
        }
        
        // Swap buffers: current becomes previous
        std::swap(d_resized1, d_resized2);
        prev_buffer_idx = buf_idx;
        current_buffer_idx++;
        images_processed++;
        
        // Free host memory
        free(img_data.host_data);
        
        printf("  Total frames written: %d (%.1f seconds)\n", 
               total_frames_written, (float)total_frames_written / OUTPUT_FPS);
    }
    
    // ========================================================================
    // Phase 7: Cleanup
    // ========================================================================
    printf("\n=== Phase 6: Finalizing ===\n");
    
    // Wait for loader threads
    for (int i = 0; i < NUM_LOADER_THREADS; i++) {
        pthread_join(loader_threads[i], NULL);
    }
    
    g_loading_complete.store(true);
    g_consumer_cv.notify_all();


    // Finalize video
    encoder.finish();
    
    // Free GPU memory
    for (int i = 0; i < GPU_PREFETCH_BUFFER; i++) {
        CUDA_CHECK(cudaFree(gpu_buffers[i].d_float_data));
        CUDA_CHECK(cudaFree(gpu_buffers[i].d_rgb_data));
    }
    CUDA_CHECK(cudaFree(d_resized1));
    CUDA_CHECK(cudaFree(d_resized2));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaStreamDestroy(stream));
    free(h_output);
    
    printf("\n=== Complete ===\n");
    printf("Output: %s\n", output_file);
    printf("Total frames: %d\n", total_frames_written);
    printf("Duration: %.2f seconds\n", (float)total_frames_written / OUTPUT_FPS);
    printf("Images processed: %zu\n", g_image_files.size());
    
    return 0;
}

