/** 20260128 

  nvcc -O3 -arch=sm_89 sentinel2_mp4.cu -o sentinel2_mp4 -lpthread -lavformat -lavcodec -lavutil -lswscale

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
#define OUTPUT_WIDTH 1920
#define OUTPUT_HEIGHT 1080
#define OUTPUT_FPS 60
#define TRANSITION_DURATION_SEC 0.5f
#define FRAMES_PER_TRANSITION ((int)(OUTPUT_FPS * TRANSITION_DURATION_SEC))
#define HISTOGRAM_TRIM_PERCENT 2.0f
#define MAX_BANDS_USED 3

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
// Histogram Trimming (CPU - for computing bounds)
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
            codec_ctx->bit_rate = 8000000;  // 8 Mbps
        } else {
            av_opt_set(codec_ctx->priv_data, "preset", "medium", 0);
            av_opt_set(codec_ctx->priv_data, "crf", "23", 0);
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
    // Phase 1: Compute histogram bounds from all images
    // ========================================================================
    printf("=== Phase 1: Computing histogram bounds ===\n");
    
    std::vector<BandStats> all_stats(g_image_files.size() * MAX_BANDS_USED);
    
    #pragma omp parallel for num_threads(NUM_LOADER_THREADS)
    for (size_t i = 0; i < g_image_files.size(); i++) {
        ENVIHeader header;
        float* data = readENVIFile(g_image_files[i], header);
        if (data) {
            int bands = std::min(header.bands, MAX_BANDS_USED);
            computeHistogramBounds(data, header.samples, header.lines, bands,
                                  &all_stats[i * MAX_BANDS_USED], HISTOGRAM_TRIM_PERCENT);
            printf("  Image %zu bounds: R[%.3f, %.3f] G[%.3f, %.3f] B[%.3f, %.3f]\n",
                   i, all_stats[i*MAX_BANDS_USED].min_val, all_stats[i*MAX_BANDS_USED].max_val,
                   bands >= 2 ? all_stats[i*MAX_BANDS_USED+1].min_val : 0,
                   bands >= 2 ? all_stats[i*MAX_BANDS_USED+1].max_val : 0,
                   bands >= 3 ? all_stats[i*MAX_BANDS_USED+2].min_val : 0,
                   bands >= 3 ? all_stats[i*MAX_BANDS_USED+2].max_val : 0);
            free(data);
        }
    }
    
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
    // Phase 2: Setup GPU and CUDA resources
    // ========================================================================
    printf("=== Phase 2: Initializing GPU ===\n");
    
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
    // Phase 3: Start loader threads
    // ========================================================================
    printf("\n=== Phase 3: Starting video generation ===\n");
    
    pthread_t loader_threads[NUM_LOADER_THREADS];
    LoaderThreadData thread_data[NUM_LOADER_THREADS];
    
    for (int i = 0; i < NUM_LOADER_THREADS; i++) {
        thread_data[i].thread_id = i;
        pthread_create(&loader_threads[i], NULL, loaderThreadFunc, &thread_data[i]);
    }
    
    // ========================================================================
    // Phase 4: Initialize video encoder
    // ========================================================================
    VideoEncoder encoder;
    if (!encoder.init(output_file, OUTPUT_WIDTH, OUTPUT_HEIGHT, OUTPUT_FPS)) {
        fprintf(stderr, "Failed to initialize video encoder\n");
        return 1;
    }
    
    // ========================================================================
    // Phase 5: Main processing loop
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
        ImageData img_data;
        {
            std::unique_lock<std::mutex> lock(g_queue_mutex);
            g_consumer_cv.wait(lock, []{ return !g_image_queue.empty(); });
            img_data = g_image_queue.front();
            g_image_queue.pop();
            g_producer_cv.notify_one();
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
            
            for (int f = 0; f < FRAMES_PER_TRANSITION; f++) {
                float progress = (float)(f + 1) / FRAMES_PER_TRANSITION;
                
                // Wipe transition
                wipeTransitionKernel<<<grid_2d_output, block_2d, 0, stream>>>(
                    d_resized1, d_resized2, d_output,
                    OUTPUT_WIDTH, OUTPUT_HEIGHT, progress
                );
                
                // Download and encode
                CUDA_CHECK(cudaMemcpyAsync(h_output, d_output, output_rgb_size,
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                
                encoder.writeFrame(h_output);
                total_frames_written++;
            }
        } else {
            // First image - just write one frame to start
            CUDA_CHECK(cudaMemcpy(h_output, d_resized2, output_rgb_size, cudaMemcpyDeviceToHost));
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
    // Phase 6: Cleanup
    // ========================================================================
    printf("\n=== Phase 6: Finalizing ===\n");
    
    // Wait for loader threads
    for (int i = 0; i < NUM_LOADER_THREADS; i++) {
        pthread_join(loader_threads[i], NULL);
    }
    
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



