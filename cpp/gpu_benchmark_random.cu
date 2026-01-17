/* gpu_random_benchmark.cu - Benchmark random access patterns
   Measures GPU and RAM disk performance with non-sequential access */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <pthread.h>
#include <unistd.h>
#include <algorithm>

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// Test configuration
#define MIN_TEST_TIME 2.0
#define GPU_TEST_SIZE (1024*1024*1024)  // 1GB
#define RAM_TEST_SIZE (512*1024*1024)   // 512MB
#define MAX_THREADS 32

// Structure to hold benchmark results
typedef struct {
    // GPU sequential results
    double gpu_seq_write_bandwidth;
    double gpu_seq_read_bandwidth;
    int gpu_seq_write_streams;
    int gpu_seq_read_streams;
    
    // GPU random results
    double gpu_rand_write_bandwidth;
    double gpu_rand_read_bandwidth;
    int gpu_rand_write_streams;
    int gpu_rand_read_streams;
    
    // RAM sequential results
    double ram_seq_write_speed;
    double ram_seq_read_speed;
    int ram_seq_write_threads;
    int ram_seq_read_threads;
    
    // RAM random results
    double ram_rand_write_speed;
    double ram_rand_read_speed;
    int ram_rand_write_threads;
    int ram_rand_read_threads;
} benchmark_results;

benchmark_results g_results;

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

// GPU kernels for sequential access
__global__ void sequential_write_kernel(float* data, size_t n, float value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        data[idx] = value;
    }
}

__global__ void sequential_read_kernel(float* data, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        output[idx] = data[idx];
    }
}

// GPU kernels for random access
__global__ void random_write_kernel(float* data, size_t* indices, size_t n, float value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        data[indices[idx]] = value;
    }
}

__global__ void random_read_kernel(float* data, size_t* indices, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        output[idx] = data[indices[idx]];
    }
}

// Generate random indices for random access pattern
void generate_random_indices(size_t* indices, size_t n) {
    // Create sequential indices first
    for(size_t i = 0; i < n; i++) {
        indices[i] = i;
    }
    
    // Fisher-Yates shuffle for truly random permutation
    for(size_t i = n - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        size_t temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}

// Benchmark GPU with varying stream counts
void benchmark_gpu_streams(float** d_arrays, float** d_temp_arrays, size_t** d_indices,
                          size_t chunk_floats, int max_streams, 
                          double* best_bandwidth, int* best_streams,
                          int is_write, int is_random, const char* label) {
    
    int blockSize = 256;
    int gridSize = (chunk_floats + blockSize - 1) / blockSize;
    
    printf("\nTesting GPU %s %s:\n", is_random ? "RANDOM" : "SEQUENTIAL", label);
    printf("%-10s %-15s %-15s\n", "Streams", "Time(s)", "Bandwidth(GB/s)");
    printf("------------------------------------------\n");
    
    *best_bandwidth = 0;
    *best_streams = 1;
    
    for(int num_streams = 1; num_streams <= max_streams; num_streams *= 2) {
        cudaStream_t streams[32];
        for(int i = 0; i < num_streams; i++) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
        }
        
        double start = get_time();
        
        if(is_random) {
            if(is_write) {
                for(int i = 0; i < num_streams; i++) {
                    random_write_kernel<<<gridSize, blockSize, 0, streams[i]>>>(
                        d_arrays[i], d_indices[i], chunk_floats, (float)i);
                }
            } else {
                for(int i = 0; i < num_streams; i++) {
                    random_read_kernel<<<gridSize, blockSize, 0, streams[i]>>>(
                        d_arrays[i], d_indices[i], d_temp_arrays[i], chunk_floats);
                }
            }
        } else {
            if(is_write) {
                for(int i = 0; i < num_streams; i++) {
                    sequential_write_kernel<<<gridSize, blockSize, 0, streams[i]>>>(
                        d_arrays[i], chunk_floats, (float)i);
                }
            } else {
                for(int i = 0; i < num_streams; i++) {
                    sequential_read_kernel<<<gridSize, blockSize, 0, streams[i]>>>(
                        d_arrays[i], d_temp_arrays[i], chunk_floats);
                }
            }
        }
        
        for(int i = 0; i < num_streams; i++) {
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        }
        
        double end = get_time();
        double elapsed = end - start;
        
        size_t total_bytes = (size_t)num_streams * chunk_floats * sizeof(float);
        double bandwidth = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / elapsed;
        
        printf("%-10d %-15.6f %-15.1f\n", num_streams, elapsed, bandwidth);
        
        if(bandwidth > *best_bandwidth) {
            *best_bandwidth = bandwidth;
            *best_streams = num_streams;
        }
        
        for(int i = 0; i < num_streams; i++) {
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
        }
        
        if(num_streams > 1 && bandwidth < *best_bandwidth * 0.85) {
            break;
        }
    }
    
    printf("Best: %d streams @ %.1f GB/s\n", *best_streams, *best_bandwidth);
}

void benchmark_gpu_memory() {
    printf("\n========================================\n");
    printf("Benchmarking: GPU Memory\n");
    printf("========================================\n");
    
    size_t num_floats = GPU_TEST_SIZE / sizeof(float);
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int max_streams = (prop.asyncEngineCount > 0) ? 32 : 1;
    
    printf("GPU: %s\n", prop.name);
    printf("Max concurrent streams: %d\n", max_streams);
    
    // Allocate arrays for stream testing
    size_t chunk_floats = num_floats / max_streams;
    float* d_arrays[32];
    float* d_temp_arrays[32];
    size_t* d_indices[32];
    
    for(int i = 0; i < max_streams; i++) {
        CUDA_CHECK(cudaMalloc(&d_arrays[i], chunk_floats * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_temp_arrays[i], chunk_floats * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_indices[i], chunk_floats * sizeof(size_t)));
    }
    
    // Generate random indices on host, then copy to device
    size_t* h_indices = (size_t*)malloc(chunk_floats * sizeof(size_t));
    generate_random_indices(h_indices, chunk_floats);
    
    for(int i = 0; i < max_streams; i++) {
        CUDA_CHECK(cudaMemcpy(d_indices[i], h_indices, 
                             chunk_floats * sizeof(size_t),
                             cudaMemcpyHostToDevice));
    }
    free(h_indices);
    
    // Sequential write
    double seq_write_bw;
    int seq_write_streams;
    benchmark_gpu_streams(d_arrays, d_temp_arrays, d_indices, chunk_floats, max_streams,
                         &seq_write_bw, &seq_write_streams, 1, 0, "WRITE");
    
    // Sequential read
    double seq_read_bw;
    int seq_read_streams;
    benchmark_gpu_streams(d_arrays, d_temp_arrays, d_indices, chunk_floats, max_streams,
                         &seq_read_bw, &seq_read_streams, 0, 0, "READ");
    
    // Random write
    double rand_write_bw;
    int rand_write_streams;
    benchmark_gpu_streams(d_arrays, d_temp_arrays, d_indices, chunk_floats, max_streams,
                         &rand_write_bw, &rand_write_streams, 1, 1, "WRITE");
    
    // Random read
    double rand_read_bw;
    int rand_read_streams;
    benchmark_gpu_streams(d_arrays, d_temp_arrays, d_indices, chunk_floats, max_streams,
                         &rand_read_bw, &rand_read_streams, 0, 1, "READ");
    
    // Store results
    g_results.gpu_seq_write_bandwidth = seq_write_bw;
    g_results.gpu_seq_read_bandwidth = seq_read_bw;
    g_results.gpu_seq_write_streams = seq_write_streams;
    g_results.gpu_seq_read_streams = seq_read_streams;
    g_results.gpu_rand_write_bandwidth = rand_write_bw;
    g_results.gpu_rand_read_bandwidth = rand_read_bw;
    g_results.gpu_rand_write_streams = rand_write_streams;
    g_results.gpu_rand_read_streams = rand_read_streams;
    
    // Cleanup
    for(int i = 0; i < max_streams; i++) {
        CUDA_CHECK(cudaFree(d_arrays[i]));
        CUDA_CHECK(cudaFree(d_temp_arrays[i]));
        CUDA_CHECK(cudaFree(d_indices[i]));
    }
}

// Thread function for sequential write
typedef struct {
    char* filename;
    size_t offset;
    size_t size;
    int thread_id;
} seq_thread_args;

void* seq_write_thread(void* args) {
    seq_thread_args* a = (seq_thread_args*)args;
    char* buffer = (char*)malloc(a->size);
    memset(buffer, 0xAB, a->size);
    
    FILE* f = fopen(a->filename, "r+b");
    if(!f) {
        fprintf(stderr, "Failed to open file for sequential write\n");
        free(buffer);
        return NULL;
    }
    
    fseek(f, a->offset, SEEK_SET);
    fwrite(buffer, 1, a->size, f);
    fclose(f);
    free(buffer);
    return NULL;
}

void* seq_read_thread(void* args) {
    seq_thread_args* a = (seq_thread_args*)args;
    char* buffer = (char*)malloc(a->size);
    
    FILE* f = fopen(a->filename, "rb");
    if(!f) {
        fprintf(stderr, "Failed to open file for sequential read\n");
        free(buffer);
        return NULL;
    }
    
    fseek(f, a->offset, SEEK_SET);
    fread(buffer, 1, a->size, f);
    fclose(f);
    free(buffer);
    return NULL;
}

// Thread function for random write
typedef struct {
    char* filename;
    size_t* offsets;
    size_t num_accesses;
    size_t access_size;
    int thread_id;
} rand_thread_args;

void* rand_write_thread(void* args) {
    rand_thread_args* a = (rand_thread_args*)args;
    char* buffer = (char*)malloc(a->access_size);
    memset(buffer, 0xCD, a->access_size);
    
    FILE* f = fopen(a->filename, "r+b");
    if(!f) {
        fprintf(stderr, "Failed to open file for random write\n");
        free(buffer);
        return NULL;
    }
    
    for(size_t i = 0; i < a->num_accesses; i++) {
        fseek(f, a->offsets[i], SEEK_SET);
        fwrite(buffer, 1, a->access_size, f);
    }
    
    fclose(f);
    free(buffer);
    return NULL;
}

void* rand_read_thread(void* args) {
    rand_thread_args* a = (rand_thread_args*)args;
    char* buffer = (char*)malloc(a->access_size);
    
    FILE* f = fopen(a->filename, "rb");
    if(!f) {
        fprintf(stderr, "Failed to open file for random read\n");
        free(buffer);
        return NULL;
    }
    
    for(size_t i = 0; i < a->num_accesses; i++) {
        fseek(f, a->offsets[i], SEEK_SET);
        fread(buffer, 1, a->access_size, f);
    }
    
    fclose(f);
    free(buffer);
    return NULL;
}

void benchmark_ramdisk() {
    printf("\n========================================\n");
    printf("Benchmarking: RAM Disk (/ram/)\n");
    printf("========================================\n");
    
    const char* filename = "/ram/benchmark.tmp";
    size_t file_size = RAM_TEST_SIZE;
    size_t access_size = 4096;  // 4KB per random access
    
    // Create test file
    printf("Creating test file...\n");
    FILE* f = fopen(filename, "wb");
    if(!f) {
        fprintf(stderr, "Failed to create test file\n");
        return;
    }
    char* zeros = (char*)calloc(1, file_size);
    fwrite(zeros, 1, file_size, f);
    fclose(f);
    free(zeros);
    
    // Sequential write test
    printf("\nTesting SEQUENTIAL WRITE:\n");
    printf("%-10s %-15s %-15s\n", "Threads", "Time(s)", "Speed(MB/s)");
    printf("------------------------------------------\n");
    
    double best_seq_write_speed = 0;
    int best_seq_write_threads = 1;
    
    for(int num_threads = 1; num_threads <= MAX_THREADS; num_threads *= 2) {
        pthread_t threads[MAX_THREADS];
        seq_thread_args args[MAX_THREADS];
        size_t chunk_size = file_size / num_threads;
        
        double start = get_time();
        
        for(int t = 0; t < num_threads; t++) {
            args[t].filename = (char*)filename;
            args[t].offset = t * chunk_size;
            args[t].size = chunk_size;
            args[t].thread_id = t;
            pthread_create(&threads[t], NULL, seq_write_thread, &args[t]);
        }
        
        for(int t = 0; t < num_threads; t++) {
            pthread_join(threads[t], NULL);
        }
        
        double elapsed = get_time() - start;
        double speed = (file_size / (1024.0 * 1024.0)) / elapsed;
        
        printf("%-10d %-15.3f %-15.1f\n", num_threads, elapsed, speed);
        
        if(speed > best_seq_write_speed) {
            best_seq_write_speed = speed;
            best_seq_write_threads = num_threads;
        }
        
        if(num_threads > 1 && speed < best_seq_write_speed * 0.8) {
            break;
        }
    }
    
    // Sequential read test
    printf("\nTesting SEQUENTIAL READ:\n");
    printf("%-10s %-15s %-15s\n", "Threads", "Time(s)", "Speed(MB/s)");
    printf("------------------------------------------\n");
    
    double best_seq_read_speed = 0;
    int best_seq_read_threads = 1;
    
    for(int num_threads = 1; num_threads <= MAX_THREADS; num_threads *= 2) {
        pthread_t threads[MAX_THREADS];
        seq_thread_args args[MAX_THREADS];
        size_t chunk_size = file_size / num_threads;
        
        double start = get_time();
        
        for(int t = 0; t < num_threads; t++) {
            args[t].filename = (char*)filename;
            args[t].offset = t * chunk_size;
            args[t].size = chunk_size;
            args[t].thread_id = t;
            pthread_create(&threads[t], NULL, seq_read_thread, &args[t]);
        }
        
        for(int t = 0; t < num_threads; t++) {
            pthread_join(threads[t], NULL);
        }
        
        double elapsed = get_time() - start;
        double speed = (file_size / (1024.0 * 1024.0)) / elapsed;
        
        printf("%-10d %-15.3f %-15.1f\n", num_threads, elapsed, speed);
        
        if(speed > best_seq_read_speed) {
            best_seq_read_speed = speed;
            best_seq_read_threads = num_threads;
        }
        
        if(num_threads > 1 && speed < best_seq_read_speed * 0.8) {
            break;
        }
    }
    
    // Generate random offsets
    size_t num_random_accesses = file_size / access_size;
    size_t* random_offsets = (size_t*)malloc(num_random_accesses * sizeof(size_t));
    for(size_t i = 0; i < num_random_accesses; i++) {
        random_offsets[i] = (rand() % (file_size / access_size)) * access_size;
    }
    
    // Random write test
    printf("\nTesting RANDOM WRITE (4KB accesses):\n");
    printf("%-10s %-15s %-15s\n", "Threads", "Time(s)", "Speed(MB/s)");
    printf("------------------------------------------\n");
    
    double best_rand_write_speed = 0;
    int best_rand_write_threads = 1;
    
    for(int num_threads = 1; num_threads <= MAX_THREADS; num_threads *= 2) {
        pthread_t threads[MAX_THREADS];
        rand_thread_args args[MAX_THREADS];
        size_t accesses_per_thread = num_random_accesses / num_threads;
        
        double start = get_time();
        
        for(int t = 0; t < num_threads; t++) {
            args[t].filename = (char*)filename;
            args[t].offsets = &random_offsets[t * accesses_per_thread];
            args[t].num_accesses = accesses_per_thread;
            args[t].access_size = access_size;
            args[t].thread_id = t;
            pthread_create(&threads[t], NULL, rand_write_thread, &args[t]);
        }
        
        for(int t = 0; t < num_threads; t++) {
            pthread_join(threads[t], NULL);
        }
        
        double elapsed = get_time() - start;
        size_t total_bytes = num_threads * accesses_per_thread * access_size;
        double speed = (total_bytes / (1024.0 * 1024.0)) / elapsed;
        
        printf("%-10d %-15.3f %-15.1f\n", num_threads, elapsed, speed);
        
        if(speed > best_rand_write_speed) {
            best_rand_write_speed = speed;
            best_rand_write_threads = num_threads;
        }
        
        if(num_threads > 1 && speed < best_rand_write_speed * 0.8) {
            break;
        }
    }
    
    // Random read test
    printf("\nTesting RANDOM READ (4KB accesses):\n");
    printf("%-10s %-15s %-15s\n", "Threads", "Time(s)", "Speed(MB/s)");
    printf("------------------------------------------\n");
    
    double best_rand_read_speed = 0;
    int best_rand_read_threads = 1;
    
    for(int num_threads = 1; num_threads <= MAX_THREADS; num_threads *= 2) {
        pthread_t threads[MAX_THREADS];
        rand_thread_args args[MAX_THREADS];
        size_t accesses_per_thread = num_random_accesses / num_threads;
        
        double start = get_time();
        
        for(int t = 0; t < num_threads; t++) {
            args[t].filename = (char*)filename;
            args[t].offsets = &random_offsets[t * accesses_per_thread];
            args[t].num_accesses = accesses_per_thread;
            args[t].access_size = access_size;
            args[t].thread_id = t;
            pthread_create(&threads[t], NULL, rand_read_thread, &args[t]);
        }
        
        for(int t = 0; t < num_threads; t++) {
            pthread_join(threads[t], NULL);
        }
        
        double elapsed = get_time() - start;
        size_t total_bytes = num_threads * accesses_per_thread * access_size;
        double speed = (total_bytes / (1024.0 * 1024.0)) / elapsed;
        
        printf("%-10d %-15.3f %-15.1f\n", num_threads, elapsed, speed);
        
        if(speed > best_rand_read_speed) {
            best_rand_read_speed = speed;
            best_rand_read_threads = num_threads;
        }
        
        if(num_threads > 1 && speed < best_rand_read_speed * 0.8) {
            break;
        }
    }
    
    // Store results
    g_results.ram_seq_write_speed = best_seq_write_speed;
    g_results.ram_seq_read_speed = best_seq_read_speed;
    g_results.ram_seq_write_threads = best_seq_write_threads;
    g_results.ram_seq_read_threads = best_seq_read_threads;
    g_results.ram_rand_write_speed = best_rand_write_speed;
    g_results.ram_rand_read_speed = best_rand_read_speed;
    g_results.ram_rand_write_threads = best_rand_write_threads;
    g_results.ram_rand_read_threads = best_rand_read_threads;
    
    // Cleanup
    free(random_offsets);
    unlink(filename);
}

void print_summary_table() {
    printf("\n\n");
    printf("================================================================================\n");
    printf("                    SEQUENTIAL vs RANDOM ACCESS COMPARISON                     \n");
    printf("================================================================================\n\n");
    
    printf("┌───────────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│ GPU MEMORY PERFORMANCE                                                                │\n");
    printf("├─────────────────────┬─────────────┬─────────────┬──────────────┬───────────────────┤\n");
    printf("│ Access Pattern      │ Operation   │ Bandwidth   │ Streams      │ Slowdown Factor   │\n");
    printf("├─────────────────────┼─────────────┼─────────────┼──────────────┼───────────────────┤\n");
    printf("│ Sequential          │ Write       │ %8.1f GB/s│ %5d        │ baseline          │\n",
           g_results.gpu_seq_write_bandwidth, g_results.gpu_seq_write_streams);
    printf("│ Random              │ Write       │ %8.1f GB/s│ %5d        │ %.1fx slower       │\n",
           g_results.gpu_rand_write_bandwidth, g_results.gpu_rand_write_streams,
           g_results.gpu_seq_write_bandwidth / g_results.gpu_rand_write_bandwidth);
    printf("│ Sequential          │ Read        │ %8.1f GB/s│ %5d        │ baseline          │\n",
           g_results.gpu_seq_read_bandwidth, g_results.gpu_seq_read_streams);
    printf("│ Random              │ Read        │ %8.1f GB/s│ %5d        │ %.1fx slower       │\n",
           g_results.gpu_rand_read_bandwidth, g_results.gpu_rand_read_streams,
           g_results.gpu_seq_read_bandwidth / g_results.gpu_rand_read_bandwidth);
    printf("└─────────────────────┴─────────────┴─────────────┴──────────────┴───────────────────┘\n\n");
    
    printf("┌───────────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│ RAMDISK PERFORMANCE (/ram/)                                                           │\n");
    printf("├─────────────────────┬─────────────┬─────────────┬──────────────┬───────────────────┤\n");
    printf("│ Access Pattern      │ Operation   │ Speed       │ Threads      │ Slowdown Factor   │\n");
    printf("├─────────────────────┼─────────────┼─────────────┼──────────────┼───────────────────┤\n");
    printf("│ Sequential          │ Write       │ %8.1f MB/s│ %5d        │ baseline          │\n",
           g_results.ram_seq_write_speed, g_results.ram_seq_write_threads);
    printf("│ Random (4KB)        │ Write       │ %8.1f MB/s│ %5d        │ %.1fx slower       │\n",
           g_results.ram_rand_write_speed, g_results.ram_rand_write_threads,
           g_results.ram_seq_write_speed / g_results.ram_rand_write_speed);
    printf("│ Sequential          │ Read        │ %8.1f MB/s│ %5d        │ baseline          │\n",
           g_results.ram_seq_read_speed, g_results.ram_seq_read_threads);
    printf("│ Random (4KB)        │ Read        │ %8.1f MB/s│ %5d        │ %.1fx slower       │\n",
           g_results.ram_rand_read_speed, g_results.ram_rand_read_threads,
           g_results.ram_seq_read_speed / g_results.ram_rand_read_speed);
    printf("└─────────────────────┴─────────────┴─────────────┴──────────────┴───────────────────┘\n\n");
    
    printf("┌───────────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│ KEY INSIGHTS                                                                          │\n");
    printf("├───────────────────────────────────────────────────────────────────────────────────────┤\n");
    printf("│ GPU Memory:                                                                           │\n");
    printf("│   • Sequential access achieves %.0f%% of peak bandwidth                                │\n",
           100.0 * g_results.gpu_seq_write_bandwidth / g_results.gpu_seq_write_bandwidth);
    printf("│   • Random access degradation: ~%.0fx slower                                          │\n",
           g_results.gpu_seq_write_bandwidth / g_results.gpu_rand_write_bandwidth);
    printf("│   • Random access may benefit from more streams (%.0fx increase in optimal streams)   │\n",
           (float)g_results.gpu_rand_write_streams / g_results.gpu_seq_write_streams);
    printf("│                                                                                       │\n");
    printf("│ RAM Disk:                                                                             │\n");
    printf("│   • Sequential access: %.0f MB/s                                                      │\n",
           (g_results.ram_seq_write_speed + g_results.ram_seq_read_speed) / 2);
    printf("│   • Random access (4KB): ~%.0fx slower                                                │\n",
           (g_results.ram_seq_read_speed / g_results.ram_rand_read_speed +
            g_results.ram_seq_write_speed / g_results.ram_rand_write_speed) / 2);
    printf("│   • Random access limited by seek/syscall overhead                                   │\n");
    printf("│                                                                                       │\n");
    printf("│ Recommendations:                                                                      │\n");
    printf("│   • Always try to use sequential access patterns when possible                       │\n");
    printf("│   • If random access is necessary, increase parallelism (more streams/threads)       │\n");
    printf("│   • GPU: Random access can benefit from larger stream counts                         │\n");
    printf("│   • RAM: Consider larger block sizes for random I/O to reduce overhead               │\n");
    printf("└───────────────────────────────────────────────────────────────────────────────────────┘\n");
}

int main(int argc, char** argv) {
    srand(time(NULL));
    memset(&g_results, 0, sizeof(benchmark_results));
    
    printf("GPU and RAM Disk Random Access Benchmark\n");
    printf("=========================================\n\n");
    
    // Check CUDA device
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if(deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("GPU Memory: %.1f GB\n\n", prop.totalGlobalMem / (1024.0*1024.0*1024.0));
    
    // Run benchmarks
    benchmark_gpu_memory();
    benchmark_ramdisk();
    
    // Print summary
    print_summary_table();
    
    printf("\n========================================\n");
    printf("Benchmark Complete\n");
    printf("========================================\n");
    
    return 0;
}

