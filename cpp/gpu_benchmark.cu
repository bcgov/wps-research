/*20160116 

   nvcc -O3 gpu_benchmark.cu -o gpu_benchmark -lpthread
./gpu_benchmark
   
   gpu_benchmark.cu - Benchmark GPU memory, RAM disk, and filesystem I/O
   Measures data transfer rates and optimal channel capacity */

/* gpu_benchmark.cu - Benchmark GPU memory, RAM disk, and filesystem I/O
   Measures data transfer rates and optimal channel capacity */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <pthread.h>
#include <unistd.h>

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// Test configuration
#define MIN_TEST_TIME 2.0  // Minimum seconds per test
#define GPU_TEST_SIZE (1024*1024*1024)  // 1GB for GPU
#define DISK_TEST_SIZE (512*1024*1024)  // 512MB for disk tests
#define MAX_THREADS 32

// Structure to hold benchmark results
typedef struct {
    // GPU results
    double gpu_write_bandwidth;
    double gpu_read_bandwidth;
    double gpu_h2d_bandwidth;
    double gpu_d2h_bandwidth;
    int gpu_write_iterations;
    int gpu_read_iterations;
    int gpu_h2d_iterations;
    int gpu_d2h_iterations;
    
    // /ram/ results
    double ram_write_speed;
    int ram_write_threads;
    double ram_read_speed;
    int ram_read_threads;
    
    // /data/ results
    double data_write_speed;
    int data_write_threads;
    double data_read_speed;
    int data_read_threads;
} benchmark_results;

benchmark_results g_results;

// Global variables for threaded I/O
typedef struct {
    char* filename;
    size_t chunk_size;
    size_t num_chunks;
    int thread_id;
    double* elapsed_time;
    pthread_mutex_t* mutex;
    size_t* completed_chunks;
} io_thread_args;

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

// Write thread function
void* write_thread_func(void* args) {
    io_thread_args* a = (io_thread_args*)args;
    char* buffer = (char*)malloc(a->chunk_size);
    memset(buffer, 0xAB, a->chunk_size);  // Fill with pattern
    
    char fname[256];
    sprintf(fname, "%s.%d", a->filename, a->thread_id);
    
    double start = get_time();
    FILE* f = fopen(fname, "wb");
    if(!f) {
        fprintf(stderr, "Failed to open %s for writing\n", fname);
        free(buffer);
        return NULL;
    }
    
    for(size_t i = 0; i < a->num_chunks; i++) {
        size_t written = fwrite(buffer, 1, a->chunk_size, f);
        if(written != a->chunk_size) {
            fprintf(stderr, "Write failed in thread %d\n", a->thread_id);
            break;
        }
        pthread_mutex_lock(a->mutex);
        (*a->completed_chunks)++;
        pthread_mutex_unlock(a->mutex);
    }
    
    fclose(f);
    double end = get_time();
    
    pthread_mutex_lock(a->mutex);
    *a->elapsed_time += (end - start);
    pthread_mutex_unlock(a->mutex);
    
    free(buffer);
    return NULL;
}

// Read thread function
void* read_thread_func(void* args) {
    io_thread_args* a = (io_thread_args*)args;
    char* buffer = (char*)malloc(a->chunk_size);
    
    char fname[256];
    sprintf(fname, "%s.%d", a->filename, a->thread_id);
    
    double start = get_time();
    FILE* f = fopen(fname, "rb");
    if(!f) {
        fprintf(stderr, "Failed to open %s for reading\n", fname);
        free(buffer);
        return NULL;
    }
    
    for(size_t i = 0; i < a->num_chunks; i++) {
        size_t bytes_read = fread(buffer, 1, a->chunk_size, f);
        if(bytes_read != a->chunk_size && !feof(f)) {
            fprintf(stderr, "Read failed in thread %d\n", a->thread_id);
            break;
        }
        pthread_mutex_lock(a->mutex);
        (*a->completed_chunks)++;
        pthread_mutex_unlock(a->mutex);
    }
    
    fclose(f);
    double end = get_time();
    
    pthread_mutex_lock(a->mutex);
    *a->elapsed_time += (end - start);
    pthread_mutex_unlock(a->mutex);
    
    free(buffer);
    return NULL;
}

// Benchmark disk I/O with varying thread counts
void benchmark_disk_io(const char* path, int is_ram) {
    printf("\n========================================\n");
    printf("Benchmarking: %s\n", path);
    printf("========================================\n");
    
    char filename[256];
    sprintf(filename, "%s/tmp.tmp", path);
    
    size_t chunk_size = 4 * 1024 * 1024;  // 4MB chunks
    size_t chunks_per_thread = DISK_TEST_SIZE / chunk_size / 4;  // Divided among threads
    
    int best_write_threads = 1;
    double best_write_speed = 0;
    int best_read_threads = 1;
    double best_read_speed = 0;
    
    printf("\nTesting WRITE performance:\n");
    printf("%-10s %-15s %-15s %-15s\n", "Threads", "Time(s)", "Speed(MB/s)", "Total(MB)");
    printf("----------------------------------------------------------\n");
    
    for(int num_threads = 1; num_threads <= MAX_THREADS; num_threads *= 2) {
        pthread_t threads[MAX_THREADS];
        io_thread_args thread_args[MAX_THREADS];
        pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
        double total_time = 0;
        size_t completed_chunks = 0;
        
        // Launch write threads
        for(int t = 0; t < num_threads; t++) {
            thread_args[t].filename = filename;
            thread_args[t].chunk_size = chunk_size;
            thread_args[t].num_chunks = chunks_per_thread;
            thread_args[t].thread_id = t;
            thread_args[t].elapsed_time = &total_time;
            thread_args[t].mutex = &mutex;
            thread_args[t].completed_chunks = &completed_chunks;
            pthread_create(&threads[t], NULL, write_thread_func, &thread_args[t]);
        }
        
        // Wait for completion
        for(int t = 0; t < num_threads; t++) {
            pthread_join(threads[t], NULL);
        }
        
        size_t total_bytes = (size_t)num_threads * chunks_per_thread * chunk_size;
        double avg_time = total_time / num_threads;
        double speed_mbps = (total_bytes / (1024.0 * 1024.0)) / avg_time;
        
        printf("%-10d %-15.3f %-15.1f %-15.1f\n", 
               num_threads, avg_time, speed_mbps, total_bytes / (1024.0 * 1024.0));
        
        if(speed_mbps > best_write_speed) {
            best_write_speed = speed_mbps;
            best_write_threads = num_threads;
        }
        
        // Stop if performance degrades significantly
        if(num_threads > 1 && speed_mbps < best_write_speed * 0.8) {
            break;
        }
    }
    
    printf("\nTesting READ performance:\n");
    printf("%-10s %-15s %-15s %-15s\n", "Threads", "Time(s)", "Speed(MB/s)", "Total(MB)");
    printf("----------------------------------------------------------\n");
    
    for(int num_threads = 1; num_threads <= MAX_THREADS; num_threads *= 2) {
        pthread_t threads[MAX_THREADS];
        io_thread_args thread_args[MAX_THREADS];
        pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
        double total_time = 0;
        size_t completed_chunks = 0;
        
        // Launch read threads
        for(int t = 0; t < num_threads; t++) {
            thread_args[t].filename = filename;
            thread_args[t].chunk_size = chunk_size;
            thread_args[t].num_chunks = chunks_per_thread;
            thread_args[t].thread_id = t;
            thread_args[t].elapsed_time = &total_time;
            thread_args[t].mutex = &mutex;
            thread_args[t].completed_chunks = &completed_chunks;
            pthread_create(&threads[t], NULL, read_thread_func, &thread_args[t]);
        }
        
        // Wait for completion
        for(int t = 0; t < num_threads; t++) {
            pthread_join(threads[t], NULL);
        }
        
        size_t total_bytes = (size_t)num_threads * chunks_per_thread * chunk_size;
        double avg_time = total_time / num_threads;
        double speed_mbps = (total_bytes / (1024.0 * 1024.0)) / avg_time;
        
        printf("%-10d %-15.3f %-15.1f %-15.1f\n", 
               num_threads, avg_time, speed_mbps, total_bytes / (1024.0 * 1024.0));
        
        if(speed_mbps > best_read_speed) {
            best_read_speed = speed_mbps;
            best_read_threads = num_threads;
        }
        
        // Stop if performance degrades significantly
        if(num_threads > 1 && speed_mbps < best_read_speed * 0.8) {
            break;
        }
    }
    
    // Cleanup test files
    for(int t = 0; t < MAX_THREADS; t++) {
        char fname[256];
        sprintf(fname, "%s.%d", filename, t);
        unlink(fname);
    }
    
    printf("\n--- RESULTS for %s ---\n", path);
    printf("Optimal WRITE: %d threads @ %.1f MB/s\n", best_write_threads, best_write_speed);
    printf("Optimal READ:  %d threads @ %.1f MB/s\n", best_read_threads, best_read_speed);
    
    // Store results
    if(is_ram) {
        g_results.ram_write_speed = best_write_speed;
        g_results.ram_write_threads = best_write_threads;
        g_results.ram_read_speed = best_read_speed;
        g_results.ram_read_threads = best_read_threads;
    } else {
        g_results.data_write_speed = best_write_speed;
        g_results.data_write_threads = best_write_threads;
        g_results.data_read_speed = best_read_speed;
        g_results.data_read_threads = best_read_threads;
    }
}

// GPU memory benchmark kernel
__global__ void write_kernel(float* data, size_t n, float value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        data[idx] = value;
    }
}

__global__ void read_kernel(float* data, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        output[idx] = data[idx];
    }
}

void benchmark_gpu_memory() {
    printf("\n========================================\n");
    printf("Benchmarking: GPU Memory\n");
    printf("========================================\n");
    
    size_t num_floats = GPU_TEST_SIZE / sizeof(float);
    size_t bytes = num_floats * sizeof(float);
    
    float *d_data, *d_temp;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMalloc(&d_temp, bytes));
    
    int blockSize = 256;
    int gridSize = (num_floats + blockSize - 1) / blockSize;
    
    // Warm up
    write_kernel<<<gridSize, blockSize>>>(d_data, num_floats, 1.0f);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("\nTesting GPU WRITE (kernel write) performance:\n");
    printf("%-10s %-15s %-15s\n", "Iteration", "Time(s)", "Bandwidth(GB/s)");
    printf("------------------------------------------\n");
    
    double total_write_time = 0;
    int write_iterations = 0;
    
    while(total_write_time < MIN_TEST_TIME) {
        double start = get_time();
        write_kernel<<<gridSize, blockSize>>>(d_data, num_floats, (float)write_iterations);
        CUDA_CHECK(cudaDeviceSynchronize());
        double end = get_time();
        
        double elapsed = end - start;
        total_write_time += elapsed;
        write_iterations++;
        
        double bandwidth = (bytes / (1024.0 * 1024.0 * 1024.0)) / elapsed;
        printf("%-10d %-15.6f %-15.1f\n", write_iterations, elapsed, bandwidth);
    }
    
    double avg_write_time = total_write_time / write_iterations;
    double write_bandwidth = (bytes / (1024.0 * 1024.0 * 1024.0)) / avg_write_time;
    
    printf("\nTesting GPU READ (kernel read) performance:\n");
    printf("%-10s %-15s %-15s\n", "Iteration", "Time(s)", "Bandwidth(GB/s)");
    printf("------------------------------------------\n");
    
    double total_read_time = 0;
    int read_iterations = 0;
    
    while(total_read_time < MIN_TEST_TIME) {
        double start = get_time();
        read_kernel<<<gridSize, blockSize>>>(d_data, d_temp, num_floats);
        CUDA_CHECK(cudaDeviceSynchronize());
        double end = get_time();
        
        double elapsed = end - start;
        total_read_time += elapsed;
        read_iterations++;
        
        double bandwidth = (bytes / (1024.0 * 1024.0 * 1024.0)) / elapsed;
        printf("%-10d %-15.6f %-15.1f\n", read_iterations, elapsed, bandwidth);
    }
    
    double avg_read_time = total_read_time / read_iterations;
    double read_bandwidth = (bytes / (1024.0 * 1024.0 * 1024.0)) / avg_read_time;
    
    printf("\nTesting GPU Host-to-Device transfer:\n");
    printf("%-10s %-15s %-15s\n", "Iteration", "Time(s)", "Bandwidth(GB/s)");
    printf("------------------------------------------\n");
    
    float* h_data = (float*)malloc(bytes);
    memset(h_data, 0, bytes);
    
    double total_h2d_time = 0;
    int h2d_iterations = 0;
    
    while(total_h2d_time < MIN_TEST_TIME) {
        double start = get_time();
        CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
        double end = get_time();
        
        double elapsed = end - start;
        total_h2d_time += elapsed;
        h2d_iterations++;
        
        double bandwidth = (bytes / (1024.0 * 1024.0 * 1024.0)) / elapsed;
        printf("%-10d %-15.6f %-15.1f\n", h2d_iterations, elapsed, bandwidth);
    }
    
    double avg_h2d_time = total_h2d_time / h2d_iterations;
    double h2d_bandwidth = (bytes / (1024.0 * 1024.0 * 1024.0)) / avg_h2d_time;
    
    printf("\nTesting GPU Device-to-Host transfer:\n");
    printf("%-10s %-15s %-15s\n", "Iteration", "Time(s)", "Bandwidth(GB/s)");
    printf("------------------------------------------\n");
    
    double total_d2h_time = 0;
    int d2h_iterations = 0;
    
    while(total_d2h_time < MIN_TEST_TIME) {
        double start = get_time();
        CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
        double end = get_time();
        
        double elapsed = end - start;
        total_d2h_time += elapsed;
        d2h_iterations++;
        
        double bandwidth = (bytes / (1024.0 * 1024.0 * 1024.0)) / elapsed;
        printf("%-10d %-15.6f %-15.1f\n", d2h_iterations, elapsed, bandwidth);
    }
    
    double avg_d2h_time = total_d2h_time / d2h_iterations;
    double d2h_bandwidth = (bytes / (1024.0 * 1024.0 * 1024.0)) / avg_d2h_time;
    
    printf("\n--- RESULTS for GPU Memory ---\n");
    printf("GPU Write Bandwidth:  %.1f GB/s (avg over %d iterations)\n", 
           write_bandwidth, write_iterations);
    printf("GPU Read Bandwidth:   %.1f GB/s (avg over %d iterations)\n", 
           read_bandwidth, read_iterations);
    printf("Host->Device:         %.1f GB/s (avg over %d iterations)\n", 
           h2d_bandwidth, h2d_iterations);
    printf("Device->Host:         %.1f GB/s (avg over %d iterations)\n", 
           d2h_bandwidth, d2h_iterations);
    printf("Note: GPU memory bandwidth is effectively unlimited channels (massively parallel)\n");
    
    // Store results
    g_results.gpu_write_bandwidth = write_bandwidth;
    g_results.gpu_read_bandwidth = read_bandwidth;
    g_results.gpu_h2d_bandwidth = h2d_bandwidth;
    g_results.gpu_d2h_bandwidth = d2h_bandwidth;
    g_results.gpu_write_iterations = write_iterations;
    g_results.gpu_read_iterations = read_iterations;
    g_results.gpu_h2d_iterations = h2d_iterations;
    g_results.gpu_d2h_iterations = d2h_iterations;
    
    free(h_data);
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_temp));
}

void print_summary_table() {
    printf("\n\n");
    printf("================================================================================\n");
    printf("                        BENCHMARK SUMMARY TABLE                                 \n");
    printf("================================================================================\n\n");
    
    printf("┌─────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│ GPU MEMORY PERFORMANCE                                                      │\n");
    printf("├─────────────────────────────────┬───────────────────┬───────────────────────┤\n");
    printf("│ Operation                       │ Bandwidth         │ Iterations            │\n");
    printf("├─────────────────────────────────┼───────────────────┼───────────────────────┤\n");
    printf("│ GPU Kernel Write                │ %8.1f GB/s     │ %5d                 │\n", 
           g_results.gpu_write_bandwidth, g_results.gpu_write_iterations);
    printf("│ GPU Kernel Read                 │ %8.1f GB/s     │ %5d                 │\n", 
           g_results.gpu_read_bandwidth, g_results.gpu_read_iterations);
    printf("│ Host to Device Transfer         │ %8.1f GB/s     │ %5d                 │\n", 
           g_results.gpu_h2d_bandwidth, g_results.gpu_h2d_iterations);
    printf("│ Device to Host Transfer         │ %8.1f GB/s     │ %5d                 │\n", 
           g_results.gpu_d2h_bandwidth, g_results.gpu_d2h_iterations);
    printf("└─────────────────────────────────┴───────────────────┴───────────────────────┘\n");
    printf("  Note: GPU has massively parallel architecture (thousands of concurrent ops)\n\n");
    
    printf("┌─────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│ RAMDISK PERFORMANCE (/ram/)                                                 │\n");
    printf("├─────────────────────────────────┬───────────────────┬───────────────────────┤\n");
    printf("│ Operation                       │ Speed             │ Optimal Channels      │\n");
    printf("├─────────────────────────────────┼───────────────────┼───────────────────────┤\n");
    printf("│ Write                           │ %8.1f MB/s     │ %5d threads         │\n", 
           g_results.ram_write_speed, g_results.ram_write_threads);
    printf("│ Read                            │ %8.1f MB/s     │ %5d threads         │\n", 
           g_results.ram_read_speed, g_results.ram_read_threads);
    printf("└─────────────────────────────────┴───────────────────┴───────────────────────┘\n\n");
    
    printf("┌─────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│ FILESYSTEM PERFORMANCE (/data/)                                             │\n");
    printf("├─────────────────────────────────┬───────────────────┬───────────────────────┤\n");
    printf("│ Operation                       │ Speed             │ Optimal Channels      │\n");
    printf("├─────────────────────────────────┼───────────────────┼───────────────────────┤\n");
    printf("│ Write                           │ %8.1f MB/s     │ %5d threads         │\n", 
           g_results.data_write_speed, g_results.data_write_threads);
    printf("│ Read                            │ %8.1f MB/s     │ %5d threads         │\n", 
           g_results.data_read_speed, g_results.data_read_threads);
    printf("└─────────────────────────────────┴───────────────────┴───────────────────────┘\n\n");
    
    printf("┌─────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│ COMPARATIVE ANALYSIS                                                        │\n");
    printf("├─────────────────────────────────────────────────────────────────────────────┤\n");
    
    // Convert to same units (GB/s) for comparison
    double ram_write_gbs = g_results.ram_write_speed / 1024.0;
    double ram_read_gbs = g_results.ram_read_speed / 1024.0;
    double data_write_gbs = g_results.data_write_speed / 1024.0;
    double data_read_gbs = g_results.data_read_speed / 1024.0;
    
    printf("│ GPU vs RAM Write Speed:          GPU is %.1fx faster                        │\n",
           g_results.gpu_write_bandwidth / ram_write_gbs);
    printf("│ GPU vs Disk Write Speed:         GPU is %.1fx faster                        │\n",
           g_results.gpu_write_bandwidth / data_write_gbs);
    printf("│ RAM vs Disk Write Speed:         RAM is %.1fx faster                        │\n",
           ram_write_gbs / data_write_gbs);
    printf("│                                                                             │\n");
    printf("│ GPU vs RAM Read Speed:           GPU is %.1fx faster                        │\n",
           g_results.gpu_read_bandwidth / ram_read_gbs);
    printf("│ GPU vs Disk Read Speed:          GPU is %.1fx faster                        │\n",
           g_results.gpu_read_bandwidth / data_read_gbs);
    printf("│ RAM vs Disk Read Speed:          RAM is %.1fx faster                        │\n",
           ram_read_gbs / data_read_gbs);
    printf("└─────────────────────────────────────────────────────────────────────────────┘\n\n");
    
    printf("┌─────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│ RECOMMENDATIONS                                                             │\n");
    printf("├─────────────────────────────────────────────────────────────────────────────┤\n");
    printf("│ • Use GPU memory for compute-intensive operations (best bandwidth)          │\n");
    printf("│ • Use /ram/ for temporary files requiring fast I/O                          │\n");
    printf("│ • Use /data/ for persistent storage                                         │\n");
    printf("│                                                                             │\n");
    printf("│ Optimal Thread Counts:                                                      │\n");
    printf("│ • /ram/ operations: %d threads                                              │\n",
           (g_results.ram_read_threads + g_results.ram_write_threads) / 2);
    printf("│ • /data/ operations: %d threads                                             │\n",
           (g_results.data_read_threads + g_results.data_write_threads) / 2);
    printf("└─────────────────────────────────────────────────────────────────────────────┘\n");
}

int main(int argc, char** argv) {
    // Initialize results structure
    memset(&g_results, 0, sizeof(benchmark_results));
    
    printf("GPU and Disk I/O Benchmark\n");
    printf("Test size: GPU=%d MB, Disk=%d MB\n", 
           GPU_TEST_SIZE/(1024*1024), DISK_TEST_SIZE/(1024*1024));
    printf("Minimum test time: %.1f seconds per configuration\n\n", MIN_TEST_TIME);
    
    // Check CUDA device
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if(deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Using GPU: %s\n", prop.name);
    printf("GPU Memory: %.1f GB\n\n", prop.totalGlobalMem / (1024.0*1024.0*1024.0));
    
    // Run benchmarks
    benchmark_gpu_memory();
    benchmark_disk_io("/ram", 1);
    benchmark_disk_io("/data", 0);
    
    // Print comprehensive summary table
    print_summary_table();
    
    printf("\n========================================\n");
    printf("Benchmark Complete\n");
    printf("========================================\n");
    
    return 0;
}

