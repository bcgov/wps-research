/*
CUDA Hex Probability Aggregation

Sums ALL pixels within each hexagon boundary from classification rasters.
Uses background loading to overlap I/O with GPU computation.

Compile:
nvcc -O3 -arch=sm_80 hex_aggregate_cuda.cu -o hex_aggregate_cuda \
    -I/usr/include/gdal -lgdal -lpthread

Usage:
./hex_aggregate_cuda <aoi.shp> <output.shp> [hex_spacing_m]

Finds all *_classification.bin files in current directory.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <glob.h>
#include <regex.h>
#include <cuda_runtime.h>
#include <gdal.h>
#include <ogr_api.h>
#include <ogr_srs_api.h>
#include <cpl_conv.h>
#include <cpl_string.h>
#include <set>
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include <queue>

// ============ PARAMETERS ============
#define DEFAULT_HEX_SPACING 500.0f
#define NUM_GPU_BUFFERS 4          // Double-buffering for overlap
#define BLOCK_SIZE 256
#define MAX_RASTERS 256
#define MAX_PATH_LEN 512
// ====================================

// Raster metadata - C++ struct with proper initialization
struct RasterInfo {
    char path[MAX_PATH_LEN];
    int year;
    int width, height;
    double geo_transform[6];
    double inv_geo_transform[6];
    double min_x, max_x, min_y, max_y;
    char proj_wkt[2048];
    std::set<std::string> fire_numbers;
    
    // Constructor to ensure proper initialization
    RasterInfo() : year(0), width(0), height(0), min_x(0), max_x(0), min_y(0), max_y(0) {
        path[0] = '\0';
        proj_wkt[0] = '\0';
        for (int i = 0; i < 6; i++) {
            geo_transform[i] = 0.0;
            inv_geo_transform[i] = 0.0;
        }
        fire_numbers = std::set<std::string>();
    }
};

// Hex cell result
typedef struct {
    double cx, cy;           // Center coordinates (AOI CRS)
    double prob_sum;         // Sum of probabilities
    int pixel_count;         // Number of valid pixels
    int best_year;           // Year of data source
    int src_raster_idx;      // Index of source raster (for newest year)
    int component_id;        // Connected component ID (-1 = no data)
} HexResult;

// GPU buffer for async loading
typedef struct {
    float* d_data;           // Device pointer
    float* h_data;           // Host pointer (pinned)
    int raster_idx;          // Which raster is loaded
    int ready;               // Flag: data ready on GPU
    int width, height;
    cudaStream_t stream;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} GPUBuffer;

// Global state for background loader
typedef struct {
    RasterInfo* rasters;
    int n_rasters;
    GPUBuffer* buffers;
    int n_buffers;
    int next_to_load;        // Next raster index to load
    int shutdown;
    pthread_mutex_t mutex;
} LoaderState;

// ============ CUDA KERNELS ============

// Check if point is inside hexagon (pointy-top)
__device__ bool point_in_hex(float px, float py, float cx, float cy, float spacing) {
    float r = spacing / sqrtf(3.0f);
    
    // Transform to hex-local coordinates
    float dx = fabsf(px - cx);
    float dy = fabsf(py - cy);
    
    // Hexagon bounds check (pointy-top)
    float h = spacing * 0.5f;
    float w = r;
    
    if (dx > w || dy > h) return false;
    
    // Check against sloped edges
    return (h * w - h * dx - 0.5f * w * dy) >= 0;
}

// Debug kernel: count non-NaN pixels
__global__ void count_valid_pixels_kernel(
    const float* __restrict__ raster_data,
    int total_pixels,
    int* __restrict__ valid_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;
    
    float val = raster_data[idx];
    if (!isnan(val)) {
        atomicAdd(valid_count, 1);
    }
}

// Kernel: For each pixel in raster, determine which hex it belongs to and accumulate
__global__ void accumulate_pixels_kernel(
    const float* __restrict__ raster_data,
    int raster_width, int raster_height,
    const double* __restrict__ geo_transform,
    HexResult* __restrict__ hex_results,
    int n_hexes,
    float hex_spacing,
    float hex_min_x, float hex_min_y,
    int hex_cols, int hex_rows,
    int raster_year, int raster_idx)
{
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = raster_width * raster_height;
    
    if (pixel_idx >= total_pixels) return;
    
    int px = pixel_idx % raster_width;
    int py = pixel_idx / raster_width;
    
    float val = raster_data[pixel_idx];
    
    // Skip NaN pixels
    if (isnan(val)) return;
    
    // Convert pixel to geographic coordinates
    double gx = geo_transform[0] + px * geo_transform[1] + py * geo_transform[2];
    double gy = geo_transform[3] + px * geo_transform[4] + py * geo_transform[5];
    
    // Find candidate hex cell
    float v_spacing = hex_spacing * sqrtf(3.0f) * 0.5f;
    
    int row = (int)((gy - hex_min_y) / v_spacing + 0.5f);
    float x_offset = (row % 2 == 1) ? hex_spacing * 0.5f : 0.0f;
    int col = (int)((gx - hex_min_x - x_offset) / hex_spacing + 0.5f);
    
    // Check this and neighboring cells (pixel might be on boundary)
    for (int dr = -1; dr <= 1; dr++) {
        for (int dc = -1; dc <= 1; dc++) {
            int r = row + dr;
            int c = col + dc;
            
            if (r < 0 || r >= hex_rows || c < 0 || c >= hex_cols) continue;
            
            int hex_idx = r * hex_cols + c;
            if (hex_idx < 0 || hex_idx >= n_hexes) continue;
            
            // Calculate hex center
            float hex_x_offset = (r % 2 == 1) ? hex_spacing * 0.5f : 0.0f;
            float hcx = hex_min_x + c * hex_spacing + hex_x_offset;
            float hcy = hex_min_y + r * v_spacing;
            
            if (point_in_hex((float)gx, (float)gy, hcx, hcy, hex_spacing)) {
                // Atomic accumulation
                atomicAdd(&hex_results[hex_idx].prob_sum, (double)val);
                atomicAdd(&hex_results[hex_idx].pixel_count, 1);
                
                // Update year if this is newer (simple race, but ok for metadata)
                if (raster_year > hex_results[hex_idx].best_year) {
                    hex_results[hex_idx].best_year = raster_year;
                    hex_results[hex_idx].src_raster_idx = raster_idx;
                }
                return;  // Pixel belongs to one hex only
            }
        }
    }
}

// ============ BACKGROUND LOADER ============

void* background_loader_thread(void* arg) {
    LoaderState* state = (LoaderState*)arg;
    
    while (1) {
        pthread_mutex_lock(&state->mutex);
        
        if (state->shutdown) {
            pthread_mutex_unlock(&state->mutex);
            break;
        }
        
        int raster_idx = state->next_to_load;
        if (raster_idx >= state->n_rasters) {
            pthread_mutex_unlock(&state->mutex);
            usleep(10000);  // 10ms sleep
            continue;
        }
        
        // Find a free buffer
        GPUBuffer* buf = NULL;
        for (int i = 0; i < state->n_buffers; i++) {
            pthread_mutex_lock(&state->buffers[i].mutex);
            if (state->buffers[i].raster_idx == -1 || 
                state->buffers[i].raster_idx < raster_idx - state->n_buffers) {
                buf = &state->buffers[i];
                buf->raster_idx = raster_idx;
                state->next_to_load = raster_idx + 1;
                pthread_mutex_unlock(&state->buffers[i].mutex);
                break;
            }
            pthread_mutex_unlock(&state->buffers[i].mutex);
        }
        
        pthread_mutex_unlock(&state->mutex);
        
        if (buf == NULL) {
            usleep(10000);
            continue;
        }
        
        // Load raster data
        RasterInfo* info = &state->rasters[raster_idx];
        
        GDALDatasetH ds = GDALOpen(info->path, GA_ReadOnly);
        if (ds == NULL) {
            pthread_mutex_lock(&buf->mutex);
            buf->ready = 0;
            buf->raster_idx = -1;
            pthread_cond_signal(&buf->cond);
            pthread_mutex_unlock(&buf->mutex);
            continue;
        }
        
        GDALRasterBandH band = GDALGetRasterBand(ds, 1);
        
        // Read into pinned host memory
        pthread_mutex_lock(&buf->mutex);
        buf->width = info->width;
        buf->height = info->height;
        
        size_t size = (size_t)info->width * info->height * sizeof(float);
        
        GDALRasterIO(band, GF_Read, 0, 0, info->width, info->height,
                     buf->h_data, info->width, info->height, GDT_Float32, 0, 0);
        
        GDALClose(ds);
        
        // Async copy to GPU
        cudaMemcpyAsync(buf->d_data, buf->h_data, size, 
                        cudaMemcpyHostToDevice, buf->stream);
        cudaStreamSynchronize(buf->stream);
        
        buf->ready = 1;
        pthread_cond_signal(&buf->cond);
        pthread_mutex_unlock(&buf->mutex);
    }
    
    return NULL;
}

// ============ UTILITY FUNCTIONS ============

// Extract fire numbers from filename (e.g., "g80320", "g90461")
void extract_fire_numbers(const char* filename, std::set<std::string>& fire_numbers) {
    regex_t regex;
    regmatch_t matches[2];
    
    // Pattern: letter followed by 5 digits
    const char* pattern = "[a-zA-Z][0-9]{5}";
    
    if (regcomp(&regex, pattern, REG_EXTENDED) != 0) {
        return;
    }
    
    const char* cursor = filename;
    while (regexec(&regex, cursor, 2, matches, 0) == 0) {
        int len = matches[0].rm_eo - matches[0].rm_so;
        char fire_num[16];
        strncpy(fire_num, cursor + matches[0].rm_so, len);
        fire_num[len] = '\0';
        fire_numbers.insert(std::string(fire_num));
        cursor += matches[0].rm_eo;
    }
    
    regfree(&regex);
}

int extract_year_from_filename(const char* filename) {
    regex_t regex;
    regmatch_t match[2];
    
    // Try YYYYMM pattern
    if (regcomp(&regex, "_([0-9]{4})[0-9]{2,4}_", REG_EXTENDED) == 0) {
        if (regexec(&regex, filename, 2, match, 0) == 0) {
            char year_str[5];
            int len = match[1].rm_eo - match[1].rm_so;
            strncpy(year_str, filename + match[1].rm_so, len);
            year_str[len] = '\0';
            regfree(&regex);
            int year = atoi(year_str);
            if (year >= 2000 && year <= 2100) return year;
        }
        regfree(&regex);
    }
    
    // Try simple _YYYY_ pattern
    if (regcomp(&regex, "_([0-9]{4})_", REG_EXTENDED) == 0) {
        if (regexec(&regex, filename, 2, match, 0) == 0) {
            char year_str[5];
            int len = match[1].rm_eo - match[1].rm_so;
            strncpy(year_str, filename + match[1].rm_so, len);
            year_str[len] = '\0';
            regfree(&regex);
            int year = atoi(year_str);
            if (year >= 2000 && year <= 2100) return year;
        }
        regfree(&regex);
    }
    
    return 0;
}

int compare_rasters_by_year(const void* a, const void* b) {
    return ((RasterInfo*)b)->year - ((RasterInfo*)a)->year;  // Descending
}

void print_progress(int current, int total, time_t start_time) {
    float pct = 100.0f * current / total;
    time_t now = time(NULL);
    double elapsed = difftime(now, start_time);
    
    if (current > 0 && elapsed > 0) {
        double rate = current / elapsed;
        double remaining = (total - current) / rate;
        printf("\r  Processing raster %d/%d (%.1f%%) | ETA: %dm %02ds   ",
               current, total, pct, (int)(remaining/60), (int)fmod(remaining, 60));
    } else {
        printf("\r  Processing raster %d/%d (%.1f%%)   ", current, total, pct);
    }
    fflush(stdout);
}

// ============ CONNECTED COMPONENT ANALYSIS ============

// Find neighbors of a hex in the grid
void get_hex_neighbors(int row, int col, int hex_rows, int hex_cols, 
                       std::vector<std::pair<int,int>>& neighbors) {
    neighbors.clear();
    
    // Pointy-top hex has 6 neighbors
    int dr_even[6] = {-1, -1, 0, 1, 1, 0};
    int dc_even[6] = {-1, 0, 1, 0, -1, -1};
    
    int dr_odd[6] = {-1, -1, 0, 1, 1, 0};
    int dc_odd[6] = {0, 1, 1, 1, 0, -1};
    
    int* dr = (row % 2 == 0) ? dr_even : dr_odd;
    int* dc = (row % 2 == 0) ? dc_even : dc_odd;
    
    for (int i = 0; i < 6; i++) {
        int nr = row + dr[i];
        int nc = col + dc[i];
        if (nr >= 0 && nr < hex_rows && nc >= 0 && nc < hex_cols) {
            neighbors.push_back(std::make_pair(nr, nc));
        }
    }
}

// Label connected components using BFS
int label_connected_components(HexResult* hexes, int hex_rows, int hex_cols, 
                               OGRGeometryH aoi_geom) {
    int n_hexes = hex_rows * hex_cols;
    int component_id = 0;
    
    // BFS to label components
    for (int start_idx = 0; start_idx < n_hexes; start_idx++) {
        HexResult* start_hex = &hexes[start_idx];
        
        // Skip if already labeled or has no data
        if (start_hex->component_id >= 0 || start_hex->pixel_count == 0) continue;
        
        // Check if in AOI
        OGRGeometryH pt = OGR_G_CreateGeometry(wkbPoint);
        OGR_G_SetPoint_2D(pt, 0, start_hex->cx, start_hex->cy);
        int in_aoi = OGR_G_Contains(aoi_geom, pt);
        OGR_G_DestroyGeometry(pt);
        
        if (!in_aoi) continue;
        
        // Start new component
        std::queue<int> queue;
        queue.push(start_idx);
        start_hex->component_id = component_id;
        
        while (!queue.empty()) {
            int idx = queue.front();
            queue.pop();
            
            int row = idx / hex_cols;
            int col = idx % hex_cols;
            
            std::vector<std::pair<int,int>> neighbors;
            get_hex_neighbors(row, col, hex_rows, hex_cols, neighbors);
            
            for (auto& nb : neighbors) {
                int nb_idx = nb.first * hex_cols + nb.second;
                HexResult* nb_hex = &hexes[nb_idx];
                
                // Add to component if it has data and not yet labeled
                if (nb_hex->pixel_count > 0 && nb_hex->component_id < 0) {
                    // Check if in AOI
                    OGRGeometryH nb_pt = OGR_G_CreateGeometry(wkbPoint);
                    OGR_G_SetPoint_2D(nb_pt, 0, nb_hex->cx, nb_hex->cy);
                    int nb_in_aoi = OGR_G_Contains(aoi_geom, nb_pt);
                    OGR_G_DestroyGeometry(nb_pt);
                    
                    if (nb_in_aoi) {
                        nb_hex->component_id = component_id;
                        queue.push(nb_idx);
                    }
                }
            }
        }
        
        component_id++;
    }
    
    return component_id;
}

// Collect fire numbers for a component
std::string get_component_fire_label(HexResult* hexes, int n_hexes, int component_id,
                                     RasterInfo* rasters, int n_rasters) {
    printf("    [LABEL_FUNC] Entry: comp_id=%d\n", component_id);
    fflush(stdout);
    
    if (!hexes || !rasters || n_hexes <= 0 || n_rasters <= 0) {
        printf("    [LABEL_FUNC] Null/invalid params\n");
        fflush(stdout);
        return "unknown";
    }
    
    printf("    [LABEL_FUNC] Params valid\n");
    fflush(stdout);
    
    std::set<std::string> all_fire_numbers;
    
    printf("    [LABEL_FUNC] Set created\n");
    fflush(stdout);
    
    // First, find the bounding box of this component
    double comp_min_x = 1e10, comp_max_x = -1e10;
    double comp_min_y = 1e10, comp_max_y = -1e10;
    bool found_any = false;
    
    printf("    [LABEL_FUNC] Starting hex scan\n");
    fflush(stdout);
    
    for (int i = 0; i < n_hexes; i++) {
        if (hexes[i].component_id == component_id && hexes[i].pixel_count > 0) {
            if (hexes[i].cx < comp_min_x) comp_min_x = hexes[i].cx;
            if (hexes[i].cx > comp_max_x) comp_max_x = hexes[i].cx;
            if (hexes[i].cy < comp_min_y) comp_min_y = hexes[i].cy;
            if (hexes[i].cy > comp_max_y) comp_max_y = hexes[i].cy;
            found_any = true;
        }
    }
    
    printf("    [LABEL_FUNC] Hex scan done, found=%d\n", found_any);
    fflush(stdout);
    
    if (!found_any) {
        return "unknown";
    }
    
    // Now check which rasters spatially overlap with this component
    // Add a small buffer to account for edge cases
    double buffer = 1000.0;  // 1km buffer
    comp_min_x -= buffer;
    comp_max_x += buffer;
    comp_min_y -= buffer;
    comp_max_y += buffer;
    
    printf("    [LABEL_FUNC] Starting raster overlap check\n");
    fflush(stdout);
    
    for (int r = 0; r < n_rasters; r++) {
        // Check if raster extent overlaps with component extent
        bool overlaps = !(rasters[r].max_x < comp_min_x || 
                         rasters[r].min_x > comp_max_x ||
                         rasters[r].max_y < comp_min_y || 
                         rasters[r].min_y > comp_max_y);
        
        if (overlaps) {
            printf("    [LABEL_FUNC] Raster %d overlaps\n", r);
            fflush(stdout);
            
            // Add all fire numbers from this raster
            try {
                printf("    [LABEL_FUNC] Accessing fire_numbers for raster %d\n", r);
                fflush(stdout);
                
                // CRITICAL: Check if the set is valid before iterating
                size_t set_size = 0;
                try {
                    set_size = rasters[r].fire_numbers.size();
                    printf("    [LABEL_FUNC] Set size: %zu\n", set_size);
                    fflush(stdout);
                } catch (...) {
                    printf("    [LABEL_FUNC] Exception getting size, skipping raster %d\n", r);
                    fflush(stdout);
                    continue;
                }
                
                // Sanity check - if size is unreasonably large, the set is corrupted
                if (set_size > 100) {
                    printf("    [LABEL_FUNC] WARNING: Raster %d has suspicious set size %zu, skipping\n", r, set_size);
                    fflush(stdout);
                    continue;
                }
                
                int elem_count = 0;
                for (const auto& fire_num : rasters[r].fire_numbers) {
                    elem_count++;
                    printf("    [LABEL_FUNC] Element %d, length=%zu\n", elem_count, fire_num.size());
                    fflush(stdout);
                    
                    // Check if string length is reasonable
                    if (fire_num.size() > 50) {
                        printf("    [LABEL_FUNC] WARNING: String too long (%zu chars), corrupted data\n", fire_num.size());
                        fflush(stdout);
                        break;
                    }
                    
                    if (!fire_num.empty()) {
                        all_fire_numbers.insert(fire_num);
                    }
                    
                    if (elem_count > set_size) {
                        printf("    [LABEL_FUNC] ERROR: Iteration exceeded set size, breaking\n");
                        fflush(stdout);
                        break;
                    }
                }
                
                printf("    [LABEL_FUNC] Done with raster %d\n", r);
                fflush(stdout);
            } catch (const std::exception& e) {
                printf("    [LABEL_FUNC] Exception on raster %d: %s\n", r, e.what());
                fflush(stdout);
            } catch (...) {
                printf("    [LABEL_FUNC] Unknown exception on raster %d\n", r);
                fflush(stdout);
            }
        }
    }
    
    printf("    [LABEL_FUNC] All rasters checked\n");
    fflush(stdout);
    
    if (all_fire_numbers.empty()) {
        return "unknown";
    }
    
    // Sort and join with underscores
    std::vector<std::string> sorted_fires(all_fire_numbers.begin(), all_fire_numbers.end());
    std::sort(sorted_fires.begin(), sorted_fires.end());
    
    std::string result;
    for (size_t i = 0; i < sorted_fires.size(); i++) {
        if (i > 0) result += "_";
        result += sorted_fires[i];
    }
    
    printf("    [LABEL_FUNC] Returning: %s\n", result.c_str());
    fflush(stdout);
    
    return result;
}

// ============ RASTER EXPORT ============

void write_component_geotiff(const char* tif_path, HexResult* hexes, int n_hexes,
                             int component_id, float hex_spacing,
                             OGRSpatialReferenceH srs, double global_min_prob, double global_max_prob) {
    // Find component extent
    double min_x = 1e10, max_x = -1e10, min_y = 1e10, max_y = -1e10;
    int hex_count = 0;
    
    for (int i = 0; i < n_hexes; i++) {
        if (hexes[i].component_id == component_id && hexes[i].pixel_count > 0) {
            if (hexes[i].cx < min_x) min_x = hexes[i].cx;
            if (hexes[i].cx > max_x) max_x = hexes[i].cx;
            if (hexes[i].cy < min_y) min_y = hexes[i].cy;
            if (hexes[i].cy > max_y) max_y = hexes[i].cy;
            hex_count++;
        }
    }
    
    if (hex_count == 0) return;
    
    // Add buffer
    float buffer = hex_spacing * 2.0f;
    min_x -= buffer;
    max_x += buffer;
    min_y -= buffer;
    max_y += buffer;
    
    // Create raster at 100m resolution
    float pixel_size = 100.0f;
    int width = (int)((max_x - min_x) / pixel_size) + 1;
    int height = (int)((max_y - min_y) / pixel_size) + 1;
    
    // Allocate raster data (3 bands: R, G, B)
    float* raster_r = (float*)calloc(width * height, sizeof(float));
    float* raster_g = (float*)calloc(width * height, sizeof(float));
    float* raster_b = (float*)calloc(width * height, sizeof(float));
    
    // Initialize with NaN
    for (int i = 0; i < width * height; i++) {
        raster_r[i] = NAN;
        raster_g[i] = NAN;
        raster_b[i] = NAN;
    }
    
    // Rasterize hexagons with 'cool' colormap (cyan to magenta)
    float hex_radius = hex_spacing / sqrtf(3.0f);
    
    for (int i = 0; i < n_hexes; i++) {
        HexResult* hr = &hexes[i];
        if (hr->component_id != component_id || hr->pixel_count == 0) continue;
        
        double prob_mean = hr->prob_sum / hr->pixel_count;
        
        // Normalize to [0, 1] using global min/max
        float norm_val = 0.5f;
        if (global_max_prob > global_min_prob) {
            norm_val = (prob_mean - global_min_prob) / (global_max_prob - global_min_prob);
            if (norm_val < 0.0f) norm_val = 0.0f;
            if (norm_val > 1.0f) norm_val = 1.0f;
        }
        
        // 'cool' colormap: cyan (0,1,1) to magenta (1,0,1)
        // At 0: cyan (R=0, G=1, B=1)
        // At 1: magenta (R=1, G=0, B=1)
        float r = norm_val;
        float g = 1.0f - norm_val;
        float b = 1.0f;
        
        // Find pixels within hex boundary
        int cx_pix = (int)((hr->cx - min_x) / pixel_size);
        int cy_pix = (int)((hr->cy - min_y) / pixel_size);
        
        int radius_pix = (int)(hex_radius / pixel_size) + 2;
        
        for (int py = cy_pix - radius_pix; py <= cy_pix + radius_pix; py++) {
            for (int px = cx_pix - radius_pix; px <= cx_pix + radius_pix; px++) {
                if (px < 0 || px >= width || py < 0 || py >= height) continue;
                
                // Check if pixel center is inside hexagon
                double pixel_x = min_x + px * pixel_size + pixel_size / 2.0;
                double pixel_y = min_y + py * pixel_size + pixel_size / 2.0;
                
                // Point-in-hex test (simplified: use circle for now)
                double dx = pixel_x - hr->cx;
                double dy = pixel_y - hr->cy;
                double dist = sqrt(dx*dx + dy*dy);
                
                if (dist <= hex_radius) {
                    int idx = py * width + px;
                    raster_r[idx] = r * 255.0f;
                    raster_g[idx] = g * 255.0f;
                    raster_b[idx] = b * 255.0f;
                }
            }
        }
    }
    
    // Write GeoTIFF
    GDALDriverH gtiff_driver = GDALGetDriverByName("GTiff");
    
    char** options = NULL;
    options = CSLSetNameValue(options, "COMPRESS", "DEFLATE");
    options = CSLSetNameValue(options, "ZLEVEL", "9");
    
    GDALDatasetH out_ds = GDALCreate(gtiff_driver, tif_path, width, height, 3, GDT_Float32, options);
    CSLDestroy(options);
    
    if (!out_ds) {
        free(raster_r);
        free(raster_g);
        free(raster_b);
        return;
    }
    
    // Set geotransform
    double geo_transform[6];
    geo_transform[0] = min_x;
    geo_transform[1] = pixel_size;
    geo_transform[2] = 0.0;
    geo_transform[3] = max_y;
    geo_transform[4] = 0.0;
    geo_transform[5] = -pixel_size;
    
    GDALSetGeoTransform(out_ds, geo_transform);
    
    // Set projection
    char* wkt = NULL;
    OSRExportToWkt(srs, &wkt);
    if (wkt) {
        GDALSetProjection(out_ds, wkt);
        CPLFree(wkt);
    }
    
    // Write bands
    GDALRasterBandH band_r = GDALGetRasterBand(out_ds, 1);
    GDALRasterBandH band_g = GDALGetRasterBand(out_ds, 2);
    GDALRasterBandH band_b = GDALGetRasterBand(out_ds, 3);
    
    GDALSetRasterNoDataValue(band_r, NAN);
    GDALSetRasterNoDataValue(band_g, NAN);
    GDALSetRasterNoDataValue(band_b, NAN);
    
    GDALRasterIO(band_r, GF_Write, 0, 0, width, height, raster_r, width, height, GDT_Float32, 0, 0);
    GDALRasterIO(band_g, GF_Write, 0, 0, width, height, raster_g, width, height, GDT_Float32, 0, 0);
    GDALRasterIO(band_b, GF_Write, 0, 0, width, height, raster_b, width, height, GDT_Float32, 0, 0);
    
    GDALClose(out_ds);
    
    free(raster_r);
    free(raster_g);
    free(raster_b);
}

// ============ KML EXPORT ============

void write_kml_file(const char* kml_path, HexResult* hexes, int n_hexes,
                   float hex_spacing, OGRGeometryH aoi_geom,
                   OGRSpatialReferenceH src_srs) {
    printf("Creating KML file: %s\n", kml_path);
    
    // Create WGS84 SRS for KML
    OGRSpatialReferenceH wgs84 = OSRNewSpatialReference(NULL);
    OSRImportFromEPSG(wgs84, 4326);
    
    OGRCoordinateTransformationH transform = OCTNewCoordinateTransformation(src_srs, wgs84);
    if (!transform) {
        fprintf(stderr, "Failed to create coordinate transformation for KML\n");
        OSRDestroySpatialReference(wgs84);
        return;
    }
    
    FILE* kml = fopen(kml_path, "w");
    if (!kml) {
        fprintf(stderr, "Failed to create KML file\n");
        OCTDestroyCoordinateTransformation(transform);
        OSRDestroySpatialReference(wgs84);
        return;
    }
    
    // Write KML header
    fprintf(kml, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    fprintf(kml, "<kml xmlns=\"http://www.opengis.net/kml/2.2\">\n");
    fprintf(kml, "<Document>\n");
    fprintf(kml, "  <name>Hex Probability Aggregation</name>\n");
    
    // Define styles for probability ranges
    const char* colors[10] = {
        "c8ffffff", "c8ffe5ff", "c8ffdcff", "c8ffc8aa",
        "c8f0aa87", "c8dc8c64", "c8c86e46", "c8af502d",
        "c891371a", "c8781e00"
    };
    
    for (int i = 0; i < 10; i++) {
        fprintf(kml, "  <Style id=\"prob%d\">\n", i);
        fprintf(kml, "    <LineStyle><color>64808080</color><width>1</width></LineStyle>\n");
        fprintf(kml, "    <PolyStyle><color>%s</color></PolyStyle>\n", colors[i]);
        fprintf(kml, "  </Style>\n");
    }
    
    float hex_radius = hex_spacing / sqrtf(3.0f);
    
    // Write hexagons
    for (int i = 0; i < n_hexes; i++) {
        HexResult* hr = &hexes[i];
        
        if (hr->pixel_count == 0) continue;
        
        // Check if in AOI
        OGRGeometryH pt = OGR_G_CreateGeometry(wkbPoint);
        OGR_G_SetPoint_2D(pt, 0, hr->cx, hr->cy);
        int in_aoi = OGR_G_Contains(aoi_geom, pt);
        OGR_G_DestroyGeometry(pt);
        
        if (!in_aoi) continue;
        
        double prob_mean = hr->prob_sum / hr->pixel_count;
        int style_idx = (int)(prob_mean * 10);
        if (style_idx > 9) style_idx = 9;
        
        fprintf(kml, "  <Placemark>\n");
        fprintf(kml, "    <name>Hex %.2f</name>\n", prob_mean);
        fprintf(kml, "    <description>\n");
        fprintf(kml, "      Probability: %.4f&lt;br/&gt;\n", prob_mean);
        fprintf(kml, "      Pixels: %d&lt;br/&gt;\n", hr->pixel_count);
        fprintf(kml, "      Year: %d\n", hr->best_year);
        fprintf(kml, "    </description>\n");
        fprintf(kml, "    <styleUrl>#prob%d</styleUrl>\n", style_idx);
        fprintf(kml, "    <Polygon>\n");
        fprintf(kml, "      <outerBoundaryIs><LinearRing><coordinates>\n");
        
        // Create hex vertices and transform to WGS84
        for (int v = 0; v <= 6; v++) {
            float angle = M_PI / 6.0f + (v % 6) * M_PI / 3.0f;
            double vx = hr->cx + hex_radius * cosf(angle);
            double vy = hr->cy + hex_radius * sinf(angle);
            double vz = 0.0;
            
            OCTTransform(transform, 1, &vx, &vy, &vz);
            
            fprintf(kml, "        %.8f,%.8f,0\n", vx, vy);
        }
        
        fprintf(kml, "      </coordinates></LinearRing></outerBoundaryIs>\n");
        fprintf(kml, "    </Polygon>\n");
        fprintf(kml, "  </Placemark>\n");
    }
    
    fprintf(kml, "</Document>\n");
    fprintf(kml, "</kml>\n");
    fclose(kml);
    
    OCTDestroyCoordinateTransformation(transform);
    OSRDestroySpatialReference(wgs84);
}

// Write component-specific files

// ============ MAIN ============

int main(int argc, char** argv) {
    printf("\n");
    printf("========================================================================\n");
    printf("  CUDA Hex Probability Aggregation\n");
    printf("  - True pixel summation within hex boundaries\n");
    printf("  - Background loading with %d GPU buffers\n", NUM_GPU_BUFFERS);
    printf("========================================================================\n");
    
    if (argc < 3) {
        printf("\nUsage: %s <aoi.shp> <output.shp> [hex_spacing_m]\n", argv[0]);
        printf("  Finds all *_classification.bin files in current directory\n");
        return 1;
    }
    
    const char* aoi_shp = argv[1];
    const char* output_shp = argv[2];
    float hex_spacing = (argc > 3) ? atof(argv[3]) : DEFAULT_HEX_SPACING;
    
    printf("\nParameters:\n");
    printf("  AOI:         %s\n", aoi_shp);
    printf("  Output:      %s\n", output_shp);
    printf("  Hex spacing: %.1f m\n", hex_spacing);
    
    // Initialize GDAL/OGR
    GDALAllRegister();
    OGRRegisterAll();
    
    // ---- Load AOI ----
    printf("\nLoading AOI...\n");
    OGRDataSourceH aoi_ds = OGROpen(aoi_shp, 0, NULL);
    if (aoi_ds == NULL) {
        fprintf(stderr, "ERROR: Cannot open AOI shapefile\n");
        return 1;
    }
    
    OGRLayerH aoi_layer = OGR_DS_GetLayer(aoi_ds, 0);
    OGREnvelope aoi_extent;
    OGR_L_GetExtent(aoi_layer, &aoi_extent, 1);
    
    OGRSpatialReferenceH aoi_srs = OGR_L_GetSpatialRef(aoi_layer);
    char* aoi_wkt = NULL;
    OSRExportToWkt(aoi_srs, &aoi_wkt);
    
    // Get AOI geometry for point-in-polygon tests
    OGR_L_ResetReading(aoi_layer);
    OGRFeatureH aoi_feat = OGR_L_GetNextFeature(aoi_layer);
    OGRGeometryH aoi_geom = OGR_G_Clone(OGR_F_GetGeometryRef(aoi_feat));
    OGR_F_Destroy(aoi_feat);
    
    printf("  Extent: X(%.1f to %.1f), Y(%.1f to %.1f)\n",
           aoi_extent.MinX, aoi_extent.MaxX, aoi_extent.MinY, aoi_extent.MaxY);
    
    // ---- Find classification files ----
    printf("\nSearching for classification files...\n");
    glob_t glob_result;
    glob("*_classification.bin", 0, NULL, &glob_result);
    
    if (glob_result.gl_pathc == 0) {
        fprintf(stderr, "ERROR: No *_classification.bin files found\n");
        return 1;
    }
    
    std::vector<RasterInfo> rasters_vec;
    rasters_vec.reserve(glob_result.gl_pathc);
    
    size_t max_raster_size = 0;
    
    for (size_t i = 0; i < glob_result.gl_pathc; i++) {
        GDALDatasetH ds = GDALOpen(glob_result.gl_pathv[i], GA_ReadOnly);
        if (ds == NULL) continue;
        
        RasterInfo r;
        strncpy(r.path, glob_result.gl_pathv[i], MAX_PATH_LEN);
        r.year = extract_year_from_filename(glob_result.gl_pathv[i]);
        extract_fire_numbers(glob_result.gl_pathv[i], r.fire_numbers);  // Extract fire numbers
        r.width = GDALGetRasterXSize(ds);
        r.height = GDALGetRasterYSize(ds);
        
        GDALGetGeoTransform(ds, r.geo_transform);
        GDALInvGeoTransform(r.geo_transform, r.inv_geo_transform);
        
        const char* proj = GDALGetProjectionRef(ds);
        if (proj) strncpy(r.proj_wkt, proj, 2047);
        
        r.min_x = r.geo_transform[0];
        r.max_x = r.geo_transform[0] + r.width * r.geo_transform[1];
        r.max_y = r.geo_transform[3];
        r.min_y = r.geo_transform[3] + r.height * r.geo_transform[5];
        
        size_t size = (size_t)r.width * r.height * sizeof(float);
        if (size > max_raster_size) max_raster_size = size;
        
        rasters_vec.push_back(r);  // Add to vector
        GDALClose(ds);
    }
    
    int n_rasters = rasters_vec.size();
    RasterInfo* rasters = rasters_vec.data();  // Get pointer for compatibility
    globfree(&glob_result);
    
    // Sort by year (newest first)
    qsort(rasters, n_rasters, sizeof(RasterInfo), compare_rasters_by_year);
    
    printf("  Found %d rasters (sorted by year, newest first):\n", n_rasters);
    for (int i = 0; i < n_rasters && i < 5; i++) {
        printf("    [%d] %s (%dx%d)\n", rasters[i].year, rasters[i].path,
               rasters[i].width, rasters[i].height);
    }
    if (n_rasters > 5) printf("    ... and %d more\n", n_rasters - 5);
    
    printf("  Max raster size: %.1f MB\n", max_raster_size / 1e6);
    
    // Debug: Compare coordinate extents
    printf("\n  --- Coordinate Debug ---\n");
    printf("  AOI extent (original CRS):\n");
    printf("    X: %.1f to %.1f\n", aoi_extent.MinX, aoi_extent.MaxX);
    printf("    Y: %.1f to %.1f\n", aoi_extent.MinY, aoi_extent.MaxY);
    if (n_rasters > 0) {
        printf("  First raster extent:\n");
        printf("    X: %.1f to %.1f\n", rasters[0].min_x, rasters[0].max_x);
        printf("    Y: %.1f to %.1f\n", rasters[0].min_y, rasters[0].max_y);
        printf("    Pixel size: %.4f x %.4f\n", rasters[0].geo_transform[1], rasters[0].geo_transform[5]);
        
        // Check if extents overlap
        int overlaps = !(rasters[0].max_x < aoi_extent.MinX || 
                        rasters[0].min_x > aoi_extent.MaxX ||
                        rasters[0].max_y < aoi_extent.MinY || 
                        rasters[0].min_y > aoi_extent.MaxY);
        printf("    Extents overlap with AOI: %s\n", overlaps ? "YES" : "NO - NEED TO TRANSFORM AOI");
    }
    
    // Transform AOI to raster CRS if needed
    OGRSpatialReferenceH raster_srs = NULL;
    if (n_rasters > 0 && strlen(rasters[0].proj_wkt) > 0) {
        raster_srs = OSRNewSpatialReference(rasters[0].proj_wkt);
        
        if (raster_srs && !OSRIsSame(raster_srs, aoi_srs)) {
            printf("\n  Transforming AOI to raster CRS...\n");
            
            OGRCoordinateTransformationH transform = OCTNewCoordinateTransformation(aoi_srs, raster_srs);
            if (transform) {
                // Transform AOI geometry
                OGR_G_Transform(aoi_geom, transform);
                
                // Update AOI extent
                OGREnvelope new_extent;
                OGR_G_GetEnvelope(aoi_geom, &new_extent);
                aoi_extent = new_extent;
                
                printf("  Transformed AOI extent:\n");
                printf("    X: %.1f to %.1f\n", aoi_extent.MinX, aoi_extent.MaxX);
                printf("    Y: %.1f to %.1f\n", aoi_extent.MinY, aoi_extent.MaxY);
                
                // Update aoi_srs to raster CRS for output
                aoi_srs = raster_srs;
                
                OCTDestroyCoordinateTransformation(transform);
            }
        }
    }
    
    // ---- Generate hex grid ----
    printf("\nGenerating hex grid...\n");
    
    float v_spacing = hex_spacing * sqrtf(3.0f) * 0.5f;
    float hex_min_x = aoi_extent.MinX - hex_spacing;
    float hex_min_y = aoi_extent.MinY - hex_spacing;
    float hex_max_x = aoi_extent.MaxX + hex_spacing;
    float hex_max_y = aoi_extent.MaxY + hex_spacing;
    
    int hex_cols = (int)((hex_max_x - hex_min_x) / hex_spacing) + 2;
    int hex_rows = (int)((hex_max_y - hex_min_y) / v_spacing) + 2;
    int n_hexes_total = hex_cols * hex_rows;
    
    printf("  Grid: %d cols x %d rows = %d cells\n", hex_cols, hex_rows, n_hexes_total);
    
    // Initialize hex results on host
    HexResult* h_hex_results = (HexResult*)calloc(n_hexes_total, sizeof(HexResult));
    
    // Set hex centers and filter to AOI
    int n_hexes_in_aoi = 0;
    for (int row = 0; row < hex_rows; row++) {
        for (int col = 0; col < hex_cols; col++) {
            int idx = row * hex_cols + col;
            float x_offset = (row % 2 == 1) ? hex_spacing * 0.5f : 0.0f;
            h_hex_results[idx].cx = hex_min_x + col * hex_spacing + x_offset;
            h_hex_results[idx].cy = hex_min_y + row * v_spacing;
            h_hex_results[idx].prob_sum = 0.0;
            h_hex_results[idx].pixel_count = 0;
            h_hex_results[idx].best_year = 0;
            h_hex_results[idx].src_raster_idx = -1;
            h_hex_results[idx].component_id = -1;  // Initialize component ID
            
            // Check if in AOI
            OGRGeometryH pt = OGR_G_CreateGeometry(wkbPoint);
            OGR_G_SetPoint_2D(pt, 0, h_hex_results[idx].cx, h_hex_results[idx].cy);
            if (OGR_G_Contains(aoi_geom, pt)) {
                n_hexes_in_aoi++;
            }
            OGR_G_DestroyGeometry(pt);
        }
    }
    printf("  Hexes in AOI: %d\n", n_hexes_in_aoi);
    
    // ---- Allocate GPU resources ----
    printf("\nAllocating GPU resources...\n");
    
    HexResult* d_hex_results;
    cudaMalloc(&d_hex_results, n_hexes_total * sizeof(HexResult));
    cudaMemcpy(d_hex_results, h_hex_results, n_hexes_total * sizeof(HexResult), 
               cudaMemcpyHostToDevice);
    
    double* d_geo_transform;
    cudaMalloc(&d_geo_transform, 6 * sizeof(double));
    
    // Allocate GPU buffers for async loading
    GPUBuffer* buffers = (GPUBuffer*)calloc(NUM_GPU_BUFFERS, sizeof(GPUBuffer));
    for (int i = 0; i < NUM_GPU_BUFFERS; i++) {
        cudaMalloc(&buffers[i].d_data, max_raster_size);
        cudaMallocHost(&buffers[i].h_data, max_raster_size);
        cudaStreamCreate(&buffers[i].stream);
        pthread_mutex_init(&buffers[i].mutex, NULL);
        pthread_cond_init(&buffers[i].cond, NULL);
        buffers[i].raster_idx = -1;
        buffers[i].ready = 0;
    }
    
    printf("  Allocated %d GPU buffers (%.1f MB each)\n", 
           NUM_GPU_BUFFERS, max_raster_size / 1e6);
    
    // ---- Start background loader ----
    LoaderState loader_state = {
        .rasters = rasters,
        .n_rasters = n_rasters,
        .buffers = buffers,
        .n_buffers = NUM_GPU_BUFFERS,
        .next_to_load = 0,
        .shutdown = 0
    };
    pthread_mutex_init(&loader_state.mutex, NULL);
    
    pthread_t loader_thread;
    pthread_create(&loader_thread, NULL, background_loader_thread, &loader_state);
    
    // ---- Process rasters ----
    printf("\nProcessing rasters...\n");
    time_t start_time = time(NULL);
    
    for (int r_idx = 0; r_idx < n_rasters; r_idx++) {
        print_progress(r_idx + 1, n_rasters, start_time);
        
        RasterInfo* rinfo = &rasters[r_idx];
        
        // Find buffer with this raster
        GPUBuffer* buf = NULL;
        while (buf == NULL) {
            for (int i = 0; i < NUM_GPU_BUFFERS; i++) {
                pthread_mutex_lock(&buffers[i].mutex);
                if (buffers[i].raster_idx == r_idx && buffers[i].ready) {
                    buf = &buffers[i];
                    break;
                }
                pthread_mutex_unlock(&buffers[i].mutex);
            }
            if (buf == NULL) usleep(1000);
        }
        
        // Debug: Count valid pixels in first raster
        if (r_idx == 0) {
            int* d_valid_count;
            int h_valid_count = 0;
            cudaMalloc(&d_valid_count, sizeof(int));
            cudaMemset(d_valid_count, 0, sizeof(int));
            
            int total_pixels = rinfo->width * rinfo->height;
            int blocks = (total_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
            count_valid_pixels_kernel<<<blocks, BLOCK_SIZE>>>(buf->d_data, total_pixels, d_valid_count);
            cudaDeviceSynchronize();
            
            cudaMemcpy(&h_valid_count, d_valid_count, sizeof(int), cudaMemcpyDeviceToHost);
            printf("\n  DEBUG: First raster has %d valid (non-NaN) pixels out of %d total\n", 
                   h_valid_count, total_pixels);
            cudaFree(d_valid_count);
            
            // Debug: Print first few pixel coordinates
            printf("  DEBUG: First pixel geo coords: (%.1f, %.1f)\n",
                   rinfo->geo_transform[0], rinfo->geo_transform[3]);
            printf("  DEBUG: Hex grid origin: (%.1f, %.1f)\n", hex_min_x, hex_min_y);
            printf("  DEBUG: Hex spacing: %.1f, v_spacing: %.1f\n", hex_spacing, v_spacing);
        }
        
        // Copy geo transform to GPU
        cudaMemcpy(d_geo_transform, rinfo->geo_transform, 6 * sizeof(double),
                   cudaMemcpyHostToDevice);
        
        // Launch kernel
        int total_pixels = rinfo->width * rinfo->height;
        int blocks = (total_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        accumulate_pixels_kernel<<<blocks, BLOCK_SIZE>>>(
            buf->d_data, rinfo->width, rinfo->height,
            d_geo_transform, d_hex_results, n_hexes_total,
            hex_spacing, hex_min_x, hex_min_y, hex_cols, hex_rows,
            rinfo->year, r_idx
        );
        
        cudaDeviceSynchronize();
        
        // Release buffer
        buf->ready = 0;
        buf->raster_idx = -1;
        pthread_mutex_unlock(&buf->mutex);
    }
    printf("\n");
    
    // Shutdown loader thread
    pthread_mutex_lock(&loader_state.mutex);
    loader_state.shutdown = 1;
    pthread_mutex_unlock(&loader_state.mutex);
    pthread_join(loader_thread, NULL);
    
    // ---- Copy results back ----
    printf("\nCopying results from GPU...\n");
    cudaMemcpy(h_hex_results, d_hex_results, n_hexes_total * sizeof(HexResult),
               cudaMemcpyDeviceToHost);
    
    // ---- Connected component analysis ----
    printf("\nLabeling connected components...\n");
    int n_components = label_connected_components(h_hex_results, hex_rows, hex_cols, aoi_geom);
    printf("Found %d connected components\n", n_components);
    
    // ---- Write output shapefile ----
    printf("\nWriting output shapefile...\n");
    
    OGRSFDriverH shp_driver = OGRGetDriverByName("ESRI Shapefile");
    OGRDataSourceH out_ds = OGR_Dr_CreateDataSource(shp_driver, output_shp, NULL);
    OGRLayerH out_layer = OGR_DS_CreateLayer(out_ds, "hex_probabilities", aoi_srs, wkbPolygon, NULL);
    
    OGRFieldDefnH fld_prob = OGR_Fld_Create("prob_mean", OFTReal);
    OGRFieldDefnH fld_sum = OGR_Fld_Create("prob_sum", OFTReal);
    OGRFieldDefnH fld_count = OGR_Fld_Create("n_pixels", OFTInteger);
    OGRFieldDefnH fld_year = OGR_Fld_Create("year", OFTInteger);
    OGRFieldDefnH fld_comp = OGR_Fld_Create("component", OFTInteger);
    
    OGR_L_CreateField(out_layer, fld_prob, 0);
    OGR_L_CreateField(out_layer, fld_sum, 0);
    OGR_L_CreateField(out_layer, fld_count, 0);
    OGR_L_CreateField(out_layer, fld_year, 0);
    OGR_L_CreateField(out_layer, fld_comp, 0);
    
    OGR_Fld_Destroy(fld_prob);
    OGR_Fld_Destroy(fld_sum);
    OGR_Fld_Destroy(fld_count);
    OGR_Fld_Destroy(fld_year);
    OGR_Fld_Destroy(fld_comp);
    
    // Hex geometry
    float hex_radius = hex_spacing / sqrtf(3.0f);
    
    int n_with_data = 0;
    int n_nodata = 0;
    double total_prob_sum = 0.0;
    long long total_pixel_count = 0;
    int class_counts[10] = {0};
    
    for (int i = 0; i < n_hexes_total; i++) {
        HexResult* hr = &h_hex_results[i];
        
        // Check if in AOI
        OGRGeometryH pt = OGR_G_CreateGeometry(wkbPoint);
        OGR_G_SetPoint_2D(pt, 0, hr->cx, hr->cy);
        int in_aoi = OGR_G_Contains(aoi_geom, pt);
        OGR_G_DestroyGeometry(pt);
        
        if (!in_aoi) continue;
        
        // IMPORTANT: Only create polygons for hexagons with data (pixel_count > 0)
        // This dramatically reduces file size
        if (hr->pixel_count == 0) {
            n_nodata++;
            continue;
        }
        
        // Create hex polygon
        OGRGeometryH hex_ring = OGR_G_CreateGeometry(wkbLinearRing);
        float first_vx = hr->cx + hex_radius * cosf(M_PI / 6.0f);
        float first_vy = hr->cy + hex_radius * sinf(M_PI / 6.0f);
        OGR_G_AddPoint_2D(hex_ring, first_vx, first_vy);
        for (int v = 1; v < 6; v++) {
            float angle = M_PI / 6.0f + v * M_PI / 3.0f;
            float vx = hr->cx + hex_radius * cosf(angle);
            float vy = hr->cy + hex_radius * sinf(angle);
            OGR_G_AddPoint_2D(hex_ring, vx, vy);
        }
        OGR_G_AddPoint_2D(hex_ring, first_vx, first_vy);  // Close with exact same point
        
        OGRGeometryH hex_poly = OGR_G_CreateGeometry(wkbPolygon);
        OGR_G_AddGeometryDirectly(hex_poly, hex_ring);
        
        // Create feature
        OGRFeatureH feat = OGR_F_Create(OGR_L_GetLayerDefn(out_layer));
        OGR_F_SetGeometry(feat, hex_poly);
        
        // Set field values (only writing hexagons with data now)
        double prob_mean = hr->prob_sum / hr->pixel_count;
        OGR_F_SetFieldDouble(feat, 0, prob_mean);
        OGR_F_SetFieldDouble(feat, 1, hr->prob_sum);
        OGR_F_SetFieldInteger(feat, 2, hr->pixel_count);
        OGR_F_SetFieldInteger(feat, 3, hr->best_year);
        OGR_F_SetFieldInteger(feat, 4, hr->component_id);
        
        n_with_data++;
        total_prob_sum += hr->prob_sum;
        total_pixel_count += hr->pixel_count;
        
        int class_idx = (int)(prob_mean * 10);
        if (class_idx > 9) class_idx = 9;
        class_counts[class_idx]++;
        
        OGR_L_CreateFeature(out_layer, feat);
        OGR_F_Destroy(feat);
        OGR_G_DestroyGeometry(hex_poly);
    }
    
    OGR_DS_Destroy(out_ds);
    
    // ---- Write QML style file IMMEDIATELY after shapefile ----
    char qml_path[MAX_PATH_LEN];
    snprintf(qml_path, MAX_PATH_LEN, "%s", output_shp);
    char* qml_dot = strrchr(qml_path, '.');
    if (qml_dot) strcpy(qml_dot, ".qml");
    
    printf("Creating style file: %s\n", qml_path);
    
    FILE* qml = fopen(qml_path, "w");
    if (qml) {
        fprintf(qml, "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n");
        fprintf(qml, "<qgis version=\"3.0\" styleCategories=\"Symbology\">\n");
        fprintf(qml, "  <renderer-v2 type=\"graduatedSymbol\" attr=\"prob_mean\" graduatedMethod=\"GraduatedColor\">\n");
        fprintf(qml, "    <ranges>\n");
        for (int i = 0; i < 10; i++) {
            fprintf(qml, "      <range lower=\"%.1f\" upper=\"%.1f\" symbol=\"%d\" label=\"%.1f - %.1f\"/>\n",
                    i/10.0, (i+1)/10.0, i, i/10.0, (i+1)/10.0);
        }
        fprintf(qml, "    </ranges>\n");
        fprintf(qml, "    <symbols>\n");
        
        // White to blue gradient
        int colors[10][3] = {
            {255, 255, 255}, {229, 240, 255}, {203, 225, 255}, {170, 200, 255},
            {135, 170, 240}, {100, 140, 220}, {70, 110, 200}, {45, 80, 175},
            {25, 55, 145}, {0, 30, 120}
        };
        
        for (int i = 0; i < 10; i++) {
            fprintf(qml, "      <symbol type=\"fill\" name=\"%d\"><layer class=\"SimpleFill\">\n", i);
            fprintf(qml, "        <prop k=\"color\" v=\"%d,%d,%d,180\"/>", colors[i][0], colors[i][1], colors[i][2]);
            fprintf(qml, "<prop k=\"outline_color\" v=\"128,128,128,100\"/>");
            fprintf(qml, "<prop k=\"outline_width\" v=\"0.1\"/>\n");
            fprintf(qml, "      </layer></symbol>\n");
        }
        
        fprintf(qml, "    </symbols>\n");
        fprintf(qml, "  </renderer-v2>\n");
        fprintf(qml, "</qgis>\n");
        fclose(qml);
        printf("Style file created successfully.\n");
    } else {
        fprintf(stderr, "WARNING: Could not create QML style file\n");
    }
    
    // ---- Write component-specific SHAPEFILES (no KML yet) ----
    printf("\nWriting component-specific shapefiles...\n");
    fflush(stdout);
    
    printf("DEBUG: Starting component loop, n_components=%d\n", n_components);
    fflush(stdout);
    
    for (int comp_id = 0; comp_id < n_components; comp_id++) {
        printf("  Component %d/%d... [A: Loop entered]\n", comp_id + 1, n_components);
        fflush(stdout);
        
        printf("  Component %d/%d... [B: Creating string]\n", comp_id + 1, n_components);
        fflush(stdout);
        
        std::string fire_label;
        
        printf("  Component %d/%d... [C: String created, calling function]\n", comp_id + 1, n_components);
        fflush(stdout);
        
        try {
            printf("  Component %d/%d... [D: Inside try block]\n", comp_id + 1, n_components);
            fflush(stdout);
            
            fire_label = get_component_fire_label(h_hex_results, n_hexes_total, 
                                                  comp_id, rasters, n_rasters);
            
            printf("  Component %d/%d... [E: Function returned]\n", comp_id + 1, n_components);
            fflush(stdout);
        } catch (const std::exception& e) {
            printf(" ERROR: %s\n", e.what());
            fflush(stdout);
            continue;
        } catch (...) {
            printf(" ERROR: Unknown exception\n");
            fflush(stdout);
            continue;
        }
        
        printf("  Component %d/%d... [F: After try-catch, label='%s']\n", comp_id + 1, n_components, fire_label.c_str());
        fflush(stdout);
        
        if (fire_label.empty() || fire_label == "unknown") {
            printf(" skipped (label: %s)\n", fire_label.empty() ? "empty" : "unknown");
            continue;
        }
        
        // Extract base name without extension
        char base_name[MAX_PATH_LEN];
        strncpy(base_name, output_shp, MAX_PATH_LEN - 1);
        base_name[MAX_PATH_LEN - 1] = '\0';
        char* dot = strrchr(base_name, '.');
        if (dot) *dot = '\0';
        
        char shp_path[MAX_PATH_LEN];
        int ret = snprintf(shp_path, MAX_PATH_LEN, "%s_component_%s.shp", base_name, fire_label.c_str());
        if (ret < 0 || ret >= MAX_PATH_LEN) {
            printf(" ERROR: Path too long\n");
            continue;
        }
        
        printf(" %s...", fire_label.c_str());
        fflush(stdout);
        
        // Create shapefile for this component
        OGRSFDriverH shp_driver = OGRGetDriverByName("ESRI Shapefile");
        if (!shp_driver) {
            printf(" ERROR: Cannot get driver\n");
            continue;
        }
        
        OGRDataSourceH ds = OGR_Dr_CreateDataSource(shp_driver, shp_path, NULL);
        if (!ds) {
            printf(" ERROR: Cannot create datasource\n");
            continue;
        }
        
        OGRLayerH layer = OGR_DS_CreateLayer(ds, "hex_component", aoi_srs, wkbPolygon, NULL);
        if (!layer) {
            printf(" ERROR: Cannot create layer\n");
            OGR_DS_Destroy(ds);
            continue;
        }
        
        OGRFieldDefnH fld_prob = OGR_Fld_Create("prob_mean", OFTReal);
        OGRFieldDefnH fld_sum = OGR_Fld_Create("prob_sum", OFTReal);
        OGRFieldDefnH fld_count = OGR_Fld_Create("n_pixels", OFTInteger);
        OGRFieldDefnH fld_year = OGR_Fld_Create("year", OFTInteger);
        
        OGR_L_CreateField(layer, fld_prob, 0);
        OGR_L_CreateField(layer, fld_sum, 0);
        OGR_L_CreateField(layer, fld_count, 0);
        OGR_L_CreateField(layer, fld_year, 0);
        
        OGR_Fld_Destroy(fld_prob);
        OGR_Fld_Destroy(fld_sum);
        OGR_Fld_Destroy(fld_count);
        OGR_Fld_Destroy(fld_year);
        
        // Write hexes belonging to this component
        int hex_count = 0;
        for (int i = 0; i < n_hexes_total; i++) {
            HexResult* hr = &h_hex_results[i];
            
            if (hr->component_id != comp_id || hr->pixel_count == 0) continue;
            
            // Create hex polygon
            OGRGeometryH hex_ring = OGR_G_CreateGeometry(wkbLinearRing);
            float first_vx = hr->cx + hex_radius * cosf(M_PI / 6.0f);
            float first_vy = hr->cy + hex_radius * sinf(M_PI / 6.0f);
            OGR_G_AddPoint_2D(hex_ring, first_vx, first_vy);
            for (int v = 1; v < 6; v++) {
                float angle = M_PI / 6.0f + v * M_PI / 3.0f;
                float vx = hr->cx + hex_radius * cosf(angle);
                float vy = hr->cy + hex_radius * sinf(angle);
                OGR_G_AddPoint_2D(hex_ring, vx, vy);
            }
            OGR_G_AddPoint_2D(hex_ring, first_vx, first_vy);
            
            OGRGeometryH hex_poly = OGR_G_CreateGeometry(wkbPolygon);
            OGR_G_AddGeometryDirectly(hex_poly, hex_ring);
            
            OGRFeatureH feat = OGR_F_Create(OGR_L_GetLayerDefn(layer));
            OGR_F_SetGeometry(feat, hex_poly);
            
            double prob_mean = hr->prob_sum / hr->pixel_count;
            OGR_F_SetFieldDouble(feat, 0, prob_mean);
            OGR_F_SetFieldDouble(feat, 1, hr->prob_sum);
            OGR_F_SetFieldInteger(feat, 2, hr->pixel_count);
            OGR_F_SetFieldInteger(feat, 3, hr->best_year);
            
            OGR_L_CreateFeature(layer, feat);
            OGR_F_Destroy(feat);
            OGR_G_DestroyGeometry(hex_poly);
            hex_count++;
        }
        
        OGR_DS_Destroy(ds);
        printf(" %s (%d hexes)\n", fire_label.c_str(), hex_count);
        fflush(stdout);
    }
    
    printf("Component shapefiles completed.\n\n");
    fflush(stdout);
    
    // ---- Calculate global min/max probability for color scaling ----
    double global_min_prob = 1.0;
    double global_max_prob = 0.0;
    
    for (int i = 0; i < n_hexes_total; i++) {
        if (h_hex_results[i].pixel_count > 0) {
            double prob_mean = h_hex_results[i].prob_sum / h_hex_results[i].pixel_count;
            if (prob_mean < global_min_prob) global_min_prob = prob_mean;
            if (prob_mean > global_max_prob) global_max_prob = prob_mean;
        }
    }
    
    printf("Probability range: %.4f to %.4f\n", global_min_prob, global_max_prob);
    printf("Color mapping: %.4f=cyan, %.4f=magenta\n\n", global_min_prob, global_max_prob);
    
    // ---- Write component GeoTIFF rasters (100m resolution, 'cool' colormap) ----
    printf("Creating component GeoTIFF rasters (100m resolution)...\n");
    fflush(stdout);
    
    for (int comp_id = 0; comp_id < n_components; comp_id++) {
        printf("  Component TIF %d/%d...", comp_id + 1, n_components);
        fflush(stdout);
        
        std::string fire_label;
        try {
            fire_label = get_component_fire_label(h_hex_results, n_hexes_total, 
                                                  comp_id, rasters, n_rasters);
        } catch (...) {
            printf(" ERROR: Cannot get fire label\n");
            continue;
        }
        
        if (fire_label.empty() || fire_label == "unknown") {
            printf(" skipped (label: %s)\n", fire_label.empty() ? "empty" : "unknown");
            continue;
        }
        
        // Extract base name
        char base_name[MAX_PATH_LEN];
        strncpy(base_name, output_shp, MAX_PATH_LEN - 1);
        base_name[MAX_PATH_LEN - 1] = '\0';
        char* dot_tif = strrchr(base_name, '.');
        if (dot_tif) *dot_tif = '\0';
        
        char tif_path[MAX_PATH_LEN];
        snprintf(tif_path, MAX_PATH_LEN, "%s_component_%s.tif", base_name, fire_label.c_str());
        
        write_component_geotiff(tif_path, h_hex_results, n_hexes_total, comp_id, 
                               hex_spacing, aoi_srs, global_min_prob, global_max_prob);
        
        printf(" %s\n", fire_label.c_str());
        fflush(stdout);
    }
    
    printf("Component GeoTIFFs completed.\n\n");
    fflush(stdout);
    
    // ---- NOW write KML files (after all shapefiles and TIFFs are safe) ----
    printf("Creating KML files...\n");
    fflush(stdout);
    
    // Main KML
    char kml_path[MAX_PATH_LEN];
    strncpy(kml_path, output_shp, MAX_PATH_LEN);
    char* dot = strrchr(kml_path, '.');
    if (dot) strcpy(dot, ".kml");
    
    write_kml_file(kml_path, h_hex_results, n_hexes_total, hex_spacing, aoi_geom, aoi_srs);
    
    // Component KML files
    printf("Creating component KML files...\n");
    fflush(stdout);
    
    for (int comp_id = 0; comp_id < n_components; comp_id++) {
        printf("  Component KML %d/%d...", comp_id + 1, n_components);
        fflush(stdout);
        
        std::string fire_label = get_component_fire_label(h_hex_results, n_hexes_total, 
                                                          comp_id, rasters, n_rasters);
        if (fire_label.empty()) {
            printf(" skipped\n");
            continue;
        }
        
        // Extract base name
        char base_name[MAX_PATH_LEN];
        strncpy(base_name, output_shp, MAX_PATH_LEN);
        char* dot2 = strrchr(base_name, '.');
        if (dot2) *dot2 = '\0';
        
        char comp_kml_path[MAX_PATH_LEN];
        snprintf(comp_kml_path, MAX_PATH_LEN, "%s_component_%s.kml", base_name, fire_label.c_str());
        
        FILE* kml = fopen(comp_kml_path, "w");
        if (!kml) {
            printf(" ERROR: Cannot create file\n");
            continue;
        }
        
        OGRSpatialReferenceH wgs84 = OSRNewSpatialReference(NULL);
        OSRImportFromEPSG(wgs84, 4326);
        OGRCoordinateTransformationH transform = OCTNewCoordinateTransformation(aoi_srs, wgs84);
        
        if (!transform) {
            printf(" ERROR: Cannot create transform\n");
            fclose(kml);
            OSRDestroySpatialReference(wgs84);
            continue;
        }
        
        fprintf(kml, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        fprintf(kml, "<kml xmlns=\"http://www.opengis.net/kml/2.2\">\n");
        fprintf(kml, "<Document>\n");
        fprintf(kml, "  <n>Component: %s</n>\n", fire_label.c_str());
        
        // Define styles
        const char* colors[10] = {
            "c8ffffff", "c8ffe5ff", "c8ffdcff", "c8ffc8aa",
            "c8f0aa87", "c8dc8c64", "c8c86e46", "c8af502d",
            "c891371a", "c8781e00"
        };
        
        for (int i = 0; i < 10; i++) {
            fprintf(kml, "  <Style id=\"prob%d\">\n", i);
            fprintf(kml, "    <LineStyle><color>64808080</color><width>1</width></LineStyle>\n");
            fprintf(kml, "    <PolyStyle><color>%s</color></PolyStyle>\n", colors[i]);
            fprintf(kml, "  </Style>\n");
        }
        
        // Write hexes
        int hex_count = 0;
        for (int i = 0; i < n_hexes_total; i++) {
            HexResult* hr = &h_hex_results[i];
            
            if (hr->component_id != comp_id || hr->pixel_count == 0) continue;
            
            double prob_mean = hr->prob_sum / hr->pixel_count;
            int style_idx = (int)(prob_mean * 10);
            if (style_idx > 9) style_idx = 9;
            
            fprintf(kml, "  <Placemark>\n");
            fprintf(kml, "    <styleUrl>#prob%d</styleUrl>\n", style_idx);
            fprintf(kml, "    <Polygon>\n");
            fprintf(kml, "      <outerBoundaryIs><LinearRing><coordinates>\n");
            
            for (int v = 0; v <= 6; v++) {
                float angle = M_PI / 6.0f + (v % 6) * M_PI / 3.0f;
                double vx = hr->cx + hex_radius * cosf(angle);
                double vy = hr->cy + hex_radius * sinf(angle);
                double vz = 0.0;
                
                OCTTransform(transform, 1, &vx, &vy, &vz);
                
                fprintf(kml, "        %.8f,%.8f,0\n", vx, vy);
            }
            
            fprintf(kml, "      </coordinates></LinearRing></outerBoundaryIs>\n");
            fprintf(kml, "    </Polygon>\n");
            fprintf(kml, "  </Placemark>\n");
            hex_count++;
        }
        
        fprintf(kml, "</Document>\n");
        fprintf(kml, "</kml>\n");
        fclose(kml);
        
        OCTDestroyCoordinateTransformation(transform);
        OSRDestroySpatialReference(wgs84);
        
        printf(" %s (%d hexes)\n", fire_label.c_str(), hex_count);
        fflush(stdout);
    }
    
    printf("All KML files completed.\n\n");
    fflush(stdout);
    
    // ---- Statistics ----
    float hex_area_m2 = (3.0f * sqrtf(3.0f) / 2.0f) * hex_radius * hex_radius;
    float hex_area_km2 = hex_area_m2 / 1e6f;
    float hex_area_ha = hex_area_m2 / 1e4f;
    
    time_t end_time = time(NULL);
    double elapsed = difftime(end_time, start_time);
    
    printf("\n");
    printf("========================================================================\n");
    printf("Summary\n");
    printf("========================================================================\n");
    printf("Processing time:    %.1f seconds\n", elapsed);
    printf("Total hexes in AOI: %d\n", n_with_data + n_nodata);
    printf("With valid data:    %d\n", n_with_data);
    printf("Nodata:             %d\n", n_nodata);
    printf("Total pixels processed: %lld\n", total_pixel_count);
    printf("Connected components: %d\n", n_components);
    
    printf("\n");
    printf("------------------------------------------------------------------------\n");
    printf("Area Statistics\n");
    printf("------------------------------------------------------------------------\n");
    printf("Hex cell area: %.4f km² (%.2f ha)\n", hex_area_km2, hex_area_ha);
    
    float total_data_area_km2 = n_with_data * hex_area_km2;
    float total_data_area_ha = n_with_data * hex_area_ha;
    printf("\nTotal area with probability data:\n");
    printf("  %.2f km²\n", total_data_area_km2);
    printf("  %.2f ha\n", total_data_area_ha);
    
    printf("\nArea by probability class:\n");
    printf("  %-12s %10s %12s %12s\n", "Class", "Hexagons", "km²", "ha");
    printf("  %-12s %10s %12s %12s\n", "------------", "----------", "------------", "------------");
    
    for (int i = 0; i < 10; i++) {
        float lower = i / 10.0f;
        float upper = (i + 1) / 10.0f;
        float area_km2 = class_counts[i] * hex_area_km2;
        float area_ha = class_counts[i] * hex_area_ha;
        printf("  %.1f - %.1f      %10d %12.2f %12.2f\n", 
               lower, upper, class_counts[i], area_km2, area_ha);
    }
    
    printf("  %-12s %10s %12s %12s\n", "------------", "----------", "------------", "------------");
    printf("  %-12s %10d %12.2f %12.2f\n", "TOTAL", n_with_data, total_data_area_km2, total_data_area_ha);
    
    // Equivalent area at 100% detection
    // Each pixel represents (pixel_size)^2 area
    // We need to get pixel size from first raster
    double pixel_area_m2 = fabs(rasters[0].geo_transform[1] * rasters[0].geo_transform[5]);
    double equiv_area_m2 = total_prob_sum * pixel_area_m2;
    double equiv_area_km2 = equiv_area_m2 / 1e6;
    double equiv_area_ha = equiv_area_m2 / 1e4;
    
    printf("\n");
    printf("------------------------------------------------------------------------\n");
    printf("Equivalent Area at 100%% Detection Probability\n");
    printf("------------------------------------------------------------------------\n");
    printf("Sum of (pixel_area × probability) across all pixels:\n");
    printf("  %.2f km²\n", equiv_area_km2);
    printf("  %.2f ha\n", equiv_area_ha);
    printf("\nThis represents the equivalent area if all detections\n");
    printf("were concentrated at 100%% probability.\n");
    
    printf("\n");
    printf("========================================================================\n");
    printf("Output Files\n");
    printf("========================================================================\n");
    printf("Main shapefile:     %s\n", output_shp);
    printf("Main KML:           %s\n", kml_path);
    printf("QML style:          %s\n", qml_path);
    printf("Component files:    %d shapefiles + %d GeoTIFFs + %d KML files\n", n_components, n_components, n_components);
    printf("========================================================================\n");
    
    // ---- Cleanup ----
    for (int i = 0; i < NUM_GPU_BUFFERS; i++) {
        cudaFree(buffers[i].d_data);
        cudaFreeHost(buffers[i].h_data);
        cudaStreamDestroy(buffers[i].stream);
        pthread_mutex_destroy(&buffers[i].mutex);
        pthread_cond_destroy(&buffers[i].cond);
    }
    free(buffers);
    
    cudaFree(d_hex_results);
    cudaFree(d_geo_transform);
    free(h_hex_results);
    // rasters_vec will be automatically cleaned up
    
    OGR_G_DestroyGeometry(aoi_geom);
    OGR_DS_Destroy(aoi_ds);
    CPLFree(aoi_wkt);
    
    return 0;
}

