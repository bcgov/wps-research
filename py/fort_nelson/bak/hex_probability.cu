/*
CUDA Hex Probability Aggregation

Sums ALL pixels within each hexagon boundary from classification rasters.
Uses background loading to overlap I/O with GPU computation.

Compile:
nvcc -O3 -arch=sm_80 hex_probability.cu -o hex_probability -I/usr/include/gdal -lgdal -lpthread

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

// ============ PARAMETERS ============
#define DEFAULT_HEX_SPACING 500.0f
#define NUM_GPU_BUFFERS 4          // Double-buffering for overlap
#define BLOCK_SIZE 256
#define MAX_RASTERS 256
#define MAX_PATH_LEN 512
// ====================================

// Raster metadata
typedef struct {
    char path[MAX_PATH_LEN];
    int year;
    int width, height;
    double geo_transform[6];
    double inv_geo_transform[6];
    double min_x, max_x, min_y, max_y;
    char proj_wkt[2048];
} RasterInfo;

// Hex cell result
typedef struct {
    double cx, cy;           // Center coordinates
    double prob_sum;         // Sum of probabilities
    int pixel_count;         // Number of valid pixels
    int best_year;           // Year of data source
    int src_raster_idx;      // Index of source raster
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
    
    RasterInfo* rasters = (RasterInfo*)calloc(glob_result.gl_pathc, sizeof(RasterInfo));
    int n_rasters = 0;
    size_t max_raster_size = 0;
    
    for (size_t i = 0; i < glob_result.gl_pathc; i++) {
        GDALDatasetH ds = GDALOpen(glob_result.gl_pathv[i], GA_ReadOnly);
        if (ds == NULL) continue;
        
        RasterInfo* r = &rasters[n_rasters];
        strncpy(r->path, glob_result.gl_pathv[i], MAX_PATH_LEN);
        r->year = extract_year_from_filename(glob_result.gl_pathv[i]);
        r->width = GDALGetRasterXSize(ds);
        r->height = GDALGetRasterYSize(ds);
        
        GDALGetGeoTransform(ds, r->geo_transform);
        GDALInvGeoTransform(r->geo_transform, r->inv_geo_transform);
        
        const char* proj = GDALGetProjectionRef(ds);
        if (proj) strncpy(r->proj_wkt, proj, 2047);
        
        r->min_x = r->geo_transform[0];
        r->max_x = r->geo_transform[0] + r->width * r->geo_transform[1];
        r->max_y = r->geo_transform[3];
        r->min_y = r->geo_transform[3] + r->height * r->geo_transform[5];
        
        size_t size = (size_t)r->width * r->height * sizeof(float);
        if (size > max_raster_size) max_raster_size = size;
        
        GDALClose(ds);
        n_rasters++;
    }
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
    
    // ---- Write output shapefile ----
    printf("Writing output shapefile...\n");
    
    OGRSFDriverH shp_driver = OGRGetDriverByName("ESRI Shapefile");
    OGRDataSourceH out_ds = OGR_Dr_CreateDataSource(shp_driver, output_shp, NULL);
    OGRLayerH out_layer = OGR_DS_CreateLayer(out_ds, "hex_probabilities", aoi_srs, wkbPolygon, NULL);
    
    OGRFieldDefnH fld_prob = OGR_Fld_Create("prob_mean", OFTReal);
    OGRFieldDefnH fld_sum = OGR_Fld_Create("prob_sum", OFTReal);
    OGRFieldDefnH fld_count = OGR_Fld_Create("n_pixels", OFTInteger);
    OGRFieldDefnH fld_year = OGR_Fld_Create("year", OFTInteger);
    
    OGR_L_CreateField(out_layer, fld_prob, 0);
    OGR_L_CreateField(out_layer, fld_sum, 0);
    OGR_L_CreateField(out_layer, fld_count, 0);
    OGR_L_CreateField(out_layer, fld_year, 0);
    
    OGR_Fld_Destroy(fld_prob);
    OGR_Fld_Destroy(fld_sum);
    OGR_Fld_Destroy(fld_count);
    OGR_Fld_Destroy(fld_year);
    
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
        
        // Create hex polygon
        OGRGeometryH hex_ring = OGR_G_CreateGeometry(wkbLinearRing);
        for (int v = 0; v < 6; v++) {
            float angle = M_PI / 6.0f + v * M_PI / 3.0f;
            float vx = hr->cx + hex_radius * cosf(angle);
            float vy = hr->cy + hex_radius * sinf(angle);
            OGR_G_AddPoint_2D(hex_ring, vx, vy);
        }
        OGR_G_AddPoint_2D(hex_ring, hr->cx + hex_radius * cosf(M_PI / 6.0f),
                                    hr->cy + hex_radius * sinf(M_PI / 6.0f));
        
        OGRGeometryH hex_poly = OGR_G_CreateGeometry(wkbPolygon);
        OGR_G_AddGeometryDirectly(hex_poly, hex_ring);
        
        // Create feature
        OGRFeatureH feat = OGR_F_Create(OGR_L_GetLayerDefn(out_layer));
        OGR_F_SetGeometry(feat, hex_poly);
        
        if (hr->pixel_count > 0) {
            double prob_mean = hr->prob_sum / hr->pixel_count;
            OGR_F_SetFieldDouble(feat, 0, prob_mean);
            OGR_F_SetFieldDouble(feat, 1, hr->prob_sum);
            OGR_F_SetFieldInteger(feat, 2, hr->pixel_count);
            OGR_F_SetFieldInteger(feat, 3, hr->best_year);
            
            n_with_data++;
            total_prob_sum += hr->prob_sum;
            total_pixel_count += hr->pixel_count;
            
            int class_idx = (int)(prob_mean * 10);
            if (class_idx > 9) class_idx = 9;
            class_counts[class_idx]++;
        } else {
            OGR_F_SetFieldNull(feat, 0);
            OGR_F_SetFieldDouble(feat, 1, 0.0);
            OGR_F_SetFieldInteger(feat, 2, 0);
            OGR_F_SetFieldInteger(feat, 3, 0);
            n_nodata++;
        }
        
        OGR_L_CreateFeature(out_layer, feat);
        OGR_F_Destroy(feat);
        OGR_G_DestroyGeometry(hex_poly);
    }
    
    OGR_DS_Destroy(out_ds);
    
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
    
    // ---- Write QML style file ----
    char qml_path[MAX_PATH_LEN];
    snprintf(qml_path, MAX_PATH_LEN, "%s", output_shp);
    char* dot = strrchr(qml_path, '.');
    if (dot) strcpy(dot, ".qml");
    
    printf("\nCreating style file: %s\n", qml_path);
    
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
    }
    
    printf("\nOutput saved to: %s\n", output_shp);
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
    free(rasters);
    
    OGR_G_DestroyGeometry(aoi_geom);
    OGR_DS_Destroy(aoi_ds);
    CPLFree(aoi_wkt);
    
    return 0;
}


