/** 2026012*
 * Minimum Noise Fraction (MNF) Transform - CPU Implementation
 * Parallelized with OpenMP, using Eigen for linear algebra
 * Designed for large datasets that exceed GPU memory
 *
 * MNF transform steps:
 * 1. (Optional) FFT-based destriping to remove detector artifacts
 * 2. Estimate noise covariance matrix from spatial differences
 * 3. Compute data covariance matrix
 * 4. Solve generalized eigenvalue problem: Σ_data * v = λ * Σ_noise * v
 * 5. Project data onto eigenvectors sorted by decreasing SNR (eigenvalue)
 *
 * Usage: mnf_transform <input_raster> <output_raster> [options]
 *
 * Compile (choose one based on your Eigen installation):
 *   # If Eigen is in /usr/include/eigen3:
 *   g++ -O3 -march=native -fopenmp -DNDEBUG mnf_transform_cpu.cpp -o mnf_transform \
 *       $(gdal-config --cflags) $(gdal-config --libs) -lfftw3 -lfftw3_threads -lm
 *
 *   # If Eigen is elsewhere, specify the path:
 *   g++ -O3 -march=native -fopenmp -DNDEBUG mnf_transform_cpu.cpp -o mnf_transform \
 *       $(gdal-config --cflags) $(gdal-config --libs) -lfftw3 -lfftw3_threads -lm -I/path/to/eigen
 *
 *   # Dependencies:
 *   # On Ubuntu/Debian: sudo apt install libeigen3-dev libfftw3-dev
 *   # On RHEL/Fedora:   sudo dnf install eigen3-devel fftw-devel
 *   # On macOS:         brew install eigen fftw
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>

#include <gdal.h>
#include <gdal_priv.h>
#include <cpl_conv.h>
#include <ogr_spatialref.h>

#include <fftw3.h>

using namespace Eigen;
using namespace std;

// ============================================================================
// Configuration
// ============================================================================

struct MNFConfig {
    string input_file;
    string output_file;
    int n_components = 0;      // 0 = all
    bool do_inverse = false;
    bool quiet = false;
    int num_threads = 0;       // 0 = auto
    size_t block_rows = 1024;  // Process this many rows at a time for covariance
    int noise_window = 9;      // Window size for noise estimation (must be odd)
    bool destripe = false;     // Apply FFT-based destriping before MNF
    float stripe_threshold = 3.0f;  // Threshold for stripe detection (std devs)
};

// ============================================================================
// Timing utility
// ============================================================================

class Timer {
    chrono::high_resolution_clock::time_point start_;
    string name_;
    bool quiet_;
public:
    Timer(const string& name, bool quiet = false) : name_(name), quiet_(quiet) {
        start_ = chrono::high_resolution_clock::now();
        if (!quiet_) cout << name_ << "... " << flush;
    }
    ~Timer() {
        auto end = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double, milli>(end - start_).count();
        if (!quiet_) cout << "done (" << fixed << setprecision(1) << ms << " ms)" << endl;
    }
};

// ============================================================================
// FFT-based Destriping
// ============================================================================

/**
 * Apply FFT-based notch filter to remove horizontal stripes from a single band
 * 
 * Horizontal stripes appear as high-energy vertical lines in FFT domain
 * (at fx ≈ 0, various fy). We detect and suppress these.
 * 
 * @param band_data Input/output band data (modified in place)
 * @param width Image width
 * @param height Image height  
 * @param threshold Detection threshold in standard deviations
 * @param quiet Suppress output
 * @return Number of frequencies notched
 */
int destripe_band_fft(float* band_data, int width, int height, 
                       float threshold, bool quiet) {
    
    size_t pixels = (size_t)width * height;
    
    // Allocate FFTW arrays
    double* spatial = fftw_alloc_real(pixels);
    fftw_complex* freq = fftw_alloc_complex(height * (width/2 + 1));
    
    if (!spatial || !freq) {
        cerr << "Error: Failed to allocate FFT buffers" << endl;
        if (spatial) fftw_free(spatial);
        if (freq) fftw_free(freq);
        return 0;
    }
    
    // Copy to double for FFTW
    for (size_t i = 0; i < pixels; i++) {
        spatial[i] = band_data[i];
    }
    
    // Create FFT plans
    fftw_plan plan_fwd = fftw_plan_dft_r2c_2d(height, width, spatial, freq, FFTW_ESTIMATE);
    fftw_plan plan_inv = fftw_plan_dft_c2r_2d(height, width, freq, spatial, FFTW_ESTIMATE);
    
    // Forward FFT
    fftw_execute(plan_fwd);
    
    int freq_width = width / 2 + 1;
    int notch_count = 0;
    
    // Analyze the DC column (fx = 0) where horizontal stripes appear
    // Skip DC (fy = 0) and very low frequencies
    int min_fy = max(3, height / 100);  // Skip lowest frequencies (real features)
    
    // Compute statistics of magnitude in the stripe-prone region
    vector<double> dc_col_mags;
    dc_col_mags.reserve(height - min_fy);
    
    for (int fy = min_fy; fy < height / 2; fy++) {
        double re = freq[fy * freq_width + 0][0];
        double im = freq[fy * freq_width + 0][1];
        double mag = sqrt(re * re + im * im);
        dc_col_mags.push_back(mag);
    }
    
    // Also check symmetric frequencies
    for (int fy = height / 2 + 1; fy < height - min_fy; fy++) {
        double re = freq[fy * freq_width + 0][0];
        double im = freq[fy * freq_width + 0][1];
        double mag = sqrt(re * re + im * im);
        dc_col_mags.push_back(mag);
    }
    
    if (dc_col_mags.empty()) {
        fftw_destroy_plan(plan_fwd);
        fftw_destroy_plan(plan_inv);
        fftw_free(spatial);
        fftw_free(freq);
        return 0;
    }
    
    // Compute robust statistics (median and MAD)
    vector<double> sorted_mags = dc_col_mags;
    size_t mid = sorted_mags.size() / 2;
    nth_element(sorted_mags.begin(), sorted_mags.begin() + mid, sorted_mags.end());
    double median_mag = sorted_mags[mid];
    
    // Compute MAD
    vector<double> abs_devs(dc_col_mags.size());
    for (size_t i = 0; i < dc_col_mags.size(); i++) {
        abs_devs[i] = fabs(dc_col_mags[i] - median_mag);
    }
    nth_element(abs_devs.begin(), abs_devs.begin() + mid, abs_devs.end());
    double mad = abs_devs[mid] * 1.4826;  // Scale to approximate std dev
    
    if (mad < 1e-10) mad = median_mag * 0.1;  // Fallback if MAD is tiny
    
    double thresh_mag = median_mag + threshold * mad;
    
    // Notch filter: suppress frequencies in DC column that exceed threshold
    // Use a soft notch (Gaussian suppression) to avoid ringing
    for (int fy = min_fy; fy < height - min_fy; fy++) {
        if (fy >= height / 2 - min_fy && fy <= height / 2 + min_fy) continue;  // Skip Nyquist region
        
        double re = freq[fy * freq_width + 0][0];
        double im = freq[fy * freq_width + 0][1];
        double mag = sqrt(re * re + im * im);
        
        if (mag > thresh_mag) {
            // Soft suppression: reduce to threshold level rather than zero
            double suppression = thresh_mag / mag;
            
            // Apply to DC column and a few adjacent columns (stripe width in freq domain)
            int notch_width = max(1, width / 256);
            for (int fx = 0; fx <= notch_width && fx < freq_width; fx++) {
                // Gaussian falloff with distance from DC
                double falloff = exp(-0.5 * (fx * fx) / (notch_width * 0.5 + 0.5));
                double factor = 1.0 - (1.0 - suppression) * falloff;
                
                freq[fy * freq_width + fx][0] *= factor;
                freq[fy * freq_width + fx][1] *= factor;
            }
            notch_count++;
        }
    }
    
    // Inverse FFT
    fftw_execute(plan_inv);
    
    // Normalize and copy back (FFTW doesn't normalize)
    double norm = 1.0 / pixels;
    for (size_t i = 0; i < pixels; i++) {
        band_data[i] = (float)(spatial[i] * norm);
    }
    
    // Cleanup
    fftw_destroy_plan(plan_fwd);
    fftw_destroy_plan(plan_inv);
    fftw_free(spatial);
    fftw_free(freq);
    
    return notch_count;
}

/**
 * Apply FFT-based destriping to all bands
 */
void destripe_image(float* data, int bands, int width, int height, 
                    float threshold, bool quiet) {
    
    size_t pixels = (size_t)width * height;
    int total_notches = 0;
    
    // Initialize FFTW threads
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());
    
    for (int b = 0; b < bands; b++) {
        int notches = destripe_band_fft(&data[b * pixels], width, height, threshold, quiet);
        total_notches += notches;
        
        if (!quiet && notches > 0) {
            cout << "  Band " << (b + 1) << ": notched " << notches << " stripe frequencies" << endl;
        }
    }
    
    fftw_cleanup_threads();
    
    if (!quiet) {
        cout << "  Total stripe frequencies removed: " << total_notches << endl;
    }
}

// ============================================================================
// GDAL I/O Functions
// ============================================================================

/**
 * Read raster metadata without loading data
 */
bool get_raster_info(const string& filename, int& bands, int& width, int& height) {
    GDALDataset* dataset = (GDALDataset*)GDALOpen(filename.c_str(), GA_ReadOnly);
    if (!dataset) {
        cerr << "Error: Cannot open " << filename << endl;
        return false;
    }

    width = dataset->GetRasterXSize();
    height = dataset->GetRasterYSize();
    bands = dataset->GetRasterCount();

    GDALClose(dataset);
    return true;
}

/**
 * Read a block of rows from raster
 * Returns data in [bands x (width * block_height)] layout
 */
bool read_raster_block(GDALDataset* dataset, int start_row, int num_rows,
                       float* data, int bands, int width) {
    int actual_rows = num_rows;
    int height = dataset->GetRasterYSize();
    if (start_row + num_rows > height) {
        actual_rows = height - start_row;
    }

    int pixels = width * actual_rows;

    #pragma omp parallel for
    for (int b = 0; b < bands; b++) {
        GDALRasterBand* band = dataset->GetRasterBand(b + 1);
        #pragma omp critical
        {
            band->RasterIO(GF_Read, 0, start_row, width, actual_rows,
                          &data[b * pixels], width, actual_rows,
                          GDT_Float32, 0, 0);
        }
    }

    return true;
}

/**
 * Read entire raster into memory
 * Returns data in [bands x pixels] layout
 */
float* read_raster(const string& filename, int& bands, int& width, int& height, bool quiet = false) {
    GDALDataset* dataset = (GDALDataset*)GDALOpen(filename.c_str(), GA_ReadOnly);
    if (!dataset) {
        cerr << "Error: Cannot open " << filename << endl;
        return nullptr;
    }

    width = dataset->GetRasterXSize();
    height = dataset->GetRasterYSize();
    bands = dataset->GetRasterCount();
    size_t pixels = (size_t)width * height;

    if (!quiet) {
        cout << "Reading: " << filename << endl;
        cout << "  Dimensions: " << width << " x " << height << " x " << bands << " bands" << endl;
        cout << "  Data size: " << fixed << setprecision(2)
             << (bands * pixels * sizeof(float)) / (1024.0 * 1024.0) << " MB" << endl;
    }

    float* data = (float*)aligned_alloc(64, bands * pixels * sizeof(float));
    if (!data) {
        cerr << "Error: Failed to allocate memory for raster data" << endl;
        GDALClose(dataset);
        return nullptr;
    }

    // Read bands in parallel
    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < bands; b++) {
        GDALRasterBand* band = dataset->GetRasterBand(b + 1);
        band->RasterIO(GF_Read, 0, 0, width, height,
                      &data[b * pixels], width, height,
                      GDT_Float32, 0, 0);
    }

    GDALClose(dataset);
    return data;
}

/**
 * Write raster data using GDAL (GeoTIFF format)
 */
bool write_raster(const string& filename, const string& src_filename,
                  const float* data, int bands, int width, int height,
                  const double* eigenvalues = nullptr, bool quiet = false) {

    GDALDataset* src_dataset = nullptr;
    if (!src_filename.empty()) {
        src_dataset = (GDALDataset*)GDALOpen(src_filename.c_str(), GA_ReadOnly);
    }

    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (!driver) {
        cerr << "Error: GTiff driver not available" << endl;
        if (src_dataset) GDALClose(src_dataset);
        return false;
    }

    char** options = nullptr;
    options = CSLSetNameValue(options, "COMPRESS", "LZW");
    options = CSLSetNameValue(options, "TILED", "YES");
    options = CSLSetNameValue(options, "BIGTIFF", "IF_SAFER");
    options = CSLSetNameValue(options, "NUM_THREADS", "ALL_CPUS");

    GDALDataset* dst_dataset = driver->Create(filename.c_str(), width, height, bands,
                                               GDT_Float32, options);
    CSLDestroy(options);

    if (!dst_dataset) {
        cerr << "Error: Cannot create " << filename << endl;
        if (src_dataset) GDALClose(src_dataset);
        return false;
    }

    // Copy georeferencing
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

    size_t pixels = (size_t)width * height;

    // Write bands in parallel
    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < bands; b++) {
        GDALRasterBand* band = dst_dataset->GetRasterBand(b + 1);
        band->RasterIO(GF_Write, 0, 0, width, height,
                      (void*)&data[b * pixels], width, height,
                      GDT_Float32, 0, 0);

        if (eigenvalues) {
            char desc[64];
            snprintf(desc, sizeof(desc), "MNF Band %d (SNR: %.4f)", b + 1, eigenvalues[b]);
            band->SetDescription(desc);
        }
    }

    if (!quiet) {
        cout << "Written: " << filename << endl;
        cout << "  Dimensions: " << width << " x " << height << " x " << bands << " bands" << endl;
    }

    GDALClose(dst_dataset);
    if (src_dataset) GDALClose(src_dataset);

    return true;
}

/**
 * Write eigenvalues to text file
 */
void write_eigenvalues(const string& base_filename, const VectorXd& eigenvalues) {
    string filename = base_filename + "_eigenvalues.txt";
    ofstream fp(filename);
    if (!fp) {
        cerr << "Warning: Cannot write eigenvalues to " << filename << endl;
        return;
    }

    fp << "# MNF Eigenvalues (SNR) - Descending order\n";
    fp << "# Band\tEigenvalue\n";
    fp << fixed << setprecision(6);
    for (int i = 0; i < eigenvalues.size(); i++) {
        fp << (i + 1) << "\t" << eigenvalues(i) << "\n";
    }

    cout << "Eigenvalues written to: " << filename << endl;
}

/**
 * Write ENVI .hdr file with georeferencing from source
 */
bool write_envi_header(const string& hdr_filename, const string& src_filename,
                       int bands, int width, int height,
                       const double* eigenvalues = nullptr, bool quiet = false) {
    
    ofstream hdr(hdr_filename);
    if (!hdr) {
        cerr << "Error: Cannot create ENVI header " << hdr_filename << endl;
        return false;
    }

    // Get georeferencing info from source file
    double geotransform[6] = {0, 1, 0, 0, 0, -1};
    string projection_wkt;
    
    GDALDataset* src_dataset = nullptr;
    if (!src_filename.empty()) {
        src_dataset = (GDALDataset*)GDALOpen(src_filename.c_str(), GA_ReadOnly);
        if (src_dataset) {
            if (src_dataset->GetGeoTransform(geotransform) != CE_None) {
                // Reset to default if failed
                geotransform[0] = 0; geotransform[1] = 1; geotransform[2] = 0;
                geotransform[3] = 0; geotransform[4] = 0; geotransform[5] = -1;
            }
            
            const char* proj = src_dataset->GetProjectionRef();
            if (proj && strlen(proj) > 0) {
                projection_wkt = proj;
            }
            GDALClose(src_dataset);
        }
    }

    // Parse projection to determine type
    OGRSpatialReference srs;
    string proj_name = "Arbitrary";
    int utm_zone = 0;
    int is_north = 1;
    string datum_name = "Unknown";
    string units_name = "meters";
    
    if (!projection_wkt.empty()) {
        srs.importFromWkt(projection_wkt.c_str());
        
        // Get projection name
        if (srs.IsGeographic()) {
            proj_name = "Geographic Lat/Lon";
            units_name = "degrees";
        } else if (srs.IsProjected()) {
            const char* proj_type = srs.GetAttrValue("PROJECTION");
            utm_zone = srs.GetUTMZone(&is_north);
            if (utm_zone != 0) {
                proj_name = "UTM";
            } else if (proj_type) {
                proj_name = proj_type;
            }
        }
        
        // Get datum
        const char* datum = srs.GetAttrValue("DATUM");
        if (datum) {
            datum_name = datum;
            // Simplify common datum names for ENVI
            if (strstr(datum, "WGS") && strstr(datum, "84")) {
                datum_name = "WGS-84";
            } else if (strstr(datum, "NAD") && strstr(datum, "83")) {
                datum_name = "North America 1983";
            } else if (strstr(datum, "NAD") && strstr(datum, "27")) {
                datum_name = "North America 1927";
            }
        }
    }
    
    // Write ENVI header
    hdr << "ENVI\n";
    hdr << "description = {MNF Transform Output}\n";
    hdr << "samples = " << width << "\n";
    hdr << "lines = " << height << "\n";
    hdr << "bands = " << bands << "\n";
    hdr << "header offset = 0\n";
    hdr << "file type = ENVI Standard\n";
    hdr << "data type = 4\n";  // 4 = 32-bit float
    hdr << "interleave = bsq\n";
    hdr << "byte order = 0\n";  // Little endian
    
    // Write map info
    // ENVI map info format: {projection, ref_x, ref_y, easting, northing, x_size, y_size, zone, North/South, datum, units=...}
    // ref_x, ref_y are the pixel coordinates (1-based) of the tie point
    double ulx = geotransform[0];
    double uly = geotransform[3];
    double pixel_size_x = fabs(geotransform[1]);
    double pixel_size_y = fabs(geotransform[5]);
    
    hdr << "map info = {" << proj_name << ", 1, 1, "
        << fixed << setprecision(10) << ulx << ", " << uly << ", "
        << setprecision(10) << pixel_size_x << ", " << pixel_size_y;
    
    if (proj_name == "UTM" && utm_zone != 0) {
        hdr << ", " << utm_zone << ", " << (is_north ? "North" : "South");
    }
    
    hdr << ", " << datum_name << ", units=" << units_name << "}\n";
    
    // Write coordinate system string (full WKT)
    if (!projection_wkt.empty()) {
        hdr << "coordinate system string = {" << projection_wkt << "}\n";
    }
    
    // Write band names with eigenvalues if provided
    hdr << "band names = {";
    for (int i = 0; i < bands; i++) {
        if (i > 0) hdr << ", ";
        hdr << "MNF Band " << (i + 1);
        if (eigenvalues) {
            hdr << " (SNR: " << fixed << setprecision(4) << eigenvalues[i] << ")";
        }
    }
    hdr << "}\n";
    
    hdr.close();
    
    if (!quiet) {
        cout << "ENVI header written to: " << hdr_filename << endl;
    }
    
    return true;
}

/**
 * Write ENVI binary file with header
 * The data file uses the provided filename, header is filename with .hdr extension
 */
bool write_envi_raster(const string& filename, const string& src_filename,
                       const float* data, int bands, int width, int height,
                       const double* eigenvalues = nullptr, bool quiet = false) {
    
    size_t pixels = (size_t)width * height;
    
    // Data file is the provided filename
    string data_filename = filename;
    
    // Header file: replace extension with .hdr or append .hdr
    string hdr_filename;
    size_t dot_pos = filename.rfind('.');
    if (dot_pos != string::npos) {
        hdr_filename = filename.substr(0, dot_pos) + ".hdr";
    } else {
        hdr_filename = filename + ".hdr";
    }
    
    // Write binary data file (BSQ format - band sequential)
    ofstream data_file(data_filename, ios::binary);
    if (!data_file) {
        cerr << "Error: Cannot create ENVI data file " << data_filename << endl;
        return false;
    }
    
    // Write data in BSQ format (already in this layout)
    data_file.write(reinterpret_cast<const char*>(data), 
                    bands * pixels * sizeof(float));
    
    if (!data_file) {
        cerr << "Error: Failed to write ENVI data to " << data_filename << endl;
        return false;
    }
    data_file.close();
    
    if (!quiet) {
        cout << "ENVI data written to: " << data_filename << endl;
        cout << "  Dimensions: " << width << " x " << height << " x " << bands << " bands" << endl;
        cout << "  File size: " << fixed << setprecision(2) 
             << (bands * pixels * sizeof(float)) / (1024.0 * 1024.0) << " MB" << endl;
    }
    
    // Write header file
    if (!write_envi_header(hdr_filename, src_filename, bands, width, height, 
                           eigenvalues, quiet)) {
        return false;
    }
    
    return true;
}

// ============================================================================
// MNF Transform Implementation
// ============================================================================

/**
 * Compute band means in parallel
 */
VectorXd compute_means(const float* data, int bands, size_t pixels) {
    VectorXd means = VectorXd::Zero(bands);

    #pragma omp parallel
    {
        VectorXd local_means = VectorXd::Zero(bands);

        #pragma omp for nowait
        for (size_t p = 0; p < pixels; p++) {
            for (int b = 0; b < bands; b++) {
                local_means(b) += data[b * pixels + p];
            }
        }

        #pragma omp critical
        means += local_means;
    }

    means /= (double)pixels;
    return means;
}

/**
 * Center data in place (subtract means)
 */
void center_data(float* data, const VectorXd& means, int bands, size_t pixels) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < bands; b++) {
        for (size_t p = 0; p < pixels; p++) {
            data[b * pixels + p] -= (float)means(b);
        }
    }
}

/**
 * Compute data covariance matrix using parallel accumulation
 * Σ_data = (1/n) * Σ (x_i - μ)(x_i - μ)^T
 */
MatrixXd compute_covariance(const float* data, int bands, size_t pixels) {
    MatrixXd cov = MatrixXd::Zero(bands, bands);

    int num_threads = omp_get_max_threads();
    vector<MatrixXd> local_covs(num_threads, MatrixXd::Zero(bands, bands));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        MatrixXd& local_cov = local_covs[tid];
        VectorXd pixel_vec(bands);

        #pragma omp for nowait schedule(static, 4096)
        for (size_t p = 0; p < pixels; p++) {
            for (int b = 0; b < bands; b++) {
                pixel_vec(b) = data[b * pixels + p];
            }
            local_cov.noalias() += pixel_vec * pixel_vec.transpose();
        }
    }

    // Combine thread-local results
    for (int t = 0; t < num_threads; t++) {
        cov += local_covs[t];
    }

    cov /= (double)pixels;
    return cov;
}

/**
 * Compute noise covariance using directional max approach
 * 
 * Estimates noise separately from horizontal and vertical differences,
 * then takes the element-wise maximum. This ensures that:
 * - Horizontal stripes are captured by vertical differences
 * - Vertical stripes are captured by horizontal differences
 * 
 * This is more robust to detector striping common in pushbroom sensors.
 */
MatrixXd compute_noise_covariance_directional_max(const float* data, 
                                                   int bands, int width, int height) {
    size_t pixels = (size_t)width * height;
    
    MatrixXd cov_h = MatrixXd::Zero(bands, bands);
    MatrixXd cov_v = MatrixXd::Zero(bands, bands);
    
    int num_threads = omp_get_max_threads();
    vector<MatrixXd> local_covs_h(num_threads, MatrixXd::Zero(bands, bands));
    vector<MatrixXd> local_covs_v(num_threads, MatrixXd::Zero(bands, bands));
    
    // Horizontal differences (captures vertical stripes)
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        MatrixXd& local_cov = local_covs_h[tid];
        VectorXd noise_vec(bands);
        
        #pragma omp for nowait schedule(static, 256)
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width - 1; col++) {
                size_t p1 = row * width + col;
                size_t p2 = p1 + 1;
                
                for (int b = 0; b < bands; b++) {
                    noise_vec(b) = (data[b * pixels + p2] - data[b * pixels + p1]) * 0.5;
                }
                local_cov.noalias() += noise_vec * noise_vec.transpose();
            }
        }
    }
    
    // Vertical differences (captures horizontal stripes)
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        MatrixXd& local_cov = local_covs_v[tid];
        VectorXd noise_vec(bands);
        
        #pragma omp for nowait schedule(static, 256)
        for (int row = 0; row < height - 1; row++) {
            for (int col = 0; col < width; col++) {
                size_t p1 = row * width + col;
                size_t p2 = p1 + width;
                
                for (int b = 0; b < bands; b++) {
                    noise_vec(b) = (data[b * pixels + p2] - data[b * pixels + p1]) * 0.5;
                }
                local_cov.noalias() += noise_vec * noise_vec.transpose();
            }
        }
    }
    
    // Combine thread-local results
    for (int t = 0; t < num_threads; t++) {
        cov_h += local_covs_h[t];
        cov_v += local_covs_v[t];
    }
    
    cov_h /= (double)((width - 1) * height);
    cov_v /= (double)(width * (height - 1));
    
    // Take element-wise maximum - this ensures stripes in either direction
    // contribute to the noise estimate
    MatrixXd cov = cov_h.cwiseMax(cov_v);
    
    return cov;
}

/**
 * Compute noise covariance using median-based local residual method
 * 
 * For each pixel, noise is estimated as the difference between the pixel value
 * and the median of its local neighborhood. Median is more robust than mean to:
 * - Outliers and edges
 * - Detector striping
 * - Band misregistration  
 * - Mixed land cover within window
 *
 * Window size should be odd (e.g., 3, 5, 7, 9, 11)
 *
 * Σ_noise = (1/n) * Σ residual_i * residual_i^T
 */
MatrixXd compute_noise_covariance_median(const float* data, int bands, int width, int height, int window_size) {
    size_t pixels = (size_t)width * height;
    
    // Ensure window size is odd
    if (window_size % 2 == 0) window_size++;
    int half_win = window_size / 2;
    
    // We skip border pixels to have full windows
    int valid_width = width - 2 * half_win;
    int valid_height = height - 2 * half_win;
    
    if (valid_width <= 0 || valid_height <= 0) {
        cerr << "Error: Image too small for window size " << window_size << endl;
        return MatrixXd::Identity(bands, bands);
    }
    
    size_t valid_count = (size_t)valid_width * valid_height;
    int win_pixels = window_size * window_size;

    MatrixXd cov = MatrixXd::Zero(bands, bands);

    int num_threads = omp_get_max_threads();
    vector<MatrixXd> local_covs(num_threads, MatrixXd::Zero(bands, bands));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        MatrixXd& local_cov = local_covs[tid];
        VectorXd residual_vec(bands);
        vector<float> window_vals(win_pixels);

        #pragma omp for nowait schedule(static, 64)
        for (int row = half_win; row < height - half_win; row++) {
            for (int col = half_win; col < width - half_win; col++) {
                size_t center = row * width + col;
                
                // Compute residual for each band
                for (int b = 0; b < bands; b++) {
                    size_t band_offset = b * pixels;
                    
                    // Get center pixel value
                    float center_val = data[band_offset + center];
                    
                    // Gather window values
                    int idx = 0;
                    for (int wy = -half_win; wy <= half_win; wy++) {
                        for (int wx = -half_win; wx <= half_win; wx++) {
                            window_vals[idx++] = data[band_offset + (row + wy) * width + (col + wx)];
                        }
                    }
                    
                    // Find median using nth_element (partial sort)
                    size_t mid = win_pixels / 2;
                    nth_element(window_vals.begin(), window_vals.begin() + mid, window_vals.end());
                    float median_val = window_vals[mid];
                    
                    // Residual is difference from local median
                    residual_vec(b) = (center_val - median_val);
                }
                
                local_cov.noalias() += residual_vec * residual_vec.transpose();
            }
        }
    }

    // Combine thread-local results
    for (int t = 0; t < num_threads; t++) {
        cov += local_covs[t];
    }

    cov /= (double)valid_count;
    return cov;
}

/**
 * Main noise covariance function - combines directional max with median
 * to handle both striping and general noise robustly
 */
MatrixXd compute_noise_covariance(const float* data, int bands, int width, int height, int window_size) {
    // Get directional estimate (handles striping)
    MatrixXd cov_dir = compute_noise_covariance_directional_max(data, bands, width, height);
    
    // Get median-based estimate (handles edges/outliers)
    MatrixXd cov_med = compute_noise_covariance_median(data, bands, width, height, window_size);
    
    // Take element-wise maximum of both approaches
    // This ensures we capture noise from all sources
    MatrixXd cov = cov_dir.cwiseMax(cov_med);
    
    return cov;
}

/**
 * Solve generalized eigenvalue problem: Σ_data * v = λ * Σ_noise * v
 * Returns eigenvalues and eigenvectors sorted by descending eigenvalue (SNR)
 */
bool solve_generalized_eigen(const MatrixXd& cov_data, const MatrixXd& cov_noise,
                             VectorXd& eigenvalues, MatrixXd& eigenvectors) {
    int n = cov_data.rows();

    // Use Cholesky decomposition of noise covariance
    // Σ_noise = L * L^T
    // Transform to standard eigenvalue problem: L^-1 * Σ_data * L^-T * y = λ * y
    // Then v = L^-T * y

    LLT<MatrixXd> llt(cov_noise);
    if (llt.info() != Success) {
        // Fall back: add small regularization
        MatrixXd cov_noise_reg = cov_noise + 1e-6 * MatrixXd::Identity(n, n);
        llt.compute(cov_noise_reg);
        if (llt.info() != Success) {
            cerr << "Error: Noise covariance matrix is not positive definite" << endl;
            return false;
        }
    }

    MatrixXd L = llt.matrixL();
    MatrixXd L_inv = L.inverse();

    // Form the transformed matrix: L^-1 * Σ_data * L^-T
    MatrixXd A = L_inv * cov_data * L_inv.transpose();

    // Solve standard symmetric eigenvalue problem
    SelfAdjointEigenSolver<MatrixXd> solver(A);
    if (solver.info() != Success) {
        cerr << "Error: Eigenvalue decomposition failed" << endl;
        return false;
    }

    // Get eigenvalues and eigenvectors
    VectorXd evals = solver.eigenvalues();
    MatrixXd evecs = solver.eigenvectors();

    // Transform eigenvectors back: v = L^-T * y
    evecs = L_inv.transpose() * evecs;

    // Sort by descending eigenvalue
    vector<int> indices(n);
    for (int i = 0; i < n; i++) indices[i] = i;
    sort(indices.begin(), indices.end(), [&evals](int a, int b) {
        return evals(a) > evals(b);
    });

    eigenvalues.resize(n);
    eigenvectors.resize(n, n);
    for (int i = 0; i < n; i++) {
        eigenvalues(i) = evals(indices[i]);
        eigenvectors.col(i) = evecs.col(indices[i]);
    }

    return true;
}

/**
 * Project data onto MNF components
 * output = V^T * centered_data
 */
void project_data(const float* input, float* output,
                  const MatrixXd& eigenvectors, int bands, size_t pixels,
                  int n_components) {

    // Convert eigenvectors to float for faster computation
    MatrixXf V = eigenvectors.leftCols(n_components).cast<float>();

    #pragma omp parallel
    {
        VectorXf pixel_in(bands);
        VectorXf pixel_out(n_components);

        #pragma omp for schedule(static, 4096)
        for (size_t p = 0; p < pixels; p++) {
            for (int b = 0; b < bands; b++) {
                pixel_in(b) = input[b * pixels + p];
            }
            pixel_out.noalias() = V.transpose() * pixel_in;
            for (int c = 0; c < n_components; c++) {
                output[c * pixels + p] = pixel_out(c);
            }
        }
    }
}

/**
 * Inverse project: reconstruct from MNF components
 * output = V * mnf_data + means
 */
void inverse_project_data(const float* input, float* output,
                          const MatrixXd& eigenvectors, const VectorXd& means,
                          int bands, size_t pixels, int n_components) {

    MatrixXf V = eigenvectors.leftCols(n_components).cast<float>();
    VectorXf means_f = means.cast<float>();

    #pragma omp parallel
    {
        VectorXf pixel_in(n_components);
        VectorXf pixel_out(bands);

        #pragma omp for schedule(static, 4096)
        for (size_t p = 0; p < pixels; p++) {
            for (int c = 0; c < n_components; c++) {
                pixel_in(c) = input[c * pixels + p];
            }
            pixel_out.noalias() = V * pixel_in + means_f;
            for (int b = 0; b < bands; b++) {
                output[b * pixels + p] = pixel_out(b);
            }
        }
    }
}

// ============================================================================
// Main MNF Transform
// ============================================================================

struct MNFResult {
    VectorXd eigenvalues;
    MatrixXd eigenvectors;
    VectorXd means;
    bool success = false;
};

MNFResult mnf_transform(float* data, int bands, int width, int height,
                        int n_components, int noise_window, bool quiet = false) {
    MNFResult result;
    size_t pixels = (size_t)width * height;

    if (n_components <= 0 || n_components > bands) {
        n_components = bands;
    }

    // Step 1: Compute means
    {
        Timer t("Computing band means", quiet);
        result.means = compute_means(data, bands, pixels);
    }

    // Step 2: Center data
    {
        Timer t("Centering data", quiet);
        center_data(data, result.means, bands, pixels);
    }

    // Step 3: Compute noise covariance
    MatrixXd cov_noise;
    {
        Timer t("Computing noise covariance (median, " + to_string(noise_window) + "x" + to_string(noise_window) + ")", quiet);
        cov_noise = compute_noise_covariance(data, bands, width, height, noise_window);
    }

    // Step 4: Compute data covariance
    MatrixXd cov_data;
    {
        Timer t("Computing data covariance", quiet);
        cov_data = compute_covariance(data, bands, pixels);
    }

    // Step 5: Solve generalized eigenvalue problem
    {
        Timer t("Solving eigenvalue problem", quiet);
        if (!solve_generalized_eigen(cov_data, cov_noise,
                                     result.eigenvalues, result.eigenvectors)) {
            return result;
        }
    }

    // Step 6: Project data
    float* output = (float*)aligned_alloc(64, n_components * pixels * sizeof(float));
    {
        Timer t("Projecting data", quiet);
        project_data(data, output, result.eigenvectors, bands, pixels, n_components);
    }

    // Copy output back (only n_components bands)
    memcpy(data, output, n_components * pixels * sizeof(float));
    free(output);

    result.success = true;
    return result;
}

// ============================================================================
// Main Program
// ============================================================================

void print_usage(const char* program) {
    cout << "MNF (Minimum Noise Fraction) Transform - CPU Parallel Implementation\n\n";
    cout << "Usage: " << program << " <input_raster> <output_raster> [options]\n\n";
    cout << "Options:\n";
    cout << "  -n <num>       Number of components to output (default: all)\n";
    cout << "  -t <threads>   Number of threads (default: auto)\n";
    cout << "  --nwin <size>  Noise estimation window size, must be odd (default: 9)\n";
    cout << "  --destripe     Apply FFT-based destriping before MNF transform\n";
    cout << "  --stripe-thresh <val>  Stripe detection threshold in std devs (default: 3.0)\n";
    cout << "  -i             Also compute inverse transform (reconstruction)\n";
    cout << "  -q             Quiet mode (minimal output)\n";
    cout << "\n";
    cout << "Examples:\n";
    cout << "  " << program << " input.tif mnf_output.tif\n";
    cout << "  " << program << " input.tif mnf_output.tif -n 20\n";
    cout << "  " << program << " hyperspectral.img mnf.tif -n 50 -t 32\n";
    cout << "  " << program << " sentinel2.bin mnf.bin --nwin 15\n";
    cout << "  " << program << " sentinel2.bin mnf.bin --destripe --stripe-thresh 2.5\n";
    cout << "\n";
    cout << "Supported formats: Any GDAL-readable raster (GeoTIFF, ENVI, HDF, etc.)\n";
    cout << "Output format: ENVI binary (BSQ, float32) with .hdr sidecar file\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    MNFConfig config;
    config.input_file = argv[1];
    config.output_file = argv[2];

    // Parse arguments
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            config.n_components = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            config.num_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--nwin") == 0 && i + 1 < argc) {
            config.noise_window = atoi(argv[++i]);
            if (config.noise_window < 3) config.noise_window = 3;
            if (config.noise_window % 2 == 0) config.noise_window++;  // Ensure odd
        } else if (strcmp(argv[i], "--destripe") == 0) {
            config.destripe = true;
        } else if (strcmp(argv[i], "--stripe-thresh") == 0 && i + 1 < argc) {
            config.stripe_threshold = atof(argv[++i]);
            if (config.stripe_threshold < 0.5f) config.stripe_threshold = 0.5f;
        } else if (strcmp(argv[i], "-i") == 0) {
            config.do_inverse = true;
        } else if (strcmp(argv[i], "-q") == 0) {
            config.quiet = true;
        } else {
            cerr << "Unknown option: " << argv[i] << endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // Set number of threads
    if (config.num_threads > 0) {
        omp_set_num_threads(config.num_threads);
    }

    if (!config.quiet) {
        cout << "Using " << omp_get_max_threads() << " threads\n" << endl;
    }

    // Initialize GDAL
    GDALAllRegister();

    // Read input
    int bands, width, height;
    float* data = read_raster(config.input_file, bands, width, height, config.quiet);
    if (!data) {
        return 1;
    }

    size_t pixels = (size_t)width * height;

    // Validate n_components
    if (config.n_components <= 0 || config.n_components > bands) {
        config.n_components = bands;
    }

    if (!config.quiet) {
        cout << "\nMNF Transform Configuration:" << endl;
        cout << "  Input bands: " << bands << endl;
        cout << "  Output components: " << config.n_components << endl;
        cout << "  Image size: " << width << " x " << height
             << " (" << pixels << " pixels)" << endl;
        cout << "  Noise window: " << config.noise_window << "x" << config.noise_window 
             << " (median-based)" << endl;
        if (config.destripe) {
            cout << "  Destriping: enabled (threshold: " << config.stripe_threshold << " std devs)" << endl;
        }
        cout << endl;
    }

    // Keep a copy of original data for inverse transform
    float* original_data = nullptr;
    VectorXd original_means;
    if (config.do_inverse) {
        original_data = (float*)aligned_alloc(64, bands * pixels * sizeof(float));
        memcpy(original_data, data, bands * pixels * sizeof(float));
    }

    // Apply destriping if requested
    if (config.destripe) {
        Timer t("Applying FFT destriping", config.quiet);
        if (!config.quiet) cout << endl;
        destripe_image(data, bands, width, height, config.stripe_threshold, config.quiet);
    }

    // Run MNF transform
    auto start_time = chrono::high_resolution_clock::now();

    MNFResult result = mnf_transform(data, bands, width, height,
                                      config.n_components, config.noise_window, config.quiet);

    auto end_time = chrono::high_resolution_clock::now();
    double total_ms = chrono::duration<double, milli>(end_time - start_time).count();

    if (!result.success) {
        cerr << "Error: MNF transform failed" << endl;
        free(data);
        if (original_data) free(original_data);
        return 1;
    }

    if (!config.quiet) {
        cout << "\nTotal MNF transform time: " << fixed << setprecision(1)
             << total_ms << " ms" << endl;

        cout << "\nEigenvalues (SNR) summary:" << endl;
        int show_top = min(10, bands);
        for (int i = 0; i < show_top; i++) {
            cout << "  Band " << setw(3) << (i + 1) << ": "
                 << fixed << setprecision(4) << result.eigenvalues(i) << endl;
        }
        if (bands > 10) {
            cout << "  ..." << endl;
            for (int i = bands - 3; i < bands; i++) {
                cout << "  Band " << setw(3) << (i + 1) << ": "
                     << fixed << setprecision(4) << result.eigenvalues(i) << endl;
            }
        }
        cout << endl;
    }

    // Convert eigenvalues for GDAL
    vector<double> evals(bands);
    for (int i = 0; i < bands; i++) {
        evals[i] = result.eigenvalues(i);
    }

    // Determine base name for auxiliary files
    string base_name = config.output_file;
    size_t dot_pos = base_name.rfind('.');
    if (dot_pos != string::npos) {
        base_name = base_name.substr(0, dot_pos);
    }

    // Write output as ENVI binary + header
    if (!write_envi_raster(config.output_file, config.input_file, data,
                           config.n_components, width, height, evals.data(), config.quiet)) {
        free(data);
        if (original_data) free(original_data);
        return 1;
    }

    // Write eigenvalues
    write_eigenvalues(base_name, result.eigenvalues);

    // Inverse transform
    if (config.do_inverse && original_data) {
        if (!config.quiet) {
            cout << "\nComputing inverse transform..." << endl;
        }

        // Compute original means for reconstruction
        original_means = compute_means(original_data, bands, pixels);

        // Allocate for reconstructed data
        float* recon_data = (float*)aligned_alloc(64, bands * pixels * sizeof(float));

        // Inverse project
        inverse_project_data(data, recon_data, result.eigenvectors, original_means,
                            bands, pixels, config.n_components);

        // Compute reconstruction error
        double mse = 0.0;
        #pragma omp parallel for reduction(+:mse)
        for (size_t i = 0; i < (size_t)bands * pixels; i++) {
            double diff = original_data[i] - recon_data[i];
            mse += diff * diff;
        }
        mse /= (double)(bands * pixels);

        cout << "Reconstruction RMSE (" << config.n_components << " components): "
             << fixed << setprecision(6) << sqrt(mse) << endl;

        // Write reconstructed image as ENVI binary + header
        string recon_base = base_name + "_reconstructed";
        write_envi_raster(recon_base + ".bin", config.input_file, recon_data, bands, width, height,
                          nullptr, config.quiet);

        free(recon_data);
    }

    // Cleanup
    free(data);
    if (original_data) free(original_data);

    if (!config.quiet) {
        cout << "\nDone." << endl;
    }

    return 0;
}


