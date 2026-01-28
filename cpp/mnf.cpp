/** 2026012*
 * Minimum Noise Fraction (MNF) Transform - CPU Implementation
 * Parallelized with OpenMP, using Eigen for linear algebra
 * Designed for large datasets that exceed GPU memory
 *
 * MNF transform steps:
 * 1. Estimate noise covariance matrix from spatial differences
 * 2. Compute data covariance matrix
 * 3. Solve generalized eigenvalue problem: Σ_data * v = λ * Σ_noise * v
 * 4. Project data onto eigenvectors sorted by decreasing SNR (eigenvalue)
 *
 * Usage: mnf_transform <input_raster> <output_raster> [options]
 *
 * Compile (choose one based on your Eigen installation):
 *   # If Eigen is in /usr/include/eigen3:
    g++ -O3 -march=native -fopenmp -DNDEBUG mnf_transform_cpu.cpp -o mnf_transform  $(gdal-config --cflags) $(gdal-config --libs)
 *
 *   # If Eigen is elsewhere, specify the path:
 *   g++ -O3 -march=native -fopenmp -DNDEBUG mnf_transform_cpu.cpp -o mnf_transform \
 *       $(gdal-config --cflags) $(gdal-config --libs) -I/path/to/eigen
 *
 *   # On Ubuntu/Debian: sudo apt install libeigen3-dev
 *   # On RHEL/Fedora:   sudo dnf install eigen3-devel
 *   # On macOS:         brew install eigen
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
 * Compute noise covariance using local residual method
 * 
 * For each pixel, noise is estimated as the difference between the pixel value
 * and the mean of its local neighborhood (3x3 window). This is more robust to:
 * - Detector striping
 * - Band misregistration  
 * - Edge effects
 * - Anisotropic noise patterns
 *
 * Σ_noise = (1/n) * Σ residual_i * residual_i^T
 */
MatrixXd compute_noise_covariance(const float* data, int bands, int width, int height) {
    size_t pixels = (size_t)width * height;
    
    // We skip 1-pixel border to have full 3x3 windows
    int valid_width = width - 2;
    int valid_height = height - 2;
    size_t valid_count = (size_t)valid_width * valid_height;

    MatrixXd cov = MatrixXd::Zero(bands, bands);

    int num_threads = omp_get_max_threads();
    vector<MatrixXd> local_covs(num_threads, MatrixXd::Zero(bands, bands));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        MatrixXd& local_cov = local_covs[tid];
        VectorXd residual_vec(bands);

        #pragma omp for nowait schedule(static, 256)
        for (int row = 1; row < height - 1; row++) {
            for (int col = 1; col < width - 1; col++) {
                size_t center = row * width + col;
                
                // Compute residual for each band
                for (int b = 0; b < bands; b++) {
                    size_t band_offset = b * pixels;
                    
                    // Get center pixel value
                    float center_val = data[band_offset + center];
                    
                    // Compute mean of 3x3 neighborhood (excluding center)
                    float neighbor_sum = 
                        data[band_offset + (row-1)*width + (col-1)] +
                        data[band_offset + (row-1)*width + (col  )] +
                        data[band_offset + (row-1)*width + (col+1)] +
                        data[band_offset + (row  )*width + (col-1)] +
                        // center excluded
                        data[band_offset + (row  )*width + (col+1)] +
                        data[band_offset + (row+1)*width + (col-1)] +
                        data[band_offset + (row+1)*width + (col  )] +
                        data[band_offset + (row+1)*width + (col+1)];
                    
                    float neighbor_mean = neighbor_sum / 8.0f;
                    
                    // Residual is difference from local mean
                    // Scale factor sqrt(8/9) accounts for the variance inflation
                    // from using estimated mean rather than true mean
                    residual_vec(b) = (center_val - neighbor_mean) * 0.942809f; // sqrt(8/9)
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
                        int n_components, bool quiet = false) {
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
        Timer t("Computing noise covariance", quiet);
        cov_noise = compute_noise_covariance(data, bands, width, height);
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
    cout << "  -n <num>      Number of components to output (default: all)\n";
    cout << "  -t <threads>  Number of threads (default: auto)\n";
    cout << "  -i            Also compute inverse transform (reconstruction)\n";
    cout << "  -q            Quiet mode (minimal output)\n";
    cout << "\n";
    cout << "Examples:\n";
    cout << "  " << program << " input.tif mnf_output.tif\n";
    cout << "  " << program << " input.tif mnf_output.tif -n 20\n";
    cout << "  " << program << " hyperspectral.img mnf.tif -n 50 -t 32\n";
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
        cout << endl;
    }

    // Keep a copy of original data for inverse transform
    float* original_data = nullptr;
    VectorXd original_means;
    if (config.do_inverse) {
        original_data = (float*)aligned_alloc(64, bands * pixels * sizeof(float));
        memcpy(original_data, data, bands * pixels * sizeof(float));
    }

    // Run MNF transform
    auto start_time = chrono::high_resolution_clock::now();

    MNFResult result = mnf_transform(data, bands, width, height,
                                      config.n_components, config.quiet);

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

