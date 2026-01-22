/** 20260122
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
 * Compile:
 *   g++ -O3 -march=native -fopenmp -DNDEBUG mnf_transform_cpu.cpp -o mnf_transform \
 *       $(gdal-config --cflags) $(gdal-config --libs) -I/usr/include/eigen3
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <gdal.h>
#include <gdal_priv.h>
#include <cpl_conv.h>

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
 * Compute noise covariance using horizontal spatial differences
 * noise[b, p] = (data[b, p+1] - data[b, p]) * 0.5
 * Σ_noise = (1/n) * Σ noise_i * noise_i^T
 */
MatrixXd compute_noise_covariance(const float* data, int bands, int width, int height) {
    size_t pixels = (size_t)width * height;
    size_t noise_count = (size_t)(width - 1) * height;
    
    MatrixXd cov = MatrixXd::Zero(bands, bands);
    
    int num_threads = omp_get_max_threads();
    vector<MatrixXd> local_covs(num_threads, MatrixXd::Zero(bands, bands));
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        MatrixXd& local_cov = local_covs[tid];
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
    
    // Combine
    for (int t = 0; t < num_threads; t++) {
        cov += local_covs[t];
    }
    
    cov /= (double)noise_count;
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
    cout << "Output format: GeoTIFF with LZW compression\n";
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
    
    // Write output
    if (!write_raster(config.output_file, config.input_file, data,
                      config.n_components, width, height, evals.data(), config.quiet)) {
        free(data);
        if (original_data) free(original_data);
        return 1;
    }
    
    // Write eigenvalues
    string base_name = config.output_file;
    size_t dot_pos = base_name.rfind('.');
    if (dot_pos != string::npos) {
        base_name = base_name.substr(0, dot_pos);
    }
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
        
        // Write reconstructed image
        string recon_file = base_name + "_reconstructed.tif";
        write_raster(recon_file, config.input_file, recon_data, bands, width, height,
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



