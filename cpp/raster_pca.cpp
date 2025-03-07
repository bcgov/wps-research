/* 20250307 pca.c Principal Components Analysis for ENVI-format raster 

  sudo apt-get install liblapack-dev libblas-dev

  g++ raster_pca.cpp misc.cpp -o raster_pca -O3 -llapack -lblas -lm

  Used chatgpt to form original example. Used claude AI to modify to use a headerless (link only) method for accessing LAPACK
*/

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <getopt.h>

// External declarations for BLAS/LAPACK functions
extern "C" {
    // LAPACK functions
    extern int dgeev_(char*, char*, int*, double*, int*, double*, double*, double*, int*, double*, int*, double*, int*, int*);
    // CBLAS functions if needed
    extern double cblas_ddot(const int N, const double *X, const int incX, const double *Y, const int incY);
}

#define NUM_SAMPLES 5   // Number of samples (rows)
#define NUM_FEATURES 3  // Number of features (columns)

// Function to calculate the mean of each feature
void calculate_mean(double data[NUM_SAMPLES][NUM_FEATURES], double mean[NUM_FEATURES]) {
    for (int j = 0; j < NUM_FEATURES; j++) {
        mean[j] = 0;
        for (int i = 0; i < NUM_SAMPLES; i++) {
            mean[j] += data[i][j];
        }
        mean[j] /= NUM_SAMPLES;
    }
}

// Function to center the data (subtract the mean from each feature)
void center_data(double data[NUM_SAMPLES][NUM_FEATURES], double mean[NUM_FEATURES], double centered_data[NUM_SAMPLES][NUM_FEATURES]) {
    for (int i = 0; i < NUM_SAMPLES; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            centered_data[i][j] = data[i][j] - mean[j];
        }
    }
}

// Function to compute the covariance matrix
void compute_covariance_matrix(double centered_data[NUM_SAMPLES][NUM_FEATURES], double covariance_matrix[NUM_FEATURES][NUM_FEATURES]) {
    for (int i = 0; i < NUM_FEATURES; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            covariance_matrix[i][j] = 0;
            for (int k = 0; k < NUM_SAMPLES; k++) {
                covariance_matrix[i][j] += centered_data[k][i] * centered_data[k][j];
            }
            covariance_matrix[i][j] /= (NUM_SAMPLES - 1);  // Normalizing by N-1
        }
    }
}

// Function to compute eigenvalues and eigenvectors using LAPACK directly
void compute_eigenvalues_and_eigenvectors(double matrix[NUM_FEATURES][NUM_FEATURES], double eigenvectors[NUM_FEATURES][NUM_FEATURES], double eigenvalues[NUM_FEATURES], int num_features) {
    int N = num_features;
    char jobvl = 'V';  // Compute left eigenvectors
    char jobvr = 'N';  // Don't compute right eigenvectors
    int lda = N;
    double* wr = new double[N];  // Real parts of eigenvalues
    double* wi = new double[N];  // Imaginary parts of eigenvalues
    double* vl = &eigenvectors[0][0];  // Left eigenvectors
    int ldvl = N;
    double* vr = nullptr;  // We don't need right eigenvectors
    int ldvr = 1;
    int lwork = 4 * N;  // Work array size
    double* work = new double[lwork];
    int info;
    
    // Call LAPACK dgeev function directly
    dgeev_(&jobvl, &jobvr, &N, &matrix[0][0], &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, &info);
    
    if (info != 0) {
        std::cerr << "Error in eigenvalue decomposition: " << info << std::endl;
        exit(1);
    }
    
    // Copy eigenvalues to the result array
    for (int i = 0; i < N; i++) {
        eigenvalues[i] = wr[i];
        // If there are imaginary parts, just take the real part for simplicity in this PCA implementation
        if (std::abs(wi[i]) > 1e-10) {
            std::cerr << "Warning: Complex eigenvalue detected: " << wr[i] << " + " << wi[i] << "i" << std::endl;
        }
    }
    
    // Cleanup
    delete[] wr;
    delete[] wi;
    delete[] work;
}

// Function to sort eigenvalues and eigenvectors in descending order
void sort_eigenvectors(double eigenvalues[NUM_FEATURES], double eigenvectors[NUM_FEATURES][NUM_FEATURES], int num_features) {
    for (int i = 0; i < num_features - 1; i++) {
        for (int j = i + 1; j < num_features; j++) {
            if (eigenvalues[i] < eigenvalues[j]) {
                double temp_val = eigenvalues[i];
                eigenvalues[i] = eigenvalues[j];
                eigenvalues[j] = temp_val;

                for (int k = 0; k < num_features; k++) {
                    double temp_vec = eigenvectors[k][i];
                    eigenvectors[k][i] = eigenvectors[k][j];
                    eigenvectors[k][j] = temp_vec;
                }
            }
        }
    }
}

// Function to project the data onto the principal components
void project_data(double centered_data[NUM_SAMPLES][NUM_FEATURES], double eigenvectors[NUM_FEATURES][NUM_FEATURES], double projected_data[NUM_SAMPLES][NUM_FEATURES], int num_components) {
    for (int i = 0; i < NUM_SAMPLES; i++) {
        for (int j = 0; j < num_components; j++) {
            projected_data[i][j] = 0;
            for (int k = 0; k < NUM_FEATURES; k++) {
                projected_data[i][j] += centered_data[i][k] * eigenvectors[k][j];
            }
        }
    }
}

// Main function that reads the number of components as a command-line argument
int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <num_components>" << std::endl;
        return 1;
    }

    int num_components = atoi(argv[1]);

    // Sample data: 5 samples and 3 features
    double data[NUM_SAMPLES][NUM_FEATURES] = {
        {2.5, 2.4, 3.5},
        {0.5, 0.7, 1.5},
        {2.2, 2.9, 3.0},
        {1.9, 2.2, 2.7},
        {3.1, 3.0, 3.6}
    };

    double mean[NUM_FEATURES];
    double centered_data[NUM_SAMPLES][NUM_FEATURES];
    double covariance_matrix[NUM_FEATURES][NUM_FEATURES];
    double eigenvectors[NUM_FEATURES][NUM_FEATURES];
    double eigenvalues[NUM_FEATURES];
    double projected_data[NUM_SAMPLES][NUM_FEATURES];

    // Step 1: Calculate mean of each feature
    calculate_mean(data, mean);

    // Step 2: Center the data
    center_data(data, mean, centered_data);

    // Step 3: Compute covariance matrix
    compute_covariance_matrix(centered_data, covariance_matrix);

    // Step 4: Compute eigenvalues and eigenvectors using LAPACK
    compute_eigenvalues_and_eigenvectors(covariance_matrix, eigenvectors, eigenvalues, NUM_FEATURES);

    // Step 5: Sort eigenvalues and eigenvectors
    sort_eigenvectors(eigenvalues, eigenvectors, NUM_FEATURES);

    // Step 6: Project the data onto the new principal components
    project_data(centered_data, eigenvectors, projected_data, num_components);

    // Output eigenvectors and projected data
    std::cout << "Eigenvalues:" << std::endl;
    for (int i = 0; i < NUM_FEATURES; i++) {
        std::cout << eigenvalues[i] << " ";
    }
    std::cout << "\n\nEigenvectors:" << std::endl;
    for (int i = 0; i < NUM_FEATURES; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            std::cout << eigenvectors[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nProjected data onto first " << num_components << " principal components:" << std::endl;
    for (int i = 0; i < NUM_SAMPLES; i++) {
        for (int j = 0; j < num_components; j++) {
            std::cout << projected_data[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
