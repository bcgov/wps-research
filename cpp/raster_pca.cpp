/* 20250307 pca.c Principal Components Analysis for ENVI-format raster 

  sudo apt-get install liblapack-dev libblas-dev

  g++ raster_pca.cpp misc.cpp -o raster_pca -O3 -llapack -lblas -lm

  Used chatgpt to form original example. Used claude AI to modify to use a headerless (link only) method for accessing LAPACK
*/

#include<iostream>
#include<cstdlib>
#include<cmath>
#include<getopt.h>
#include"misc.h"

// External declarations for BLAS/LAPACK functions
extern "C" {
    // LAPACK functions
    extern int dgeev_(char*, char*, int*, double*, int*, double*, double*, double*, int*, double*, int*, double*, int*, int*);
    // CBLAS functions if needed
    extern double cblas_ddot(const int N, const double *X, const int incX, const double *Y, const int incY);
}

// Function to calculate the mean of each feature
void calculate_mean(const std::vector<std::vector<double>>& data, std::vector<double>& mean) {
    int num_samples = data.size();
    int num_features = data[0].size();
    
    mean.resize(num_features, 0.0);
    
    for (int j = 0; j < num_features; j++) {
        mean[j] = 0.0;
        for (int i = 0; i < num_samples; i++) {
            mean[j] += data[i][j];
        }
        mean[j] /= num_samples;
    }
}

// Function to center the data (subtract the mean from each feature)
void center_data(const std::vector<std::vector<double>>& data, 
                 const std::vector<double>& mean, 
                 std::vector<std::vector<double>>& centered_data) {
    int num_samples = data.size();
    int num_features = data[0].size();
    
    centered_data.resize(num_samples, std::vector<double>(num_features, 0.0));
    
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < num_features; j++) {
            centered_data[i][j] = data[i][j] - mean[j];
        }
    }
}

// Function to compute the covariance matrix
void compute_covariance_matrix(const std::vector<std::vector<double>>& centered_data, 
                              std::vector<std::vector<double>>& covariance_matrix) {
    int num_samples = centered_data.size();
    int num_features = centered_data[0].size();
    
    covariance_matrix.resize(num_features, std::vector<double>(num_features, 0.0));
    
    for (int i = 0; i < num_features; i++) {
        for (int j = 0; j < num_features; j++) {
            covariance_matrix[i][j] = 0.0;
            for (int k = 0; k < num_samples; k++) {
                covariance_matrix[i][j] += centered_data[k][i] * centered_data[k][j];
            }
            covariance_matrix[i][j] /= (num_samples - 1);  // Normalizing by N-1
        }
    }
}

// Function to compute eigenvalues and eigenvectors using LAPACK directly
void compute_eigenvalues_and_eigenvectors(std::vector<std::vector<double>>& matrix, 
                                         std::vector<std::vector<double>>& eigenvectors, 
                                         std::vector<double>& eigenvalues) {
    int num_features = matrix.size();
    
    // Flatten the 2D matrix into a 1D array for LAPACK (column-major order)
    std::vector<double> matrix_flat(num_features * num_features);
    for (int i = 0; i < num_features; i++) {
        for (int j = 0; j < num_features; j++) {
            matrix_flat[j * num_features + i] = matrix[i][j];
        }
    }
    
    char jobvl = 'V';  // Compute left eigenvectors
    char jobvr = 'N';  // Don't compute right eigenvectors
    int N = num_features;
    int lda = N;
    std::vector<double> wr(N, 0.0);  // Real parts of eigenvalues
    std::vector<double> wi(N, 0.0);  // Imaginary parts of eigenvalues
    
    // Prepare the output eigenvector matrix (flattened for LAPACK)
    std::vector<double> vl(N * N, 0.0);  // Left eigenvectors
    int ldvl = N;
    double* vr = nullptr;  // We don't need right eigenvectors
    int ldvr = 1;
    
    // Prepare workspace
    int lwork = 4 * N;
    std::vector<double> work(lwork, 0.0);
    int info;
    
    // Call LAPACK dgeev function directly
    dgeev_(&jobvl, &jobvr, &N, matrix_flat.data(), &lda, 
           wr.data(), wi.data(), vl.data(), &ldvl, vr, &ldvr, 
           work.data(), &lwork, &info);
    
    if (info != 0) {
        std::cerr << "Error in eigenvalue decomposition: " << info << std::endl;
        exit(1);
    }
    
    // Copy eigenvalues to the result array
    eigenvalues.resize(N);
    for (int i = 0; i < N; i++) {
        eigenvalues[i] = wr[i];
        // Warn if there are imaginary parts
        if (std::abs(wi[i]) > 1e-10) {
            std::cerr << "Warning: Complex eigenvalue detected: " << wr[i] << " + " << wi[i] << "i" << std::endl;
        }
    }
    
    // Copy and convert eigenvectors from column-major to row-major format
    eigenvectors.resize(N, std::vector<double>(N, 0.0));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            eigenvectors[i][j] = vl[j * N + i];
        }
    }
}

// Function to sort eigenvalues and eigenvectors in descending order
void sort_eigenvectors(std::vector<double>& eigenvalues, 
                      std::vector<std::vector<double>>& eigenvectors) {
    int num_features = eigenvalues.size();
    
    // Create index vector for sorting
    std::vector<int> indices(num_features);
    for (int i = 0; i < num_features; i++) {
        indices[i] = i;
    }
    
    // Sort indices based on eigenvalues (descending order)
    for (int i = 0; i < num_features - 1; i++) {
        for (int j = i + 1; j < num_features; j++) {
            if (eigenvalues[indices[i]] < eigenvalues[indices[j]]) {
                std::swap(indices[i], indices[j]);
            }
        }
    }
    
    // Create temporary copies of the sorted data
    std::vector<double> sorted_eigenvalues(num_features);
    std::vector<std::vector<double>> sorted_eigenvectors(num_features, std::vector<double>(num_features));
    
    for (int i = 0; i < num_features; i++) {
        sorted_eigenvalues[i] = eigenvalues[indices[i]];
        for (int j = 0; j < num_features; j++) {
            sorted_eigenvectors[j][i] = eigenvectors[j][indices[i]];
        }
    }
    
    // Copy back the sorted data
    eigenvalues = sorted_eigenvalues;
    eigenvectors = sorted_eigenvectors;
}

// Function to project the data onto the principal components
void project_data(const std::vector<std::vector<double>>& centered_data, 
                 const std::vector<std::vector<double>>& eigenvectors, 
                 std::vector<std::vector<double>>& projected_data, 
                 int num_components) {
    int num_samples = centered_data.size();
    int num_features = centered_data[0].size();
    
    // Ensure num_components is not greater than num_features
    num_components = std::min(num_components, num_features);
    
    projected_data.resize(num_samples, std::vector<double>(num_components, 0.0));
    
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < num_components; j++) {
            projected_data[i][j] = 0.0;
            for (int k = 0; k < num_features; k++) {
                projected_data[i][j] += centered_data[i][k] * eigenvectors[k][j];
            }
        }
    }
}

// Main function that reads the number of components as a command-line argument
int main(int argc, char *argv[]) {
  if(argc < 3) err("raster_pca [input ENVI-format data cube] [ number of components ]\n");

  str fn(argv[1]); // input image file name
  int num_components = atoi(argv[2]);
  if(!(exists(fn)))  err("failed to open input file");

  str hfn(hdr_fn(fn)); // input header file name
  str ofn(fn + str("_pca.bin")); // output file name
  str ohn(hdr_fn(ofn, true)); // out header file name

  size_t nrow, ncol, nband, np;
  hread(hfn, nrow, ncol, nband); // read header 1
  np = nrow * ncol;
  float * dat = bread(fn, nrow, ncol, nband);

  int num_samples = np;
  int num_features = nband;
  std::vector<std::vector<double>> data(num_samples, std::vector<double>(num_features));

  for0(i , np){
    for0(k, nband){
      data[i][k] = dat[k * np + i];
    }
  }
  
/*
  // Sample data - now using vectors
    std::vector<std::vector<double>> data = {
        {2.5, 2.4, 3.5},
        {0.5, 0.7, 1.5},
        {2.2, 2.9, 3.0},
        {1.9, 2.2, 2.7},
        {3.1, 3.0, 3.6}
    };
*/
    // Check if num_components is valid
    if (num_components <= 0 || num_components > num_features) {
        std::cerr << "Error: Number of components must be between 1 and " << num_features << std::endl;
        return 1;
    }

    // Data structures
    std::vector<double> mean;
    std::vector<std::vector<double>> centered_data;
    std::vector<std::vector<double>> covariance_matrix;
    std::vector<std::vector<double>> eigenvectors;
    std::vector<double> eigenvalues;
    std::vector<std::vector<double>> projected_data;

    // Step 1: Calculate mean of each feature
    calculate_mean(data, mean);

    // Step 2: Center the data
    center_data(data, mean, centered_data);

    // Step 3: Compute covariance matrix
    compute_covariance_matrix(centered_data, covariance_matrix);

    // Step 4: Compute eigenvalues and eigenvectors using LAPACK
    compute_eigenvalues_and_eigenvectors(covariance_matrix, eigenvectors, eigenvalues);

    // Step 5: Sort eigenvalues and eigenvectors
    sort_eigenvectors(eigenvalues, eigenvectors);

    // Step 6: Project the data onto the new principal components
    project_data(centered_data, eigenvectors, projected_data, num_components);

    // Output results
    std::cout << "Eigenvalues:" << std::endl;
    for (int i = 0; i < num_features; i++) {
        std::cout << eigenvalues[i] << " ";
    }
    
    std::cout << "\n\nEigenvectors:" << std::endl;
    for (int i = 0; i < num_features; i++) {
        for (int j = 0; j < num_features; j++) {
            std::cout << eigenvectors[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nProjected data onto first " << num_components << " principal components:" << std::endl;
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < num_components; j++) {
            std::cout << projected_data[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
