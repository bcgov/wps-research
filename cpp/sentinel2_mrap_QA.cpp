/** 20260128 

  g++ -O3 sentinel2_mrap_QA.cpp $(gdal-config --cflags) -o sentinel2_mrap_QA $(gdal-config --libs)

 * MRAP Quality Control Program
 *
 * Validates MRAP (Most Recent Available Pixel) products by checking:
 * 1. NaN consistency within each file (if one band is NaN, all bands should be NaN)
 * 2. Temporal consistency (non-NaN pixels should never revert to NaN)
 *
 * Usage: ./sentinel2_mrap_QA [optional: specific L2_* folder]
 *
 * Requires: GDAL library
 * Compile: g++ -O3 sentinel2_mrap_QA.cpp -o sentinel2_mrap_QA $(gdal-config --cflags --libs)
 */

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <dirent.h>
#include <sys/stat.h>

extern "C" {
#include <gdal.h>
}

struct MrapFile {
    std::string path;
    std::string timestamp;  // e.g., "20251103T192611"
};

// Extract timestamp from filename (3rd underscore-separated field)
std::string extract_timestamp(const std::string& filename) {
    size_t start = 0;
    int underscore_count = 0;

    for (size_t i = 0; i < filename.size(); ++i) {
        if (filename[i] == '_') {
            underscore_count++;
            if (underscore_count == 2) {
                start = i + 1;
            } else if (underscore_count == 3) {
                return filename.substr(start, i - start);
            }
        }
    }
    return "";
}

// List directories matching pattern "L2_*"
std::vector<std::string> find_l2_folders(const std::string& base_path = ".") {
    std::vector<std::string> folders;
    DIR* dir = opendir(base_path.c_str());
    if (!dir) return folders;

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name.rfind("L2_", 0) == 0) {  // starts with "L2_"
            std::string full_path = base_path + "/" + name;
            struct stat st;
            if (stat(full_path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
                folders.push_back(full_path);
            }
        }
    }
    closedir(dir);
    std::sort(folders.begin(), folders.end());
    return folders;
}

// Find all *_MRAP.bin files in a folder
std::vector<MrapFile> find_mrap_files(const std::string& folder) {
    std::vector<MrapFile> files;
    DIR* dir = opendir(folder.c_str());
    if (!dir) return files;

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name.size() > 9 && name.substr(name.size() - 9) == "_MRAP.bin") {
            MrapFile mf;
            mf.path = folder + "/" + name;
            mf.timestamp = extract_timestamp(name);
            if (!mf.timestamp.empty()) {
                files.push_back(mf);
            }
        }
    }
    closedir(dir);

    // Sort by timestamp
    std::sort(files.begin(), files.end(),
              [](const MrapFile& a, const MrapFile& b) {
                  return a.timestamp < b.timestamp;
              });

    return files;
}

// QA result structure
struct QAResult {
    bool passed = true;
    int nan_consistency_errors = 0;    // pixels where some bands are NaN but not all
    int nan_reversion_errors = 0;      // pixels that reverted from valid to NaN
    std::vector<std::string> error_messages;
};

// Check a single MRAP file for internal NaN consistency
int check_nan_consistency(GDALDatasetH dataset, int& total_nan_pixels, int& total_valid_pixels) {
    int xsize = GDALGetRasterXSize(dataset);
    int ysize = GDALGetRasterYSize(dataset);
    int nbands = GDALGetRasterCount(dataset);
    int npixels = xsize * ysize;
    int errors = 0;

    total_nan_pixels = 0;
    total_valid_pixels = 0;

    // Allocate buffers for all bands
    std::vector<std::vector<float>> bands(nbands);
    for (int b = 0; b < nbands; ++b) {
        bands[b].resize(npixels);
        GDALRasterBandH band = GDALGetRasterBand(dataset, b + 1);
        CPLErr err = GDALRasterIO(band, GF_Read, 0, 0, xsize, ysize,
                                  bands[b].data(), xsize, ysize, GDT_Float32, 0, 0);
        if (err != CE_None) {
            std::cerr << "  Warning: RasterIO failed for band " << (b + 1) << std::endl;
        }
    }

    // Check each pixel
    for (int i = 0; i < npixels; ++i) {
        int nan_count = 0;
        for (int b = 0; b < nbands; ++b) {
            if (std::isnan(bands[b][i])) {
                nan_count++;
            }
        }

        if (nan_count == nbands) {
            total_nan_pixels++;
        } else if (nan_count == 0) {
            total_valid_pixels++;
        } else {
            // Inconsistent: some bands NaN, some not
            errors++;
        }
    }

    return errors;
}

// Create a valid mask from a dataset (true = has valid data in all bands)
std::vector<bool> create_valid_mask(GDALDatasetH dataset) {
    int xsize = GDALGetRasterXSize(dataset);
    int ysize = GDALGetRasterYSize(dataset);
    int nbands = GDALGetRasterCount(dataset);
    int npixels = xsize * ysize;

    std::vector<bool> valid_mask(npixels, true);
    std::vector<float> buffer(npixels);

    for (int b = 1; b <= nbands; ++b) {
        GDALRasterBandH band = GDALGetRasterBand(dataset, b);
        CPLErr err = GDALRasterIO(band, GF_Read, 0, 0, xsize, ysize,
                                  buffer.data(), xsize, ysize, GDT_Float32, 0, 0);
        if (err != CE_None) {
            std::cerr << "  Warning: RasterIO failed for band " << b << std::endl;
        }

        for (int i = 0; i < npixels; ++i) {
            if (std::isnan(buffer[i])) {
                valid_mask[i] = false;
            }
        }
    }

    return valid_mask;
}

// Run QA on a folder
QAResult run_qa_on_folder(const std::string& folder) {
    QAResult result;

    std::vector<MrapFile> mrap_files = find_mrap_files(folder);

    if (mrap_files.empty()) {
        result.error_messages.push_back("No MRAP files found in " + folder);
        return result;
    }

    std::cout << "\n=== Processing: " << folder << " ===" << std::endl;
    std::cout << "Found " << mrap_files.size() << " MRAP files" << std::endl;

    std::vector<bool> prev_valid_mask;
    int prev_valid_count = 0;
    std::string prev_timestamp;

    for (size_t idx = 0; idx < mrap_files.size(); ++idx) {
        const MrapFile& mf = mrap_files[idx];
        std::cout << "\n[" << (idx + 1) << "/" << mrap_files.size() << "] "
                  << mf.timestamp << std::endl;
        std::cout << "  File: " << mf.path << std::endl;

        GDALDatasetH dataset = GDALOpen(mf.path.c_str(), GA_ReadOnly);
        if (!dataset) {
            result.passed = false;
            result.error_messages.push_back("Failed to open: " + mf.path);
            continue;
        }

        int xsize = GDALGetRasterXSize(dataset);
        int ysize = GDALGetRasterYSize(dataset);
        int nbands = GDALGetRasterCount(dataset);
        int npixels = xsize * ysize;

        std::cout << "  Dimensions: " << xsize << " x " << ysize
                  << " x " << nbands << " bands" << std::endl;

        // Check 1: NaN consistency within file
        int nan_pixels, valid_pixels;
        int consistency_errors = check_nan_consistency(dataset, nan_pixels, valid_pixels);

        if (consistency_errors > 0) {
            result.passed = false;
            result.nan_consistency_errors += consistency_errors;
            std::string msg = "  ERROR: " + std::to_string(consistency_errors) +
                              " pixels have inconsistent NaN across bands";
            std::cout << msg << std::endl;
            result.error_messages.push_back(mf.path + ": " + msg);
        } else {
            std::cout << "  NaN consistency: OK" << std::endl;
        }

        std::cout << "  Valid pixels: " << valid_pixels
                  << " (" << (100.0 * valid_pixels / npixels) << "%)" << std::endl;
        std::cout << "  NaN pixels: " << nan_pixels
                  << " (" << (100.0 * nan_pixels / npixels) << "%)" << std::endl;

        // Check 2: Temporal consistency (non-NaN should not revert to NaN)
        std::vector<bool> current_valid_mask = create_valid_mask(dataset);

        if (!prev_valid_mask.empty()) {
            int reversion_count = 0;
            for (int i = 0; i < npixels; ++i) {
                if (prev_valid_mask[i] && !current_valid_mask[i]) {
                    reversion_count++;
                }
            }

            if (reversion_count > 0) {
                result.passed = false;
                result.nan_reversion_errors += reversion_count;
                std::string msg = "  ERROR: " + std::to_string(reversion_count) +
                                  " pixels reverted from valid to NaN since " + prev_timestamp;
                std::cout << msg << std::endl;
                result.error_messages.push_back(mf.path + ": " + msg);
            } else {
                int new_valid = valid_pixels - prev_valid_count;
                std::cout << "  Temporal consistency: OK (+" << new_valid
                          << " newly valid pixels)" << std::endl;
            }
        }

        prev_valid_mask = std::move(current_valid_mask);
        prev_valid_count = valid_pixels;
        prev_timestamp = mf.timestamp;

        GDALClose(dataset);
    }

    return result;
}

int main(int argc, char* argv[]) {
    GDALAllRegister();

    std::cout << "MRAP Quality Control Program" << std::endl;
    std::cout << "============================" << std::endl;

    std::vector<std::string> folders;

    if (argc > 1) {
        // Process specific folder
        folders.push_back(argv[1]);
    } else {
        // Find all L2_* folders
        folders = find_l2_folders(".");
    }

    if (folders.empty()) {
        std::cerr << "No L2_* folders found" << std::endl;
        return 1;
    }

    std::cout << "Found " << folders.size() << " L2_* folder(s)" << std::endl;

    int total_folders_passed = 0;
    int total_folders_failed = 0;
    int total_nan_consistency_errors = 0;
    int total_nan_reversion_errors = 0;

    for (const auto& folder : folders) {
        QAResult result = run_qa_on_folder(folder);

        if (result.passed) {
            total_folders_passed++;
        } else {
            total_folders_failed++;
        }

        total_nan_consistency_errors += result.nan_consistency_errors;
        total_nan_reversion_errors += result.nan_reversion_errors;
    }

    // Summary
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "SUMMARY" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Folders processed: " << folders.size() << std::endl;
    std::cout << "Folders passed:    " << total_folders_passed << std::endl;
    std::cout << "Folders failed:    " << total_folders_failed << std::endl;
    std::cout << std::endl;
    std::cout << "Total NaN consistency errors: " << total_nan_consistency_errors << std::endl;
    std::cout << "Total NaN reversion errors:   " << total_nan_reversion_errors << std::endl;

    if (total_folders_failed == 0) {
        std::cout << "\n*** ALL QA CHECKS PASSED ***" << std::endl;
        return 0;
    } else {
        std::cout << "\n*** QA FAILURES DETECTED ***" << std::endl;
        return 1;
    }
}

