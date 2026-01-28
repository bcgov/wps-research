/** 20260128 

  g++ -O3 sentinel2_mrap_QA.cpp $(gdal-config --cflags) -o sentinel2_mrap_QA $(gdal-config --libs)

 * MRAP Quality Control Program
 * 
 * Validates MRAP (Most Recent Available Pixel) products by checking:
 * 1. NaN consistency within each file (if one band is NaN, all bands should be NaN)
 * 2. Temporal consistency (non-NaN pixels should never revert to NaN)
 * 3. Duplicate timestamps
 * 
 * When NaN reversion errors are found, writes an ENVI-format BSQ file marking
 * the affected pixels (1.0 = reversion, 0.0 = no reversion).
 * 
 * Usage: ./sentinel2_mrap_QA [optional: specific L2_* folder]
 * 
 * Requires: GDAL library
 * Compile: g++ -O3 sentinel2_mrap_QA.cpp $(gdal-config --cflags) -o sentinel2_mrap_QA $(gdal-config --libs) -lpthread
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <dirent.h>
#include <sys/stat.h>
#include <pthread.h>
#include <unistd.h>

extern "C" {
#include <gdal.h>
}

#define for0(i, n) for(size_t i = 0; i < n; i++)

// Parallelism infrastructure
pthread_attr_t pt_attr;
pthread_mutex_t pt_nxt_j_mtx;
pthread_mutex_t print_mtx;
size_t pt_nxt_j;
size_t pt_start_j;
size_t pt_end_j;
void (*pt_eval)(size_t);

void pt_init_mtx() {
    pthread_mutex_init(&print_mtx, NULL);
    pthread_mutex_init(&pt_nxt_j_mtx, NULL);
}

void mtx_lock(pthread_mutex_t* mtx) {
    pthread_mutex_lock(mtx);
}

void mtx_unlock(pthread_mutex_t* mtx) {
    pthread_mutex_unlock(mtx);
}

void cprint(const std::string& s) {
    mtx_lock(&print_mtx);
    std::cout << s << std::endl;
    mtx_unlock(&print_mtx);
}

void* pt_worker_fun(void* arg) {
    size_t my_nxt_j;
    while (1) {
        mtx_lock(&pt_nxt_j_mtx);
        my_nxt_j = pt_nxt_j++;
        mtx_unlock(&pt_nxt_j_mtx);
        if (my_nxt_j >= pt_end_j) return NULL;
        pt_eval(my_nxt_j);
    }
}

void parfor(size_t start_j, size_t end_j, void(*eval)(size_t), int cores_use = 0) {
    pt_eval = eval;
    pt_end_j = end_j;
    pt_nxt_j = start_j;
    pt_start_j = start_j;
    
    int cores_avail = sysconf(_SC_NPROCESSORS_ONLN);
    size_t n_cores = (cores_avail > cores_use && cores_use > 0) ? cores_use : cores_avail;
    if (cores_use == 0) n_cores = cores_avail;
    
    // Don't use more cores than jobs
    size_t n_jobs = end_j - start_j;
    if (n_cores > n_jobs) n_cores = n_jobs;
    
    std::cout << "Using " << n_cores << " cores for " << n_jobs << " jobs" << std::endl;
    
    pthread_attr_init(&pt_attr);
    pthread_attr_setdetachstate(&pt_attr, PTHREAD_CREATE_JOINABLE);
    pthread_t* my_pthread = new pthread_t[n_cores];
    
    for0(j, n_cores) {
        pthread_create(&my_pthread[j], &pt_attr, pt_worker_fun, (void*)j);
    }
    for0(j, n_cores) {
        pthread_join(my_pthread[j], NULL);
    }
    
    delete[] my_pthread;
}

// Data structures
struct MrapFile {
    std::string path;
    std::string timestamp;
};

struct ComparisonResult {
    size_t index;               // comparison index (0 = compare file 0 to file 1)
    std::string prev_file;
    std::string curr_file;
    std::string prev_timestamp;
    std::string curr_timestamp;
    int prev_valid_count;
    int curr_valid_count;
    int reversion_count;
    bool has_error;
    std::string error_msg;
    std::string reversion_mask_file;  // path to ENVI mask file if created
};

// Global state for parallel comparison
std::vector<MrapFile> g_mrap_files;
std::vector<ComparisonResult> g_results;
pthread_mutex_t g_results_mtx;
int g_xsize, g_ysize, g_nbands, g_npixels;

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

// Extract just the filename from a path
std::string basename(const std::string& path) {
    size_t pos = path.rfind('/');
    if (pos == std::string::npos) return path;
    return path.substr(pos + 1);
}

// Get map info from source file header (read existing .hdr file)
std::string get_map_info_from_header(const std::string& bin_path) {
    std::string hdr_path = bin_path + ".hdr";
    // Also try without .bin extension
    if (bin_path.size() > 4 && bin_path.substr(bin_path.size() - 4) == ".bin") {
        std::string alt_hdr = bin_path.substr(0, bin_path.size() - 4) + ".hdr";
        std::ifstream test(alt_hdr);
        if (test.good()) {
            hdr_path = alt_hdr;
        }
    }
    
    std::ifstream hdr(hdr_path);
    if (!hdr.is_open()) return "";
    
    std::string line;
    std::string map_info;
    bool in_map_info = false;
    
    while (std::getline(hdr, line)) {
        // Check for "map info" line
        if (line.find("map info") != std::string::npos || 
            line.find("map_info") != std::string::npos) {
            size_t eq_pos = line.find('=');
            if (eq_pos != std::string::npos) {
                map_info = line.substr(eq_pos + 1);
                // Trim leading whitespace
                size_t start = map_info.find_first_not_of(" \t");
                if (start != std::string::npos) {
                    map_info = map_info.substr(start);
                }
                in_map_info = true;
                // Check if map info continues on next lines (multi-line value)
                if (map_info.back() != '}') {
                    continue;
                }
            }
            break;
        }
        if (in_map_info) {
            map_info += line;
            if (line.find('}') != std::string::npos) break;
        }
    }
    
    return map_info;
}

// Write ENVI format BSQ file with NaN reversion mask
void write_envi_reversion_mask(const std::string& output_path,
                                const std::vector<float>& mask_data,
                                int xsize, int ysize,
                                const std::string& map_info,
                                const std::string& prev_file,
                                const std::string& curr_file) {
    // Write binary data (BSQ format, single band, float32)
    std::ofstream bin_out(output_path, std::ios::binary);
    if (!bin_out.is_open()) {
        cprint("ERROR: Could not create " + output_path);
        return;
    }
    bin_out.write(reinterpret_cast<const char*>(mask_data.data()), 
                  mask_data.size() * sizeof(float));
    bin_out.close();
    
    // Write ENVI header
    std::string hdr_path = output_path + ".hdr";
    std::ofstream hdr_out(hdr_path);
    if (!hdr_out.is_open()) {
        cprint("ERROR: Could not create " + hdr_path);
        return;
    }
    
    // Create band name indicating the comparison
    std::string band_name = "NaN_reversion: " + basename(prev_file) + " -> " + basename(curr_file);
    
    hdr_out << "ENVI" << std::endl;
    hdr_out << "description = {NaN reversion mask: 1.0 = pixel reverted to NaN, 0.0 = no reversion}" << std::endl;
    hdr_out << "samples = " << xsize << std::endl;
    hdr_out << "lines = " << ysize << std::endl;
    hdr_out << "bands = 1" << std::endl;
    hdr_out << "header offset = 0" << std::endl;
    hdr_out << "file type = ENVI Standard" << std::endl;
    hdr_out << "data type = 4" << std::endl;  // 4 = 32-bit float
    hdr_out << "interleave = bsq" << std::endl;
    hdr_out << "byte order = 0" << std::endl;  // little endian
    if (!map_info.empty()) {
        hdr_out << "map info = " << map_info << std::endl;
    }
    hdr_out << "band names = {" << band_name << "}" << std::endl;
    
    hdr_out.close();
    
    cprint("  Wrote reversion mask: " + output_path);
}

// List directories matching pattern "L2_*"
std::vector<std::string> find_l2_folders(const std::string& base_path = ".") {
    std::vector<std::string> folders;
    DIR* dir = opendir(base_path.c_str());
    if (!dir) return folders;
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name.rfind("L2_", 0) == 0) {
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
    
    std::sort(files.begin(), files.end(),
              [](const MrapFile& a, const MrapFile& b) {
                  return a.timestamp < b.timestamp;
              });
    
    return files;
}

// Check for duplicate timestamps, return list of duplicates
std::vector<std::string> find_duplicate_timestamps(const std::vector<MrapFile>& files) {
    std::map<std::string, int> ts_count;
    for (const auto& f : files) {
        ts_count[f.timestamp]++;
    }
    
    std::vector<std::string> duplicates;
    for (const auto& p : ts_count) {
        if (p.second > 1) {
            duplicates.push_back(p.first + " (appears " + std::to_string(p.second) + " times)");
        }
    }
    return duplicates;
}

// Create valid mask from file (true = valid data in all bands)
std::vector<bool> create_valid_mask_from_file(const std::string& path, int& valid_count) {
    GDALDatasetH dataset = GDALOpen(path.c_str(), GA_ReadOnly);
    if (!dataset) {
        return std::vector<bool>();
    }
    
    int xsize = GDALGetRasterXSize(dataset);
    int ysize = GDALGetRasterYSize(dataset);
    int nbands = GDALGetRasterCount(dataset);
    int npixels = xsize * ysize;
    
    std::vector<bool> valid_mask(npixels, true);
    std::vector<float> buffer(npixels);
    valid_count = 0;
    
    for (int b = 1; b <= nbands; ++b) {
        GDALRasterBandH band = GDALGetRasterBand(dataset, b);
        GDALRasterIO(band, GF_Read, 0, 0, xsize, ysize,
                     buffer.data(), xsize, ysize, GDT_Float32, 0, 0);
        
        for (int i = 0; i < npixels; ++i) {
            if (std::isnan(buffer[i])) {
                valid_mask[i] = false;
            }
        }
    }
    
    for (int i = 0; i < npixels; ++i) {
        if (valid_mask[i]) valid_count++;
    }
    
    GDALClose(dataset);
    return valid_mask;
}

// Parallel comparison function
void compare_pair(size_t idx) {
    // Compare file[idx] with file[idx+1]
    const MrapFile& prev_file = g_mrap_files[idx];
    const MrapFile& curr_file = g_mrap_files[idx + 1];
    
    ComparisonResult res;
    res.index = idx;
    res.prev_file = prev_file.path;
    res.curr_file = curr_file.path;
    res.prev_timestamp = prev_file.timestamp;
    res.curr_timestamp = curr_file.timestamp;
    res.has_error = false;
    res.reversion_count = 0;
    
    int prev_valid, curr_valid;
    std::vector<bool> prev_mask = create_valid_mask_from_file(prev_file.path, prev_valid);
    std::vector<bool> curr_mask = create_valid_mask_from_file(curr_file.path, curr_valid);
    
    res.prev_valid_count = prev_valid;
    res.curr_valid_count = curr_valid;
    
    if (prev_mask.empty() || curr_mask.empty()) {
        res.has_error = true;
        res.error_msg = "Failed to open file(s)";
    } else {
        // Create reversion mask and count reversions
        std::vector<float> reversion_mask(prev_mask.size(), 0.0f);
        
        for (size_t i = 0; i < prev_mask.size(); ++i) {
            if (prev_mask[i] && !curr_mask[i]) {
                res.reversion_count++;
                reversion_mask[i] = 1.0f;
            }
        }
        
        if (res.reversion_count > 0) {
            res.has_error = true;
            res.error_msg = std::to_string(res.reversion_count) + " pixels reverted to NaN";
            
            // Get dimensions from GDAL
            GDALDatasetH dataset = GDALOpen(curr_file.path.c_str(), GA_ReadOnly);
            if (dataset) {
                int xsize = GDALGetRasterXSize(dataset);
                int ysize = GDALGetRasterYSize(dataset);
                GDALClose(dataset);
                
                // Get map info from source file
                std::string map_info = get_map_info_from_header(curr_file.path);
                
                // Generate output filename: use curr_file path, replace _MRAP.bin with _NAN_reversion.bin
                std::string out_path = curr_file.path;
                size_t pos = out_path.rfind("_MRAP.bin");
                if (pos != std::string::npos) {
                    out_path = out_path.substr(0, pos) + "_NAN_reversion.bin";
                } else {
                    out_path = curr_file.path + "_NAN_reversion.bin";
                }
                
                // Write the ENVI format mask file
                write_envi_reversion_mask(out_path, reversion_mask, xsize, ysize,
                                          map_info, prev_file.path, curr_file.path);
                
                res.reversion_mask_file = out_path;
            }
        }
    }
    
    // Print progress
    std::stringstream ss;
    ss << "[" << (idx + 1) << "/" << (g_mrap_files.size() - 1) << "] "
       << prev_file.timestamp << " -> " << curr_file.timestamp;
    if (res.has_error) {
        ss << " ERROR: " << res.error_msg;
    } else {
        int new_valid = curr_valid - prev_valid;
        ss << " OK (+" << new_valid << " valid pixels)";
    }
    cprint(ss.str());
    
    // Store result
    mtx_lock(&g_results_mtx);
    g_results.push_back(res);
    mtx_unlock(&g_results_mtx);
}

// QA result structure
struct QAResult {
    bool passed = true;
    int nan_reversion_errors = 0;
    std::vector<std::string> nan_reversion_files;
    std::vector<std::string> nan_reversion_mask_files;
    std::vector<std::string> duplicate_timestamps;
    std::vector<ComparisonResult> comparisons;
    std::string folder;
};

// Run QA on a folder with parallel comparisons
QAResult run_qa_on_folder(const std::string& folder) {
    QAResult result;
    result.folder = folder;
    
    g_mrap_files = find_mrap_files(folder);
    
    if (g_mrap_files.empty()) {
        std::cout << "\n=== Processing: " << folder << " ===" << std::endl;
        std::cout << "No MRAP files found" << std::endl;
        return result;
    }
    
    std::cout << "\n=== Processing: " << folder << " ===" << std::endl;
    std::cout << "Found " << g_mrap_files.size() << " MRAP files" << std::endl;
    
    // Check for duplicate timestamps
    result.duplicate_timestamps = find_duplicate_timestamps(g_mrap_files);
    if (!result.duplicate_timestamps.empty()) {
        result.passed = false;
        std::cout << "WARNING: Duplicate timestamps detected!" << std::endl;
        for (const auto& d : result.duplicate_timestamps) {
            std::cout << "  " << d << std::endl;
        }
    }
    
    // Get dimensions from first file
    if (!g_mrap_files.empty()) {
        GDALDatasetH dataset = GDALOpen(g_mrap_files[0].path.c_str(), GA_ReadOnly);
        if (dataset) {
            g_xsize = GDALGetRasterXSize(dataset);
            g_ysize = GDALGetRasterYSize(dataset);
            g_nbands = GDALGetRasterCount(dataset);
            g_npixels = g_xsize * g_ysize;
            std::cout << "Dimensions: " << g_xsize << " x " << g_ysize
                      << " x " << g_nbands << " bands" << std::endl;
            GDALClose(dataset);
        }
    }
    
    // Run parallel comparisons (N-1 comparisons for N files)
    size_t n_comparisons = g_mrap_files.size() - 1;
    if (n_comparisons > 0) {
        std::cout << "\nRunning " << n_comparisons << " pairwise comparisons..." << std::endl;
        
        g_results.clear();
        pthread_mutex_init(&g_results_mtx, NULL);
        
        parfor(0, n_comparisons, compare_pair, 32);
        
        // Sort results by index
        std::sort(g_results.begin(), g_results.end(),
                  [](const ComparisonResult& a, const ComparisonResult& b) {
                      return a.index < b.index;
                  });
        
        result.comparisons = g_results;
        
        // Collect errors
        for (const auto& comp : g_results) {
            if (comp.reversion_count > 0) {
                result.passed = false;
                result.nan_reversion_errors += comp.reversion_count;
                result.nan_reversion_files.push_back(comp.curr_file);
                if (!comp.reversion_mask_file.empty()) {
                    result.nan_reversion_mask_files.push_back(comp.reversion_mask_file);
                }
            }
        }
    }
    
    // Write results to mrap_qa.txt in the folder
    std::string qa_file = folder + "/mrap_qa.txt";
    std::ofstream ofs(qa_file);
    if (ofs.is_open()) {
        ofs << "MRAP Quality Control Report" << std::endl;
        ofs << "Folder: " << folder << std::endl;
        ofs << "Files analyzed: " << g_mrap_files.size() << std::endl;
        ofs << "Comparisons: " << n_comparisons << std::endl;
        ofs << std::string(60, '=') << std::endl;
        
        if (!result.duplicate_timestamps.empty()) {
            ofs << "\nDUPLICATE TIMESTAMPS:" << std::endl;
            for (const auto& d : result.duplicate_timestamps) {
                ofs << "  " << d << std::endl;
            }
        }
        
        ofs << "\nPAIRWISE COMPARISONS (in order):" << std::endl;
        for (const auto& comp : result.comparisons) {
            ofs << "\n[" << (comp.index + 1) << "] " << comp.prev_timestamp
                << " -> " << comp.curr_timestamp << std::endl;
            ofs << "  Prev: " << comp.prev_file << std::endl;
            ofs << "  Curr: " << comp.curr_file << std::endl;
            ofs << "  Valid pixels: " << comp.prev_valid_count << " -> "
                << comp.curr_valid_count;
            int delta = comp.curr_valid_count - comp.prev_valid_count;
            ofs << " (delta: " << (delta >= 0 ? "+" : "") << delta << ")" << std::endl;
            if (comp.has_error) {
                ofs << "  ERROR: " << comp.error_msg << std::endl;
                if (!comp.reversion_mask_file.empty()) {
                    ofs << "  Reversion mask: " << comp.reversion_mask_file << std::endl;
                }
            } else {
                ofs << "  Status: OK" << std::endl;
            }
        }
        
        ofs << "\n" << std::string(60, '=') << std::endl;
        ofs << "SUMMARY" << std::endl;
        ofs << std::string(60, '=') << std::endl;
        ofs << "Total NaN reversion errors: " << result.nan_reversion_errors << std::endl;
        
        if (!result.nan_reversion_files.empty()) {
            ofs << "\nFiles with NaN reversion errors:" << std::endl;
            for (const auto& f : result.nan_reversion_files) {
                ofs << "  " << f << std::endl;
            }
        }
        
        if (!result.nan_reversion_mask_files.empty()) {
            ofs << "\nReversion mask files created:" << std::endl;
            for (const auto& f : result.nan_reversion_mask_files) {
                ofs << "  " << f << std::endl;
            }
        }
        
        ofs << "\nResult: " << (result.passed ? "PASSED" : "FAILED") << std::endl;
        ofs.close();
        
        std::cout << "Results written to: " << qa_file << std::endl;
    }
    
    return result;
}

int main(int argc, char* argv[]) {
    GDALAllRegister();
    pt_init_mtx();
    
    std::cout << "MRAP Quality Control Program" << std::endl;
    std::cout << "============================" << std::endl;
    
    std::vector<std::string> folders;
    
    if (argc > 1) {
        folders.push_back(argv[1]);
    } else {
        folders = find_l2_folders(".");
    }
    
    if (folders.empty()) {
        std::cerr << "No L2_* folders found" << std::endl;
        return 1;
    }
    
    std::cout << "Found " << folders.size() << " L2_* folder(s)" << std::endl;
    
    int total_folders_passed = 0;
    int total_folders_failed = 0;
    int total_nan_reversion_errors = 0;
    std::vector<std::string> all_nan_reversion_files;
    std::vector<std::string> all_nan_reversion_mask_files;
    std::vector<std::string> all_duplicate_timestamps;
    
    for (const auto& folder : folders) {
        QAResult result = run_qa_on_folder(folder);
        
        if (result.passed) {
            total_folders_passed++;
        } else {
            total_folders_failed++;
        }
        
        total_nan_reversion_errors += result.nan_reversion_errors;
        all_nan_reversion_files.insert(all_nan_reversion_files.end(),
                                       result.nan_reversion_files.begin(),
                                       result.nan_reversion_files.end());
        all_nan_reversion_mask_files.insert(all_nan_reversion_mask_files.end(),
                                            result.nan_reversion_mask_files.begin(),
                                            result.nan_reversion_mask_files.end());
        for (const auto& d : result.duplicate_timestamps) {
            all_duplicate_timestamps.push_back(folder + ": " + d);
        }
    }
    
    // Final Summary
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "FINAL SUMMARY" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Folders processed: " << folders.size() << std::endl;
    std::cout << "Folders passed:    " << total_folders_passed << std::endl;
    std::cout << "Folders failed:    " << total_folders_failed << std::endl;
    std::cout << std::endl;
    std::cout << "Total NaN reversion errors: " << total_nan_reversion_errors << std::endl;
    
    if (!all_duplicate_timestamps.empty()) {
        std::cout << "\nDuplicate timestamps detected:" << std::endl;
        for (const auto& d : all_duplicate_timestamps) {
            std::cout << "  " << d << std::endl;
        }
    }
    
    if (!all_nan_reversion_files.empty()) {
        std::cout << "\nFiles with NaN reversion errors:" << std::endl;
        for (const auto& f : all_nan_reversion_files) {
            std::cout << "  " << f << std::endl;
        }
    }
    
    if (!all_nan_reversion_mask_files.empty()) {
        std::cout << "\nReversion mask files created:" << std::endl;
        for (const auto& f : all_nan_reversion_mask_files) {
            std::cout << "  " << f << std::endl;
        }
    }
    
    if (total_folders_failed == 0) {
        std::cout << "\n*** ALL QA CHECKS PASSED ***" << std::endl;
        return 0;
    } else {
        std::cout << "\n*** QA FAILURES DETECTED ***" << std::endl;
        return 1;
    }
}

