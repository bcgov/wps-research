/* 20260120: pcp.cpp - parallel copy
   Copies files in parallel using parfor construct
   Usage: pcp [-r] <source> <destination>
*/
#include"misc.h"
#include<glob.h>
#include<unistd.h>
#include<sys/stat.h>
#include<dirent.h>
#include<libgen.h>
#include<sys/time.h>

#define THREADS 48
#define CHUNK_SIZE (4*1024*1024)  // 4MB chunks

// Global variables for file copy
vector<str> g_source_files;
vector<str> g_dest_files;
pthread_mutex_t g_copy_mtx;
size_t g_copied_count = 0;
size_t g_failed_count = 0;
size_t g_total_bytes = 0;

// Copy a single file with chunked I/O
bool copy_file(const str& src, const str& dst) {
    // Get source file size
    struct stat st;
    if(stat(src.c_str(), &st) != 0) {
        return false;
    }
    
    size_t file_size = st.st_size;
    
    // Open source and destination
    FILE* src_file = fopen(src.c_str(), "rb");
    if(!src_file) {
        return false;
    }
    
    FILE* dst_file = fopen(dst.c_str(), "wb");
    if(!dst_file) {
        fclose(src_file);
        return false;
    }
    
    // Copy in chunks
    char* buffer = (char*)malloc(CHUNK_SIZE);
    size_t bytes_copied = 0;
    bool success = true;
    
    while(bytes_copied < file_size) {
        size_t to_read = (file_size - bytes_copied < CHUNK_SIZE) ? 
                         (file_size - bytes_copied) : CHUNK_SIZE;
        
        size_t read_bytes = fread(buffer, 1, to_read, src_file);
        if(read_bytes != to_read) {
            success = false;
            break;
        }
        
        size_t written = fwrite(buffer, 1, read_bytes, dst_file);
        if(written != read_bytes) {
            success = false;
            break;
        }
        
        bytes_copied += read_bytes;
    }
    
    free(buffer);
    fclose(src_file);
    fclose(dst_file);
    
    // Preserve permissions
    if(success) {
        chmod(dst.c_str(), st.st_mode);
    }
    
    return success;
}

// Worker function for parallel copy
void copy_worker(size_t i) {
    str src = g_source_files[i];
    str dst = g_dest_files[i];
    
    bool success = copy_file(src, dst);
    
    mtx_lock(&g_copy_mtx);
    if(success) {
        g_copied_count++;
        struct stat st;
        if(stat(src.c_str(), &st) == 0) {
            g_total_bytes += st.st_size;
        }
        cout << "Copied: " << src << " -> " << dst << endl;
    } else {
        g_failed_count++;
        cout << "Failed: " << src << endl;
    }
    
    // Progress update
    size_t total_processed = g_copied_count + g_failed_count;
    size_t total_files = g_source_files.size();
    if(total_processed % 10 == 0 || total_processed == total_files) {
        cout << "Progress: " << total_processed << "/" << total_files << " files (" 
             << (100 * total_processed / total_files) << "%)" << endl;
    }
    mtx_unlock(&g_copy_mtx);
}

// Recursively collect files from directory
void collect_files_recursive(const str& src_dir, const str& dst_dir, 
                             vector<str>& src_files, vector<str>& dst_files) {
    DIR* dir = opendir(src_dir.c_str());
    if(!dir) {
        cerr << "Cannot open directory: " << src_dir << endl;
        return;
    }
    
    struct dirent* entry;
    while((entry = readdir(dir)) != NULL) {
        str name(entry->d_name);
        
        // Skip . and ..
        if(name == "." || name == "..") continue;
        
        str src_path = src_dir + "/" + name;
        str dst_path = dst_dir + "/" + name;
        
        struct stat st;
        if(stat(src_path.c_str(), &st) == 0) {
            if(S_ISDIR(st.st_mode)) {
                // Create destination directory
                mkdir(dst_path.c_str(), 0755);
                // Recurse
                collect_files_recursive(src_path, dst_path, src_files, dst_files);
            } else if(S_ISREG(st.st_mode)) {
                // Regular file
                src_files.push_back(src_path);
                dst_files.push_back(dst_path);
            }
        }
    }
    
    closedir(dir);
}

int main(int argc, char *argv[]) {
    bool recursive = false;
    int arg_offset = 1;
    
    // Check for -r flag
    if(argc > 1 && str(argv[1]) == "-r") {
        recursive = true;
        arg_offset = 2;
    }
    
    if(argc < arg_offset + 2) {
        err("Usage: pcp [-r] <source> <destination>\n"
            "  -r    recursive copy (like cp -r)");
    }
    
    str source_pattern(argv[arg_offset]);
    str destination(argv[arg_offset + 1]);
    
    // Check if destination exists and is a directory
    struct stat dst_stat;
    bool dest_is_dir = false;
    if(stat(destination.c_str(), &dst_stat) == 0) {
        dest_is_dir = S_ISDIR(dst_stat.st_mode);
    }
    
    // Expand source pattern with glob
    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));
    
    int ret = glob(source_pattern.c_str(), GLOB_TILDE | GLOB_MARK, NULL, &glob_result);
    
    if(ret == GLOB_NOMATCH) {
        cerr << "No matches found for: " << source_pattern << endl;
        return 1;
    } else if(ret != 0) {
        cerr << "Error processing pattern: " << source_pattern << endl;
        return 1;
    }
    
    // Process matched files
    for(size_t i = 0; i < glob_result.gl_pathc; i++) {
        str src_path(glob_result.gl_pathv[i]);
        
        // Remove trailing slash if present
        if(src_path.size() > 0 && src_path[src_path.size()-1] == '/') {
            src_path = src_path.substr(0, src_path.size()-1);
        }
        
        struct stat st;
        if(stat(src_path.c_str(), &st) != 0) {
            cerr << "Cannot stat: " << src_path << endl;
            continue;
        }
        
        if(S_ISDIR(st.st_mode)) {
            if(!recursive) {
                cerr << "Skipping directory (use -r): " << src_path << endl;
                continue;
            }
            
            // Determine destination directory name
            str dst_dir;
            if(dest_is_dir) {
                // Extract basename of source directory
                char* src_copy = strdup(src_path.c_str());
                str base = basename(src_copy);
                free(src_copy);
                dst_dir = destination + "/" + base;
            } else {
                dst_dir = destination;
            }
            
            // Create destination directory
            mkdir(dst_dir.c_str(), 0755);
            
            // Collect all files recursively
            collect_files_recursive(src_path, dst_dir, g_source_files, g_dest_files);
            
        } else if(S_ISREG(st.st_mode)) {
            // Regular file
            str dst_path;
            if(dest_is_dir) {
                // Extract basename
                char* src_copy = strdup(src_path.c_str());
                str base = basename(src_copy);
                free(src_copy);
                dst_path = destination + "/" + base;
            } else {
                dst_path = destination;
            }
            
            g_source_files.push_back(src_path);
            g_dest_files.push_back(dst_path);
        }
    }
    
    globfree(&glob_result);
    
    if(g_source_files.size() == 0) {
        cout << "No files to copy." << endl;
        return 0;
    }
    
    cout << "Copying " << g_source_files.size() << " files with " << THREADS << " threads..." << endl;
    
    // Initialize mutexes
    pthread_mutex_init(&print_mtx, NULL);
    pthread_mutex_init(&pt_nxt_j_mtx, NULL);
    pthread_mutex_init(&g_copy_mtx, NULL);
    
    // Perform parallel copy
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    parfor(0, g_source_files.size(), copy_worker, THREADS);
    
    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    
    // Summary
    cout << "\n=== Copy Summary ===" << endl;
    cout << "Successfully copied: " << g_copied_count << " files" << endl;
    cout << "Failed: " << g_failed_count << " files" << endl;
    cout << "Total bytes: " << (g_total_bytes / (1024.0 * 1024.0)) << " MB" << endl;
    cout << "Time: " << elapsed << " seconds" << endl;
    if(elapsed > 0) {
        cout << "Speed: " << (g_total_bytes / (1024.0 * 1024.0)) / elapsed << " MB/s" << endl;
    }
    
    return (g_failed_count > 0) ? 1 : 0;
}


