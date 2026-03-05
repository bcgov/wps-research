/* 20260120: pcp.cpp - parallel copy
   Copies files in parallel using parfor construct
   Usage: pcp [-r] [-t <threads>] <source...> <destination>
   20260305: fix partial file cleanup, timestamp preservation,
             thread count flag, usage message, print race
*/
#include"misc.h"
#include<glob.h>
#include<unistd.h>
#include<sys/stat.h>
#include<dirent.h>
#include<libgen.h>
#include<sys/time.h>

#define DEFAULT_THREADS 48
#define CHUNK_SIZE (4*1024*1024)  // 4MB chunks

// Global variables for file copy
vector<str> g_source_files;
vector<str> g_dest_files;
pthread_mutex_t g_copy_mtx;
size_t g_copied_count = 0;
size_t g_failed_count = 0;
size_t g_total_bytes = 0;
int g_threads = DEFAULT_THREADS;

// Copy a single file with chunked I/O
bool copy_file(const str& src, const str& dst, size_t& bytes_out) {
    struct stat st;
    if(stat(src.c_str(), &st) != 0) return false;

    size_t file_size = st.st_size;

    FILE* src_file = fopen(src.c_str(), "rb");
    if(!src_file) return false;

    FILE* dst_file = fopen(dst.c_str(), "wb");
    if(!dst_file){
        fclose(src_file);
        return false;
    }

    char* buffer = (char*)malloc(CHUNK_SIZE);
    size_t bytes_copied = 0;
    bool success = true;

    while(bytes_copied < file_size){
        size_t to_read = min((size_t)CHUNK_SIZE, file_size - bytes_copied);
        size_t read_bytes = fread(buffer, 1, to_read, src_file);
        if(read_bytes != to_read){ success = false; break; }
        size_t written = fwrite(buffer, 1, read_bytes, dst_file);
        if(written != read_bytes){ success = false; break; }
        bytes_copied += read_bytes;
    }

    free(buffer);
    fclose(src_file);
    fclose(dst_file);

    if(success){
        bytes_out = bytes_copied;
        // Preserve permissions
        chmod(dst.c_str(), st.st_mode);
        // Preserve timestamps
        struct timeval times[2];
        times[0].tv_sec  = st.st_atim.tv_sec;
        times[0].tv_usec = st.st_atim.tv_nsec / 1000;
        times[1].tv_sec  = st.st_mtim.tv_sec;
        times[1].tv_usec = st.st_mtim.tv_nsec / 1000;
        utimes(dst.c_str(), times);
    } else {
        // Remove partial file
        unlink(dst.c_str());
    }

    return success;
}

// Worker function for parallel copy
void copy_worker(size_t i){
    str src = g_source_files[i];
    str dst = g_dest_files[i];

    size_t file_size = 0;
    bool success = copy_file(src, dst, file_size);

    mtx_lock(&g_copy_mtx);
    if(success){
        g_copied_count++;
        g_total_bytes += file_size;
    } else {
        g_failed_count++;
    }
    size_t total_processed = g_copied_count + g_failed_count;
    size_t total_files = g_source_files.size();

    // Print inside lock to avoid interleaved output
    if(success){
        cout << "Copied: " << src << " -> " << dst << endl;
    } else {
        cout << "Failed: " << src << endl;
    }
    if(total_processed % 10 == 0 || total_processed == total_files){
        cout << "Progress: " << total_processed << "/" << total_files << " files ("
             << (100 * total_processed / total_files) << "%)" << endl;
    }
    mtx_unlock(&g_copy_mtx);
}

// Recursively collect files from directory
void collect_files_recursive(const str& src_dir, const str& dst_dir){
    DIR* dir = opendir(src_dir.c_str());
    if(!dir){
        cerr << "Cannot open directory: " << src_dir << endl;
        return;
    }

    struct dirent* entry;
    while((entry = readdir(dir)) != NULL){
        str name(entry->d_name);
        if(name == "." || name == "..") continue;

        str src_path = src_dir + "/" + name;
        str dst_path = dst_dir + "/" + name;

        struct stat st;
        if(stat(src_path.c_str(), &st) == 0){
            if(S_ISDIR(st.st_mode)){
                mkdir(dst_path.c_str(), 0755);
                collect_files_recursive(src_path, dst_path);
            } else if(S_ISREG(st.st_mode)){
                g_source_files.push_back(src_path);
                g_dest_files.push_back(dst_path);
            }
        }
    }

    closedir(dir);
}

void process_source(const str& src_path_orig, const str& destination, bool recursive, bool dest_is_dir){
    str src_path = src_path_orig;
    if(!src_path.empty() && src_path.back() == '/')
        src_path = src_path.substr(0, src_path.size()-1);

    struct stat st;
    if(stat(src_path.c_str(), &st) != 0){
        cerr << "Cannot stat: " << src_path << endl;
        return;
    }

    if(S_ISDIR(st.st_mode)){
        if(!recursive){
            cerr << "Skipping directory (use -r): " << src_path << endl;
            return;
        }

        str dst_dir;
        if(dest_is_dir){
            char* src_copy = strdup(src_path.c_str());
            dst_dir = destination + "/" + str(basename(src_copy));
            free(src_copy);
        } else {
            dst_dir = destination;
        }

        mkdir(dst_dir.c_str(), 0755);
        collect_files_recursive(src_path, dst_dir);

    } else if(S_ISREG(st.st_mode)){
        str dst_path;
        if(dest_is_dir){
            char* src_copy = strdup(src_path.c_str());
            dst_path = destination + "/" + str(basename(src_copy));
            free(src_copy);
        } else {
            dst_path = destination;
        }

        g_source_files.push_back(src_path);
        g_dest_files.push_back(dst_path);
    }
}

int main(int argc, char *argv[]){
    if(argc < 2){
        err("Usage: pcp [-r] [-t <threads>] <source...> <destination>\n"
            "\n"
            "Options:\n"
            "  -r            Recursive copy (like cp -r)\n"
            "  -t <threads>  Number of parallel threads (default: 48)\n"
            "\n"
            "Examples:\n"
            "  pcp file.txt /backup/\n"
            "  pcp *.log /archive/\n"
            "  pcp -r my_folder /backup/\n"
            "  pcp -t 16 -r src/ dst/\n"
            "  pcp file1.txt file2.txt /dest/\n");
    }

    bool recursive = false;
    vector<str> args;

    for(int i = 1; i < argc; i++){
        str a(argv[i]);
        if(a == "-r"){
            recursive = true;
        } else if(a == "-t" && i + 1 < argc){
            g_threads = atoi(argv[++i]);
            if(g_threads < 1) g_threads = 1;
        } else {
            args.push_back(a);
        }
    }

    if(args.size() < 2){
        err("Error: need at least one source and one destination");
    }

    str destination = args.back();
    args.pop_back();

    // Create destination if it doesn't exist and multiple sources
    struct stat dst_stat;
    bool dest_is_dir = false;
    if(stat(destination.c_str(), &dst_stat) == 0){
        dest_is_dir = S_ISDIR(dst_stat.st_mode);
    } else if(args.size() > 1){
        // Multiple sources — create destination dir
        mkdir(destination.c_str(), 0755);
        dest_is_dir = true;
    }

    if(args.size() > 1 && !dest_is_dir){
        err("When copying multiple sources, destination must be a directory");
    }

    for(auto& source_pattern : args){
        bool has_glob = (source_pattern.find('*') != str::npos ||
                         source_pattern.find('?') != str::npos ||
                         source_pattern.find('[') != str::npos);

        if(has_glob){
            glob_t glob_result;
            memset(&glob_result, 0, sizeof(glob_result));
            int ret = glob(source_pattern.c_str(), GLOB_TILDE | GLOB_MARK, NULL, &glob_result);
            if(ret == GLOB_NOMATCH){
                cerr << "No matches found for: " << source_pattern << endl;
            } else if(ret != 0){
                cerr << "Error processing pattern: " << source_pattern << endl;
            } else {
                for(size_t j = 0; j < glob_result.gl_pathc; j++)
                    process_source(str(glob_result.gl_pathv[j]), destination, recursive, dest_is_dir);
            }
            globfree(&glob_result);
        } else {
            process_source(source_pattern, destination, recursive, dest_is_dir);
        }
    }

    if(g_source_files.empty()){
        cout << "No files to copy." << endl;
        return 0;
    }

    cout << "Copying " << g_source_files.size() << " files with " << g_threads << " threads..." << endl;

    pthread_mutex_init(&print_mtx, NULL);
    pthread_mutex_init(&pt_nxt_j_mtx, NULL);
    pthread_mutex_init(&g_copy_mtx, NULL);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    parfor(0, g_source_files.size(), copy_worker, g_threads);

    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

    cout << "\n=== Copy Summary ===" << endl;
    cout << "Successfully copied: " << g_copied_count << " files" << endl;
    cout << "Failed:              " << g_failed_count << " files" << endl;
    cout << "Total data:          " << (g_total_bytes / (1024.0 * 1024.0)) << " MB" << endl;
    cout << "Time:                " << elapsed << " seconds" << endl;
    if(elapsed > 0)
        cout << "Speed:               " << (g_total_bytes / (1024.0 * 1024.0)) / elapsed << " MB/s" << endl;

    return (g_failed_count > 0) ? 1 : 0;
}
