/* 20260305: pmv.cpp - parallel move
   Moves files in parallel using parfor construct
   Usage: pmv [-t <threads>] <source...> <destination>
   Handles large numbers of files where shell glob expansion overflows mv

E.g. prm, pcp, pmv are used in situations where the number of files is too large for linux to even count how many files there are using the standard tools:

/data/pgfc/sentinel2/cloudfree$ mkdir MRAP
/data/pgfc/sentinel2/cloudfree$ mv *MRAP* MRAP
bash: /usr/bin/mv: Argument list too long
/data/pgfc/sentinel2/cloudfree$ ls -1 *MRAP*.bin | wc 
bash: /usr/bin/ls: Argument list too long
      0       0       0
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

// Global variables
vector<str> g_source_files;
vector<str> g_dest_files;
pthread_mutex_t g_move_mtx;
size_t g_moved_count = 0;
size_t g_failed_count = 0;
size_t g_total_files = 0;
size_t g_total_bytes = 0;
int g_threads = DEFAULT_THREADS;

// Worker function for parallel move
void move_worker(size_t i){
    str src = g_source_files[i];
    str dst = g_dest_files[i];

    // Get file size before move for stats
    struct stat st;
    size_t file_size = 0;
    if(stat(src.c_str(), &st) == 0)
        file_size = st.st_size;

    // Try rename first (fast, same filesystem)
    int result = rename(src.c_str(), dst.c_str());

    // If rename fails cross-device, fall back to copy+delete
    if(result != 0 && errno == EXDEV){
        if(stat(src.c_str(), &st) != 0){
            result = -1;
        } else {
            FILE* src_file = fopen(src.c_str(), "rb");
            FILE* dst_file = src_file ? fopen(dst.c_str(), "wb") : NULL;
            if(!src_file || !dst_file){
                if(src_file) fclose(src_file);
                if(dst_file) fclose(dst_file);
                result = -1;
            } else {
                char* buf = (char*)malloc(CHUNK_SIZE);
                size_t file_sz = st.st_size;
                size_t copied = 0;
                bool ok = true;
                while(copied < file_sz){
                    size_t to_read = min(CHUNK_SIZE, file_sz - copied);
                    size_t r = fread(buf, 1, to_read, src_file);
                    if(r != to_read){ ok = false; break; }
                    size_t w = fwrite(buf, 1, r, dst_file);
                    if(w != r){ ok = false; break; }
                    copied += r;
                }
                free(buf);
                fclose(src_file);
                fclose(dst_file);

                if(ok){
                    chmod(dst.c_str(), st.st_mode);
                    struct timeval times[2];
                    times[0].tv_sec  = st.st_atim.tv_sec;
                    times[0].tv_usec = st.st_atim.tv_nsec / 1000;
                    times[1].tv_sec  = st.st_mtim.tv_sec;
                    times[1].tv_usec = st.st_mtim.tv_nsec / 1000;
                    utimes(dst.c_str(), times);
                    unlink(src.c_str());
                    result = 0;
                } else {
                    unlink(dst.c_str());  // Remove partial file
                    result = -1;
                }
            }
        }
    }

    mtx_lock(&g_move_mtx);
    if(result == 0){
        g_moved_count++;
        g_total_bytes += file_size;
    } else {
        g_failed_count++;
    }
    size_t total_processed = g_moved_count + g_failed_count;

    if(result != 0)
        cout << "Failed: " << src << " (error: " << strerror(errno) << ")" << endl;

    if(total_processed % 100 == 0 || total_processed == g_total_files){
        cout << "Progress: " << total_processed << "/" << g_total_files << " files ("
             << (100 * total_processed / g_total_files) << "%) - "
             << "Moved: " << g_moved_count << ", Failed: " << g_failed_count << endl;
    }
    mtx_unlock(&g_move_mtx);
}

void process_source(const str& src_path_orig, const str& destination, bool dest_is_dir){
    str src_path = src_path_orig;
    if(!src_path.empty() && src_path.back() == '/')
        src_path = src_path.substr(0, src_path.size()-1);

    struct stat st;
    if(stat(src_path.c_str(), &st) != 0){
        cerr << "Cannot stat: " << src_path << endl;
        return;
    }

    if(S_ISDIR(st.st_mode)){
        cerr << "Skipping directory: " << src_path << " (pmv does not support moving directories)" << endl;
        return;
    }

    if(S_ISREG(st.st_mode)){
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
        err("Usage: pmv [-t <threads>] <source...> <destination>\n"
            "\n"
            "Options:\n"
            "  -t <threads>  Number of parallel threads (default: 48)\n"
            "\n"
            "Examples:\n"
            "  pmv *.log /archive/\n"
            "  pmv *MRAP* MRAP/\n"
            "  pmv -t 16 *.tif /data/tifs/\n"
            "  pmv file1.txt file2.txt /dest/\n");
    }

    vector<str> args;
    for(int i = 1; i < argc; i++){
        str a(argv[i]);
        if(a == "-t" && i + 1 < argc){
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

    // Check if destination exists and is a directory
    struct stat dst_stat;
    bool dest_is_dir = false;
    if(stat(destination.c_str(), &dst_stat) == 0){
        dest_is_dir = S_ISDIR(dst_stat.st_mode);
    } else if(args.size() > 1){
        mkdir(destination.c_str(), 0755);
        dest_is_dir = true;
    }

    if(args.size() > 1 && !dest_is_dir){
        err("When moving multiple sources, destination must be a directory");
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
                if(glob_result.gl_pathc > 1 && !dest_is_dir){
                    mkdir(destination.c_str(), 0755);
                    dest_is_dir = true;
                }
                for(size_t j = 0; j < glob_result.gl_pathc; j++)
                    process_source(str(glob_result.gl_pathv[j]), destination, dest_is_dir);
            }
            globfree(&glob_result);
        } else {
            process_source(source_pattern, destination, dest_is_dir);
        }
    }

    if(g_source_files.empty()){
        cout << "No files to move." << endl;
        return 0;
    }

    g_total_files = g_source_files.size();
    cout << "Found " << g_total_files << " files to move." << endl;

    // Confirmation prompt for large operations (from prm)
    if(g_total_files > 10){
        cout << "Move " << g_total_files << " files to " << destination << "? (y/n): ";
        str response;
        getline(cin, response);
        trim(response);
        lower(response);
        if(response != "y" && response != "yes"){
            cout << "Move cancelled." << endl;
            return 0;
        }
    }

    pthread_mutex_init(&print_mtx, NULL);
    pthread_mutex_init(&pt_nxt_j_mtx, NULL);
    pthread_mutex_init(&g_move_mtx, NULL);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    cout << "\nMoving files in parallel..." << endl;
    parfor(0, g_total_files, move_worker, g_threads);

    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

    cout << "\n=== Move Summary ===" << endl;
    cout << "Successfully moved: " << g_moved_count << " files" << endl;
    cout << "Failed:             " << g_failed_count << " files" << endl;
    cout << "Total data:         " << (g_total_bytes / (1024.0 * 1024.0)) << " MB" << endl;
    cout << "Time:               " << elapsed << " seconds" << endl;
    if(elapsed > 0)
        cout << "Speed:              " << (g_total_bytes / (1024.0 * 1024.0)) / elapsed << " MB/s" << endl;

    return (g_failed_count > 0) ? 1 : 0;
}


