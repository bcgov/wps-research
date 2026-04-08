/* 20260116: prm.cpp: parallel remove ( non-recursive ) with pattern matching
   20260305: added -r flag for recursive directory deletion
   20260408: added -d flag / bare directory support to bypass shell ARG_MAX limit

   Note: shell glob expansion (e.g. prm ./* or prm *.bin) happens BEFORE prm
   runs. On directories with many files this hits the OS ARG_MAX limit:
     "bash: .../prm: Argument list too long"
   To avoid this, pass the directory itself and let prm enumerate internally:
     prm .                       # delete files in cwd (auto -d)
     prm -d /path/to/dir         # delete files in dir (non-recursive)
     prm -r /path/to/dir         # delete files and subdirs recursively
     
Original documentation:

prm.cpp - parallel remove
   Removes files in parallel using parfor construct
   Supports recursive deletion (-r option)
   Supports bulk directory contents deletion (-d option)
*/
#include"misc.h"
#include<glob.h>
#include<unistd.h>
#include<sys/stat.h>
#include<dirent.h>

// Global variables for parfor
vector<str> g_files_to_delete;
pthread_mutex_t g_delete_mtx;
size_t g_deleted_count = 0;
size_t g_failed_count = 0;
size_t g_total_files = 0;

void collect_recursive(const str& path){
  struct stat st;
  if(stat(path.c_str(), &st) != 0) return;

  if(S_ISREG(st.st_mode)){
    g_files_to_delete.push_back(path);
    return;
  }

  if(S_ISDIR(st.st_mode)){
    DIR* dir = opendir(path.c_str());
    if(!dir){
      cout << "Cannot open directory: " << path << endl;
      return;
    }
    struct dirent* entry;
    while((entry = readdir(dir)) != NULL){
      str name(entry->d_name);
      if(name == "." || name == "..") continue;
      collect_recursive(path + "/" + name);
    }
    closedir(dir);
  }
}

/* collect regular files directly inside a directory (non-recursive) */
void collect_dir_contents(const str& path){
  DIR* dir = opendir(path.c_str());
  if(!dir){
    cout << "Cannot open directory: " << path << endl;
    return;
  }
  struct dirent* entry;
  while((entry = readdir(dir)) != NULL){
    str name(entry->d_name);
    if(name == "." || name == "..") continue;
    str full = path + "/" + name;
    struct stat st;
    if(stat(full.c_str(), &st) == 0 && S_ISREG(st.st_mode)){
      g_files_to_delete.push_back(full);
    }
    else if(S_ISDIR(st.st_mode)){
      cout << "Skipping subdirectory: " << full << " (use -r for recursive)" << endl;
    }
  }
  closedir(dir);
}

void delete_file(size_t i){
  str filename = g_files_to_delete[i];

  int result = unlink(filename.c_str());

  mtx_lock(&g_delete_mtx);
  if(result == 0){
    g_deleted_count++;
  }
  else{
    g_failed_count++;
    cout << "Failed to delete: " << filename << " (error: " << strerror(errno) << ")" << endl;
  }

  size_t total_processed = g_deleted_count + g_failed_count;
  if(total_processed % 1000 == 0 || total_processed == g_total_files){
    cout << "Progress: " << total_processed << "/" << g_total_files << " files ("
         << (100 * total_processed / g_total_files) << "%) - "
         << "Deleted: " << g_deleted_count << ", Failed: " << g_failed_count << endl;
  }
  mtx_unlock(&g_delete_mtx);
}

void delete_empty_dirs(const str& path){
  struct stat st;
  if(stat(path.c_str(), &st) != 0) return;
  if(!S_ISDIR(st.st_mode)) return;

  DIR* dir = opendir(path.c_str());
  if(!dir) return;
  struct dirent* entry;
  while((entry = readdir(dir)) != NULL){
    str name(entry->d_name);
    if(name == "." || name == "..") continue;
    delete_empty_dirs(path + "/" + name);
  }
  closedir(dir);

  if(rmdir(path.c_str()) != 0){
    cout << "Failed to remove directory: " << path << " (error: " << strerror(errno) << ")" << endl;
  }
  else{
    cout << "Removed directory: " << path << endl;
  }
}

int main(int argc, char *argv[]){
  if(argc < 2){
    err("Usage: prm [options] <file1> [file2] ... [fileN]\n"
        "       prm [options] <pattern>\n"
        "       prm -r <directory>\n"
        "       prm -d <directory>   (delete contents, non-recursive)\n"
        "       prm <directory>      (same as -d)\n"
        "\n"
        "Options:\n"
        "  -r    Recursively delete all files and folders in a directory\n"
        "  -d    Delete all files inside a directory (non-recursive)\n"
        "\n"
        "Examples:\n"
        "  prm file.txt\n"
        "  prm *.log\n"
        "  prm -r my_folder\n"
        "  prm -d /data/pgfc/MRAP/L2\n"
        "  prm /data/pgfc/MRAP/L2      # same as -d\n"
        "  prm file1.txt file2.txt *.tmp\n");
  }

  bool recursive = false;
  bool dir_contents = false;
  vector<str> args;
  for(int i = 1; i < argc; i++){
    str a(argv[i]);
    if(a == "-r") recursive = true;
    else if(a == "-d") dir_contents = true;
    else args.push_back(a);
  }

  /* if a single bare directory is given with no flags, treat as -d */
  if(!recursive && !dir_contents && args.size() == 1){
    struct stat st;
    if(stat(args[0].c_str(), &st) == 0 && S_ISDIR(st.st_mode)){
      dir_contents = true;
    }
  }

  // Collect all files to delete
  for(auto& pattern : args){
    struct stat st;

    if(recursive && stat(pattern.c_str(), &st) == 0 && S_ISDIR(st.st_mode)){
      cout << "Collecting files recursively from: " << pattern << endl;
      collect_recursive(pattern);
      continue;
    }

    if(dir_contents && stat(pattern.c_str(), &st) == 0 && S_ISDIR(st.st_mode)){
      cout << "Collecting files from directory: " << pattern << endl;
      collect_dir_contents(pattern);
      continue;
    }

    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));
    int ret = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);

    if(ret == 0){
      for(size_t j = 0; j < glob_result.gl_pathc; j++){
        str filepath(glob_result.gl_pathv[j]);
        if(stat(filepath.c_str(), &st) == 0){
          if(S_ISREG(st.st_mode)){
            g_files_to_delete.push_back(filepath);
          }
          else if(S_ISDIR(st.st_mode)){
            cout << "Skipping directory: " << filepath << " (use -r for directories)" << endl;
          }
          else{
            cout << "Skipping non-regular file: " << filepath << endl;
          }
        }
      }
      globfree(&glob_result);
    }
    else if(ret == GLOB_NOMATCH){
      if(stat(pattern.c_str(), &st) == 0){
        if(S_ISREG(st.st_mode)){
          g_files_to_delete.push_back(pattern);
        }
        else if(S_ISDIR(st.st_mode)){
          cout << "Cannot remove directory: " << pattern << " (use -r or -d)" << endl;
        }
      }
      else{
        cout << "No such file: " << pattern << endl;
      }
    }
    else{
      cout << "Error processing pattern: " << pattern << endl;
    }
  }

  if(g_files_to_delete.size() == 0){
    cout << "No files to delete." << endl;
    return 0;
  }

  g_total_files = g_files_to_delete.size();
  cout << "Found " << g_total_files << " files to delete." << endl;

  if(g_total_files > 10){
    cout << "Delete " << g_total_files << " files? (y/n): ";
    str response;
    getline(cin, response);
    trim(response);
    lower(response);
    if(response != "y" && response != "yes"){
      cout << "Deletion cancelled." << endl;
      return 0;
    }
  }

  pthread_mutex_init(&print_mtx, NULL);
  pthread_mutex_init(&pt_nxt_j_mtx, NULL);
  pthread_mutex_init(&g_delete_mtx, NULL);

  cout << "\nDeleting files in parallel..." << endl;
  parfor(0, g_total_files, delete_file);

  // After files are deleted, remove empty directories
  if(recursive){
    cout << "\nRemoving empty directories..." << endl;
    for(auto& pattern : args){
      struct stat st;
      if(stat(pattern.c_str(), &st) == 0 && S_ISDIR(st.st_mode)){
        delete_empty_dirs(pattern);
      }
    }
  }

  cout << "\n=== Deletion Summary ===" << endl;
  cout << "Total files processed: " << (g_deleted_count + g_failed_count) << "/" << g_total_files << endl;
  cout << "Successfully deleted: " << g_deleted_count << " files" << endl;
  cout << "Failed to delete: " << g_failed_count << " files" << endl;

  return (g_failed_count > 0) ? 1 : 0;
}


