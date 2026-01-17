/* 20260116: prm.cpp: parallel remove ( non-recursive ) with pattern matching
*/

/* prm.cpp - parallel remove
   Removes files in parallel using parfor construct
   Does not support recursive deletion (-r option)
*/
#include"misc.h"
#include<glob.h>
#include<unistd.h>
#include<sys/stat.h>

// Global variables for parfor
vector<str> g_files_to_delete;
pthread_mutex_t g_delete_mtx;
size_t g_deleted_count = 0;
size_t g_failed_count = 0;
size_t g_total_files = 0;

void delete_file(size_t i){
  str filename = g_files_to_delete[i];

  // Attempt to delete the file
  int result = unlink(filename.c_str());

  mtx_lock(&g_delete_mtx);
  if(result == 0){
    g_deleted_count++;
  }
  else{
    g_failed_count++;
    cout << "Failed to delete: " << filename << " (error: " << strerror(errno) << ")" << endl;
  }

  // Update progress
  size_t total_processed = g_deleted_count + g_failed_count;
  if(total_processed % 100 == 0 || total_processed == g_total_files){
    cout << "Progress: " << total_processed << "/" << g_total_files << " files ("
         << (100 * total_processed / g_total_files) << "%) - "
         << "Deleted: " << g_deleted_count << ", Failed: " << g_failed_count << endl;
  }
  mtx_unlock(&g_delete_mtx);
}

int main(int argc, char *argv[]){
  if(argc < 2){
    err("Usage: prm <file1> [file2] ... [fileN]\n"
        "       prm <pattern>\n"
        "Removes files in parallel. Does not support recursive deletion.");
  }

  // Collect all files to delete
  for(int i = 1; i < argc; i++){
    str pattern(argv[i]);

    // Use glob to expand patterns (e.g., *.txt)
    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));

    int ret = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);

    if(ret == 0){
      // Pattern matched files
      for(size_t j = 0; j < glob_result.gl_pathc; j++){
        str filepath(glob_result.gl_pathv[j]);

        // Check if it's a regular file (not a directory)
        struct stat st;
        if(stat(filepath.c_str(), &st) == 0){
          if(S_ISREG(st.st_mode)){
            g_files_to_delete.push_back(filepath);
          }
          else if(S_ISDIR(st.st_mode)){
            cout << "Skipping directory: " << filepath << " (use rm -r for directories)" << endl;
          }
          else{
            cout << "Skipping non-regular file: " << filepath << endl;
          }
        }
      }
      globfree(&glob_result);
    }
    else if(ret == GLOB_NOMATCH){
      // No matches - check if it's a specific file
      struct stat st;
      if(stat(pattern.c_str(), &st) == 0){
        if(S_ISREG(st.st_mode)){
          g_files_to_delete.push_back(pattern);
        }
        else if(S_ISDIR(st.st_mode)){
          cout << "Cannot remove directory: " << pattern << " (use rm -r for directories)" << endl;
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

  // Ask for confirmation if deleting many files
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

  // Initialize mutexes
  pthread_mutex_init(&print_mtx, NULL);
  pthread_mutex_init(&pt_nxt_j_mtx, NULL);
  pthread_mutex_init(&g_delete_mtx, NULL);

  // Delete files in parallel
  cout << "\nDeleting files in parallel..." << endl;
  parfor(0, g_total_files, delete_file);

  // Final progress update
  cout << "\n=== Deletion Summary ===" << endl;
  cout << "Total files processed: " << (g_deleted_count + g_failed_count) << "/" << g_total_files << endl;
  cout << "Successfully deleted: " << g_deleted_count << " files" << endl;
  cout << "Failed to delete: " << g_failed_count << " files" << endl;

  return (g_failed_count > 0) ? 1 : 0;
}
