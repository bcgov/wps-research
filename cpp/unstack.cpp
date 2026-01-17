/* Band unstacking. Should revise this with band names now
20220308 revised to accept specific bands only 

20260116 revised for parallelism */

/* Band unstacking. Should revise this with band names now
20220308 revised to accept specific bands only */
#include"misc.h"
#include <sys/stat.h>
#include <sys/time.h>
#include <limits.h>
#include <stdlib.h>

// Global variables for parfor
size_t g_nr, g_nc, g_nb, g_np;
float * g_d;
str g_ifn;
str g_hfn;
vector<str> g_band_names;
set<int> g_selected;
bool g_use_bn;
pthread_mutex_t g_progress_mtx;
size_t g_read_progress = 0;
size_t g_write_progress = 0;

void read_band(size_t i){
  // Seek to the correct position and read the band
  size_t offset = g_np * i * sizeof(float);
  float * band_data = &g_d[g_np * i];

  // Use a temporary file pointer for thread-safe reading
  FILE * f = ropen(g_ifn);
  fseek(f, offset, SEEK_SET);

  // Chunked reading for progress monitoring and better I/O
  size_t chunk_size = 1024 * 1024; // 1MB chunks
  size_t pixels_per_chunk = chunk_size / sizeof(float);
  size_t chunks = (g_np + pixels_per_chunk - 1) / pixels_per_chunk;
  size_t pixels_read = 0;

  for(size_t c = 0; c < chunks; c++){
    size_t pixels_to_read = (c == chunks - 1) ? (g_np - pixels_read) : pixels_per_chunk;
    size_t nr = fread(&band_data[pixels_read], sizeof(float), pixels_to_read, f);
    if(nr != pixels_to_read){
      fclose(f);
      cprint(str("Error reading band ") + to_string(i + 1) + str(" chunk ") + to_string(c));
      err("read_band(): failed to read expected number of pixels");
    }
    pixels_read += nr;
  }

  fclose(f);

  // Update progress
  mtx_lock(&g_progress_mtx);
  g_read_progress++;
  if(g_read_progress % 10 == 0 || g_read_progress == g_nb){
    cout << "Read progress: " << g_read_progress << "/" << g_nb << " bands ("
         << (100 * g_read_progress / g_nb) << "%)" << endl;
  }
  mtx_unlock(&g_progress_mtx);
}

void process_band(size_t i){
  if(g_selected.size() < 1 || g_selected.count((int)(i + 1)) > 0){
    // Extract last component of band name (after splitting on ":")
    str band_name = g_band_names[i];
    vector<str> band_parts = split(band_name, ':');
    str band_suffix = band_parts[band_parts.size() - 1];
    trim(band_suffix);

    // Replace spaces with underscores in band suffix
    std::replace(band_suffix.begin(), band_suffix.end(), ' ', '_');

    // Build output filename with 3-digit prefix and band suffix
    str pre(g_ifn + str("_") + zero_pad(to_string(i + 1), 3) + str("_") + band_suffix);
    str ofn(pre + str(".bin"));
    str ohn(pre + str(".hdr"));

    // Chunked writing
    FILE * f = wopen(ofn.c_str());
    size_t chunk_size = 1024 * 1024; // 1MB chunks
    size_t pixels_per_chunk = chunk_size / sizeof(float);
    size_t chunks = (g_np + pixels_per_chunk - 1) / pixels_per_chunk;
    size_t pixels_written = 0;
    float * band_data = &g_d[g_np * i];

    for(size_t c = 0; c < chunks; c++){
      size_t pixels_to_write = (c == chunks - 1) ? (g_np - pixels_written) : pixels_per_chunk;
      size_t nw = fwrite(&band_data[pixels_written], sizeof(float), pixels_to_write, f);
      if(nw != pixels_to_write){
        fclose(f);
        cprint(str("Error writing band ") + to_string(i + 1) + str(" chunk ") + to_string(c));
        err("process_band(): failed to write expected number of pixels");
      }
      pixels_written += nw;
    }
    fclose(f);

    // Create band name with original band name appended
    str output_band_name = g_ifn + str(":") + g_band_names[i];
    vector<str> band_names_vec;
    band_names_vec.push_back(output_band_name);

    hwrite(ohn, g_nr, g_nc, 1, 4, band_names_vec); // write with custom band name

    str cmd(str("python3 ~/GitHub/wps-research/py/envi_header_copy_mapinfo.py ") +
            g_hfn + str(" ") +
            ohn);
    int ret = system(cmd.c_str());
    if(ret != 0){
      cprint(str("Warning: mapinfo copy command failed for band ") + to_string(i + 1));
    }

    // Update progress
    mtx_lock(&g_progress_mtx);
    g_write_progress++;
    if(g_write_progress % 10 == 0 || g_write_progress == g_nb){
      cout << "Write progress: " << g_write_progress << "/" << g_nb << " bands ("
           << (100 * g_write_progress / g_nb) << "%)" << endl;
    }
    mtx_unlock(&g_progress_mtx);
  }
  else{
    // Update progress even for skipped bands
    mtx_lock(&g_progress_mtx);
    g_write_progress++;
    mtx_unlock(&g_progress_mtx);
  }
}

// Simple thread count detection based on path
int detect_io_channels(const str& filepath){
  // Expand to absolute path
  char abs_path[PATH_MAX];
  char * result = realpath(filepath.c_str(), abs_path);
  str absolute_filepath;

  if(result != NULL){
    absolute_filepath = str(abs_path);
    cout << "Absolute path: " << absolute_filepath << endl;
  }
  else{
    cout << "Warning: could not resolve absolute path, using relative path" << endl;
    absolute_filepath = filepath;
  }

  // Check if path starts with /ram/
  if(absolute_filepath.size() >= 5 && absolute_filepath.substr(0, 5) == str("/ram/")){
    cout << "Detected /ram/ path - using 32 threads" << endl;
    return 32;
  }

  // Default to 4 threads for all other cases
  cout << "Using default 4 threads" << endl;
  return 4;
}

int main(int argc, char *argv[]){
  // Start total time measurement
  struct timeval total_start, total_end;
  gettimeofday(&total_start, NULL);

  if(argc < 2){
    err("unstack [input data file] [optional arg: band 1-ix] ... [optional: band 1-ix]");
  }
  size_t i;
  for(i = 2; i < argc; i++) g_selected.insert(atoi(argv[i]));

  g_ifn = str(argv[1]);
  g_hfn = hdr_fn(g_ifn);

  // Read header and band names in one call to avoid redundancy
  hread(g_hfn, g_nr, g_nc, g_nb, g_band_names);

  /* don't include band names in output filenames
      if band names look like folders*/
  g_use_bn = true;
  for(vector<str>::iterator it = g_band_names.begin();
      it != g_band_names.end();
      it++){
    str x(*it);
    vector<str> y(split(x, ' '));
    vector<str> z(split(x, '/'));
    if(y.size() > 1 || z.size() > 1) g_use_bn = false;
  }

  g_np = g_nr * g_nc;
  for(set<int>::iterator it = g_selected.begin();
      it != g_selected.end();
      it++){
    if(*it < 1 || *it > g_nb){
      err("selected band out of bounds");
    }
  }

  // Detect optimal I/O parallelism
  int io_channels = detect_io_channels(g_ifn);

  // Allocate memory for all bands
  size_t nf = g_nr * g_nc * g_nb;
  g_d = falloc(nf);

  // Initialize mutexes for parfor and progress tracking
  pthread_mutex_init(&print_mtx, NULL);
  pthread_mutex_init(&pt_nxt_j_mtx, NULL);
  pthread_mutex_init(&g_progress_mtx, NULL);

  // Read input data in parallel with detected I/O channel capacity
  cout << "\n=== Reading input file in parallel ===" << endl;
  g_read_progress = 0;
  struct timeval read_start, read_end;
  gettimeofday(&read_start, NULL);
  parfor(0, g_nb, read_band, io_channels);
  gettimeofday(&read_end, NULL);
  double read_time = (read_end.tv_sec - read_start.tv_sec) +
                     (read_end.tv_usec - read_start.tv_usec) / 1e6;
  cout << "Read complete: " << g_nb << " bands in " << read_time << " seconds" << endl;

  // Write output bands in parallel with detected I/O channel capacity
  cout << "\n=== Writing output files in parallel ===" << endl;
  g_write_progress = 0;
  struct timeval write_start, write_end;
  gettimeofday(&write_start, NULL);
  parfor(0, g_nb, process_band, io_channels);
  gettimeofday(&write_end, NULL);
  double write_time = (write_end.tv_sec - write_start.tv_sec) +
                      (write_end.tv_usec - write_start.tv_usec) / 1e6;
  cout << "Write complete: " << g_nb << " bands in " << write_time << " seconds" << endl;

  gettimeofday(&total_end, NULL);
  double total_time = (total_end.tv_sec - total_start.tv_sec) +
                      (total_end.tv_usec - total_start.tv_usec) / 1e6;

  cout << "\nTotal time: " << total_time << " seconds" << endl;

  free(g_d);
  return 0;
}
