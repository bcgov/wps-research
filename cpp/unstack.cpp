/* Band unstacking. Should revise this with band names now
20220308 revised to accept specific bands only 

20260116 revised for parallelism */

/* Band unstacking. Should revise this with band names now
20220308 revised to accept specific bands only */
#include"misc.h"
#include <sys/statvfs.h>
#include <sys/stat.h>

// Global variables for parfor
size_t g_nr, g_nc, g_nb, g_np;
float * g_d;
str g_ifn;
str g_hfn;
vector<str> g_band_names;
set<int> g_selected;
bool g_use_bn;
FILE * g_input_file;

void read_band(size_t i){
  // Seek to the correct position and read the band
  size_t offset = g_np * i * sizeof(float);
  float * band_data = &g_d[g_np * i];

  // Use a temporary file pointer for thread-safe reading
  FILE * f = ropen(g_ifn);
  fseek(f, offset, SEEK_SET);
  size_t nr = fread(band_data, sizeof(float), g_np, f);
  fclose(f);

  if(nr != g_np){
    cprint(str("Error reading band ") + to_string(i + 1));
    err("read_band(): failed to read expected number of pixels");
  }
}

void process_band(size_t i){
  if(g_selected.size() < 1 || g_selected.count((int)(i + 1)) > 0){
    str pre(g_ifn + str("_") + zero_pad(to_string(i + 1), 3));
    if(g_use_bn) pre += (str("_") + g_band_names[i]);
    str ofn(pre + str(".bin"));
    str ohn(pre + str(".hdr"));
    FILE * f = wopen(ofn.c_str());
    fwrite(&g_d[g_np * i], sizeof(float), g_np, f);
    fclose(f);

    // Create band name with original band name appended
    str output_band_name = g_ifn + str(":") + g_band_names[i];
    vector<str> band_names_vec;
    band_names_vec.push_back(output_band_name);

    hwrite(ohn, g_nr, g_nc, 1, 4, band_names_vec); // write with custom band name

    str cmd(str("python3 ~/GitHub/wps-research/py/envi_header_copy_mapinfo.py ") +
            g_hfn + str(" ") +
            ohn);
    cout << cmd << endl;
    system(cmd.c_str());
  }
}

// Detect storage type and return optimal number of concurrent I/O operations
int detect_io_channels(const str& filepath){
  struct statvfs vfs;
  if(statvfs(filepath.c_str(), &vfs) != 0){
    cout << "Warning: Could not detect filesystem type, defaulting to 2 I/O channels" << endl;
    return 2;
  }

  // Try to detect if this is a RAM disk or SSD/HDD
  // Check if filesystem is in /dev/shm (RAM disk) or tmpfs
  str cmd = str("df -T ") + filepath + str(" 2>/dev/null | tail -1");
  str df_output = exec(cmd.c_str());

  int io_channels = 2; // default for HDD
  str fs_type = "";

  if(df_output.size() > 0){
    vector<str> fields = split(df_output, ' ');
    if(fields.size() >= 2){
      fs_type = fields[1];
      lower(fs_type);

      // RAM-based filesystems
      if(contains(fs_type, "tmpfs") || contains(fs_type, "ramfs") ||
         contains(df_output, "/dev/shm")){
        io_channels = sysconf(_SC_NPROCESSORS_ONLN); // Use all cores for RAM
        cout << "Detected RAM-based storage (" << fs_type << ")" << endl;
      }
      // Check for SSD indicators
      else{
        // Try to determine if it's SSD or HDD
        // Extract device name
        str device = "";
        if(fields.size() >= 1){
          device = fields[0];
          // Remove partition number to get base device
          size_t i = device.size() - 1;
          while(i > 0 && isdigit(device[i])) i--;
          device = device.substr(0, i + 1);

          // Check rotational attribute (0 = SSD, 1 = HDD)
          str rot_file = str("/sys/block/") + device.substr(device.rfind('/') + 1) + str("/queue/rotational");
          if(exists(rot_file)){
            str rot_cmd = str("cat ") + rot_file + str(" 2>/dev/null");
            str rotational = exec(rot_cmd.c_str());
            trim(rotational);

            if(rotational == str("0")){
              io_channels = 8; // SSD can handle more concurrent I/O
              cout << "Detected SSD storage" << endl;
            }
            else{
              io_channels = 2; // HDD limited to 2 concurrent reads
              cout << "Detected HDD storage" << endl;
            }
          }
          else{
            // Fallback: assume SSD if we can't determine
            io_channels = 4;
            cout << "Storage type uncertain, using moderate parallelism" << endl;
          }
        }
      }
    }
  }

  cout << "Effective I/O channel capacity: " << io_channels << " concurrent operations" << endl;
  return io_channels;
}

int main(int argc, char *argv[]){
  if(argc < 2){
    err("unstack [input data file] [optional arg: band 1-ix] ... [optional: band 1-ix]");
  }
  size_t i;
  for(i = 2; i < argc; i++) g_selected.insert(atoi(argv[i]));

  g_ifn = str(argv[1]);
  g_hfn = hdr_fn(g_ifn);
  hread(g_hfn, g_nr, g_nc, g_nb);
  g_band_names = parse_band_names(g_hfn);

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

  // Initialize mutexes for parfor
  pthread_mutex_init(&print_mtx, NULL);
  pthread_mutex_init(&pt_nxt_j_mtx, NULL);

  // Read input data in parallel with detected I/O channel capacity
  cout << "Reading input file in parallel..." << endl;
  parfor(0, g_nb, read_band, io_channels);

  // Write output bands in parallel with detected I/O channel capacity
  cout << "Writing output files in parallel..." << endl;
  parfor(0, g_nb, process_band, io_channels);

  free(g_d);
  return 0;
}
