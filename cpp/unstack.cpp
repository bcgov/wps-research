/* Band unstacking. Should revise this with band names now
20220308 revised to accept specific bands only 

20260116 revised for parallelism */

/* Band unstacking. Should revise this with band names now
20220308 revised to accept specific bands only */
#include"misc.h"

// Global variables for parfor
size_t g_nr, g_nc, g_nb, g_np;
float * g_d;
str g_ifn;
str g_hfn;
vector<str> g_band_names;
set<int> g_selected;
bool g_use_bn;

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

  g_d = bread(g_ifn, g_nr, g_nc, g_nb); // read input data

  // Initialize mutexes for parfor
  pthread_mutex_init(&print_mtx, NULL);
  pthread_mutex_init(&pt_nxt_j_mtx, NULL);

  parfor(0, g_nb, process_band);

  free(g_d);
  return 0;
}
