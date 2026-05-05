/* 20260505: raster_remove_bands.cpp: remove specified bands from ENVI BSQ float32 raster
   Usage: raster_remove_bands input.bin band1 [band2 ...] (1-indexed)
   Output: input.bin_remove_bands_1_2.bin / .hdr */
#include"misc.h"

size_t g_nr, g_nc, g_nb, g_np;
float ** g_bands;
str g_ifn;

void read_band(size_t i){
  FILE * f = ropen(g_ifn);
  fseek(f, g_np * i * sizeof(float), SEEK_SET);
  size_t nr = fread(g_bands[i], sizeof(float), g_np, f);
  fclose(f);
  if(nr != g_np) err(str("read_band(): failed on band ") + to_string(i + 1));
}

int main(int argc, char *argv[]){
  if(argc < 3) err("raster_remove_bands [input.bin] [band1] [band2] ...");

  g_ifn = str(argv[1]);
  str hfn(hdr_fn(g_ifn));

  /* collect bands to remove */
  set<int> remove;
  for(int i = 2; i < argc; i++) remove.insert(atoi(argv[i]));

  /* read header */
  vector<str> band_names;
  hread(hfn, g_nr, g_nc, g_nb, band_names);
  g_np = g_nr * g_nc;

  for(set<int>::iterator it = remove.begin(); it != remove.end(); it++)
    if(*it < 1 || *it > (int)g_nb) err("band index out of range");

  /* allocate and read all bands in parallel */
  g_bands = (float **)alloc(g_nb * sizeof(float *));
  for(size_t i = 0; i < g_nb; i++) g_bands[i] = falloc(g_np);

  pthread_mutex_init(&print_mtx, NULL);
  pthread_mutex_init(&pt_nxt_j_mtx, NULL);
  parfor(0, g_nb, read_band, 4);

  /* build output filename: input.bin_remove_bands_1_3.bin */
  str suffix("_remove_bands");
  for(set<int>::iterator it = remove.begin(); it != remove.end(); it++)
    suffix += str("_") + to_string(*it);
  str ofn(g_ifn + suffix + str(".bin"));
  str ohn(g_ifn + suffix + str(".hdr"));

  /* write kept bands sequentially */
  FILE * fo = wopen(ofn);
  vector<str> kept_names;
  for(size_t i = 0; i < g_nb; i++){
    if(remove.count((int)(i + 1)) == 0){
      fwrite(g_bands[i], sizeof(float), g_np, fo);
      kept_names.push_back(band_names[i]);
    }
  }
  fclose(fo);

  /* write output header: read original, patch bands count and band names */
  ifstream hf(hfn);
  if(!hf.is_open()) err("cannot open input header");
  ofstream of(ohn);
  if(!of.is_open()) err("cannot open output header");

  str line;
  bool in_band_names = false;
  while(getline(hf, line)){
    if(in_band_names){
      /* skip original band name lines until closing brace */
      if(line.find('}') != string::npos) in_band_names = false;
      continue;
    }
    /* patch bands count */
    vector<str> kv = split(line, '=');
    if(kv.size() == 2){
      str key(kv[0]);
      trim(key);
      if(key == "bands"){
        of << "bands = " << kept_names.size() << endl;
        continue;
      }
      if(key == "band names"){
        /* write new band names block */
        of << "band names = {" << kept_names[0];
        for(size_t i = 1; i < kept_names.size(); i++)
          of << ",\n" << kept_names[i];
        of << "}" << endl;
        /* if opening brace line didn't have closing brace, skip rest */
        if(line.find('}') == string::npos) in_band_names = true;
        continue;
      }
    }
    of << line << "\n";
  }
  hf.close();
  of.close();

  /* free */
  for(size_t i = 0; i < g_nb; i++) free(g_bands[i]);
  free(g_bands);

  cout << "Wrote " << kept_names.size() << " of " << g_nb << " bands to " << ofn << endl;
  return 0;
}
