/* 20220407 subselect SWIR bands only from Sentinel2 */
#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 2){
    err("sentinel2_swir_subselect.exe [input file name .bin]");
  }
  size_t nrow, ncol, nband, np, i, j, k, n, ij;
  float * out, * dat, * b1, * b2, * b3;
  long int bi[3];

  str fn(argv[1]); /* binary files */
  str ofn(fn + "_swir.bin");
  str hfn(hdr_fn(fn)); /* headers */
  str hf2(hdr_fn(ofn, true));
  FILE * f = ropen(argv[1]); //, "rb");

  vector<string> s, t; /* band names + ones to use */
  t.push_back(str("2190nm"));
  t.push_back(str("1610nm"));
  t.push_back(str("945nm"));

  hread(hfn, nrow, ncol, nband, s);
  for0(i, 3){
    bi[i] = -1;
  }
  np = nrow * ncol;
  n = s.size();

  // should abstract this into the misc.h/misc.cpp:
  str date_s;
  for0(i, n){
    for0(j, 3){
      if(contains(s[i], t[j])){
        bi[j] = i * np; /* found a required band */
        printf("bi[%zu]=%zu \"%s\"\n", j, bi[j], s[i].c_str());
        vector<string> w(split(s[i], ' '));
        date_s = w[0]; /* assume datetime string at beginning */
      }
    }
  }
  for0(i, 3){
    if(bi[i] < 0){
      printf("Missing band: %s\n", t[i].c_str());
      err("Missing band");
    }
    else{
      //cout << "i " << i << " bi[i] " << bi[i] << " " << bi[i] / np << endl;
    }
  }
  dat = falloc(np);
  FILE * g = fopen(ofn.c_str(), "wb");
  fclose(g);
  for0(j, 3){
    //printf("write band %d %d %s\n", (int)j, (int)(bi[j]/np), s[bi[j]/np].c_str());
    fseek(f, bi[j] * sizeof(float), SEEK_SET);
    size_t nr = fread(dat, sizeof(float), np, f);

    FILE * g = fopen(ofn.c_str(), "ab");
    if(!g) err("failed to open file");
    size_t nw = fwrite(dat, sizeof(float), np, g);
    fclose(g);
    if(nw != np){
      printf("wrote: %zu expected: %zu\n", nw, np);
      err("bappend: incorrect number of records written");
    }
  }
  hread(hfn, nrow, ncol, nband, s);
  vector<str> bn;
  for0(j, 3){
    bi[j] /= np;
    bn.push_back(s[bi[j]]); //date_s + str(" sentinel2_active.cpp"));
  }
  hwrite(hf2, nrow, ncol, 3, 4, bn);

  /* copy map info */
  str cmd("python3 ~/GitHub/wps-research/py/envi_header_copy_mapinfo.py ");
  cmd += (hfn + str(" ") + hf2);
  cout << cmd << endl;
  system(cmd.c_str());
  free(dat); /* plot spectra? */
  return 0;
}
