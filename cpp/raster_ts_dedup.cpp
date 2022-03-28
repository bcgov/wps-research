/* raster_ts_dedup.cpp: remove sequentially-redundant frames
in a video of (single band) rasters 20220321*/
#include"misc.h"

int main(int argc, char ** argv){
  size_t nrow, ncol, nband, nband2, np, i, j, k, ki, kj, n_min, n_dif;
  if(argc < 2){
    err("raster_ts_dedup.exe [raster cube] [optional arg: min # changed pix]\n");
  }
  n_min = argc > 2 ? (size_t) atol(argv[2]): 0;

  str fn(argv[1]); // input image
  str hfn(hdr_fn(fn)); // input header
  if(!exists(fn)) err("failed to open input file");

  str ofn(fn + str("_dedup.bin")); // output image
  str ohn(hdr_fn(ofn, true)); // output header, create mode

  vector<str> band_names(parse_band_names(hfn));
  vector<str> name_retain; // band names to retain

  hread(hfn, nrow, ncol, nband); // read dimensions from header
  np = nrow * ncol; // number of input pix

  float * dat = bread(fn, nrow, ncol, nband); // read input

  vector<size_t> retain;
  retain.push_back(0);
  cout << retain;
  printf("band %zu\n", 1);
  for0(k, nband - 1){
    n_dif = 0;
    ki = k * np;
    kj = ki + np;
    for0(i, np){
	    float fki = dat[ki];
	    float fkj = dat[kj];
	    if(!isnan(fki) && !isinf(fki) && !isnan(fkj) && !isinf(fkj)){
	    if(dat[ki++] != dat[kj++]) n_dif ++;
	    }
    }
    if((n_min == 0 && n_dif > 0) ||
       (n_min > 0 && n_dif >= n_min)){
      retain.push_back(k + 1);
    }
    cout << retain << " ";
    printf("band %zu npx_chg %zu\n", k + 2, n_dif);
  }
  
  nband2 = retain.size();
  for0(k, nband2) name_retain.push_back(band_names[retain[k]]);

  float * out = falloc(nrow * ncol * nband2); // output buffer
  for0(k, nband2){
    ki = np * retain[k];
    kj = np * k;
    for0(i, np) out[kj + i] = dat[ki + i];
  }

  hwrite(ohn, nrow, ncol, nband2, 4, name_retain); // write out dimensions to header
  bwrite(out, ofn, nrow, ncol, nband2); // write out image
  return 0;
}
