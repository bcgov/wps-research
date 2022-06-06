/* 20220605 add N-rasters (equal dimensions) together. Adapted from:

raster_mult.cpp: band math, multiply hyperspectral cubes together
adapted from raster_sum.cpp 20220302 */
#include"misc.h"

int main(int argc, char ** argv){
  size_t nrow, ncol, nband, np, nf, i, k, nrow2, ncol2, nband2;

  if(argc < 4){
    err("raster_sum.exe [raster cube1] .. [raster cube #N] [out cube]\n");
  }
  int n_rasters = argc - 2;
  str fn(argv[1]); // input image file name

  vector<vector<str>> bnames;

  for0(i, n_rasters){
    str fn(argv[1 + i]);
    if(!exists(fn)){
      err("please check input files");
    }
    str hfn(hdr_fn(fn));
    hread(hfn, nrow2, ncol2, nband2);
    if(i == 0){
      (nrow = nrow2), (ncol = ncol2), (nband = nband2);
    }
    else{
      if(nrow != nrow2 || ncol != ncol2 || nband != nband2){
        err("input dimensions must match");
      }
    }
    vector<str> band_names(parse_band_names(hdr_fn(fn)));
	    cout << band_names << endl;
    bnames.push_back(band_names);
  }
  np = nrow * ncol;
  nf = np * nband;

  float * out = falloc(nf); // out but
  str ofn(argv[argc -1]); // output file name
  str ohn(hdr_fn(ofn, true)); // out header file name

  for0(i, nf) out[i] = 0.;

  for0(i, n_rasters){
    float * dat = bread(str(argv[1 + i]), nrow, ncol, nband);
    for0(k, nf) out[k] += dat[k];
    free(dat);
  }

  vector<str> bn(bnames[0]);
  bn.clear();
  for0(i, n_rasters){
    if(i > 0){
      for0(k, nband){
        bn[k] += (str(" + ") + bnames[i][k]);
      }
    }
  }

  hwrite(ohn, nrow, ncol, nband, 4, bn);
  bwrite(out, ofn, nrow, ncol, nband);
  free(out);
  return 0;
}
