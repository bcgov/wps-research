/* generalizing an idea found in dr. Dey's paper, to more dimensions..
..but starting with the simpler case */
#include"misc.h"

// class to sort the values' coordinate indices..

// entropy.. ?

int main(int argc, char ** argv){
  if(argc < 2) err("raster_inequality [input binary file name]");

  str fn(argv[1]); // input file name
  str hfn(hdr_fn(fn)); // auto-detect header file name
  size_t nrow, ncol, nband, np, i, j, k, n, ip, jp, ix1, ix2; // variables
  hread(hfn, nrow, ncol, nband); // read header
  np = nrow * ncol; // number of input pix

  if(nband < 2) err("need at least 2 bands..");

  float * dat = bread(fn, nrow, ncol, nband); // load floats to array
  float * out = falloc(nrow * ncol); // output channel

  for0(i, np){
    out[i] = 0;
    float d_max = FLT_MIN;
    int   i_max = -1;
    for0(k, nband){
      d = dat[k * np + i];
      if(isnan(d) || isinf(d)){
      }
      else{
        if(d > d_max){
            d_max = d;
            i_max = k;
        }
      }
    }
    if(i_max >= 0) out[i] = i_max + 1;
  }

  str ofn(fn + str("_inequality.bin"));
  str ohfn(fn + str("_inequality.hdr"));

  FILE * f = fopen(ofn.c_str(), "wb");
  if(!f) err("failed to open output file");
  fwrite(out, sizeof(float), np, f); // write data
  fclose(f);
  free(dat);
  return 0;
}
