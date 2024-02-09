/* 20240208 l2 norm of raster */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 2) err("raster_normalize [input binary file name]");

  str fn(argv[1]); // input file name
  str hfn(hdr_fn(fn)); // auto-detect header file name
  size_t nrow, ncol, nband, np, i, j, k, n, ix, jx; // variables
  hread(hfn, nrow, ncol, nband); // read header
  np = nrow * ncol; // number of input pix
  size_t nf = np * nband;
  float * dat = bread(fn, nrow, ncol, nband); // load floats to array
  float mn, mx, d, r;

  float * out = falloc(nrow * ncol);
  for0(i, nrow){
    for0(j, ncol){
      (ix = i * ncol + j), (mn = 0);
      for0(k, nband){
        d = dat[k * np + ix];
        if(!(isnan(d) || isinf(d))) mn += d * d;
      }
      mn = sqrt(mn);
      out[ix] = mn;
    }
  }

  // write output file
  str ofn(fn + str("_abs.bin"));
  str ohfn(fn + str("_abs.hdr"));

  printf("nr %zu nc %zu nband %zu\n", nrow, ncol, nband);
  hwrite(ohfn, nrow, ncol, 1); // write output header

  FILE * f = fopen(ofn.c_str(), "wb");
  if(!f) err("failed to open output file");
  fwrite(out, sizeof(float), nrow * ncol, f); // write data
  fclose(f);
  free(dat);
  return 0;
}
