/* normalize a multispectral image or radar stack, 
input assumed ENVI type-4 32-bit IEEE standard floating-point format, BSQ
interleave

global normalization (all bands same scaling)

20221027 check how this relates to imv default scaling? */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 2) err("raster_normalize2 [input binary file name]");

  str fn(argv[1]); // input file name
  str hfn(hdr_fn(fn)); // auto-detect header file name
  size_t nrow, ncol, nband, np, nf, i;
  hread(hfn, nrow, ncol, nband); // read header
  np = nrow * ncol; // number of input pix
  nf = np * nband;  // number of floats

  float * dat = bread(fn, nrow, ncol, nband); // load floats to array
  float mn = FLT_MAX;
  float mx = -FLT_MAX;
  float d, r;

  for0(i, nf){
    d = dat[i];
    if(isnan(d) || isinf(d)){
    }
    else{
      if(d < mn) mn = d;
      if(d > mx) mx = d;
    }
  }
  r = 1. / (mx - mn);
  for0(i, nf) dat[i] = r * (dat[i] - mn);

  // write output file
  str ofn(fn + str("_norm2.bin"));
  str ohfn(fn + str("_norm2.hdr"));

  printf("nr %zu nc %zu nband %zu\n", nrow, ncol, nband);
  hwrite(ohfn, nrow, ncol, nband); // write output header

  FILE * f = fopen(ofn.c_str(), "wb");
  if(!f) err("failed to open output file");
  fwrite(dat, sizeof(float), nf, f);
  fclose(f);
  free(dat);
  return 0;
}
