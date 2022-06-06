/* 20220605 round raster to 1. */
#include"misc.h"

int main(int argc, char ** argv){
  size_t nrow, ncol, nband, np, nf, i;
  if(argc < 2) err("raster_round1.exe [raster cube1]\n");
  str fn(argv[1]); // input image file name
  str hn(hdr_fn(fn));

  hread(hn, nrow, ncol, nband);
  np = nrow * ncol;
  nf = np * nband;

  float *dat = bread(fn, nrow, ncol, nband);
  float d, *out = falloc(nf); // out buf

  for0(i, nf){
    (d = dat[i]), (out[i] = 0);
    if(isnan(d) || isinf(d)) out[i] = NAN;
    else out[i] = (d < .5)? 0.: 1.;
  }
  bwrite(out, fn, nrow, ncol, nband);
  free(out);
  return 0;
}
