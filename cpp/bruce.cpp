/* 20241122 use Dr. Bruce Chapman's (L1) normalization / color scheme for 3-band imagery */
#include"misc.h"
int main(int argc, char ** argv){

  if(argc < 2) err("bruce [input binary file name]");
  str fn(argv[1]); // input ENVI-format filename
  str hfn(hdr_fn(fn)); // detect header file name

  size_t nrow, ncol, nband, np, i, k; // variables
  hread(hfn, nrow, ncol, nband); // read header

  float * dat = bread(fn, nrow, ncol, nband); // load float array
  np = nrow * ncol; // number of pixels
  float total; // L1 quantity

  for0(i, np){
    total = 0.;

    for0(k, nband)
      total += dat[k * np + i];

    for0(k, nband)
      dat[k * np + i] /= total;
  }

  str ofn(fn + str("_bruce.bin"));
  str ohfn(fn + str("_bruce.hdr"));
  hwrite(ohfn, nrow, ncol, nband); // write output hdr
  bwrite(dat, ofn, nrow, ncol, nband); // write output data
  free(dat);
  return 0;
}
