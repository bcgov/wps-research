/* 20241122 use Dr. Bruce Chapman's (L1) normalization / color scheme for 3-band imagery */
#include"misc.h"
int main(int argc, char ** argv){

  if(argc < 2) err("bruce [input binary file name]");
  str fn(argv[1]); // input ENVI-format filename
  str hfn(hdr_fn(fn)); // auto-detect header file name

  size_t nrow, ncol, nband, np, i, k; // variables
  hread(hfn, nrow, ncol, nband); // read heade

  float * dat = bread(fn, nrow, ncol, nband); // load array of floats
  np = nrow * ncol;
  float total;

  for0(i, np){
    total = 0.;
    for0(k, nband) total += dat[k * np + i];
    for0(k, nband) dat[k * np + i] /= total;
  }

  str ofn(fn + str("_bruce.bin"));
  str ohfn(fn + str("_bruce.hdr"));
  printf("nr %zu nc %zu nband %zu\n", nrow, ncol, nband);
  bwrite(dat, ofn, nrow, ncol, nband);
  hwrite(ohfn, nrow, ncol, nband); // write output header
  free(dat);
  return 0;
}
