/* replace a given class value with NAN */
#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 2) err("class_replace_nan [input binary (classification) file name]");
  int start_0 = argc > 2;

  str fn(argv[1]); // input file name
  str ofn(fn + str("_recode.bin"));
  str ohfn(fn +str("_recode.hdr"));

  str hfn(hdr_fn(fn)); // auto-detect header file name
  size_t nrow, ncol, nband, np, i, j;
  hread(hfn, nrow, ncol, nband); // read header

  np = nrow * ncol;
  if(nband != 1) err("this program defines results for 1-band images");

  // read data into float array
  float * dat = bread(fn, nrow, ncol, nband);
  for0(i, np){
    if(dat[i] == 0.)
      dat[i] = NAN;
  }

  hwrite(ohfn, nrow, ncol, 1); // write header
  FILE * f = fopen(ofn.c_str(), "wb");
  fwrite(dat, sizeof(float), np, f);
  fclose(f);
  free(dat);
  return 0;
}
