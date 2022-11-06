/* invert binary class map */
#include"misc.h"
int main(int argc, char ** argv){

  if(argc < 2){
    err("binary_invert [input binary class file name]");
  }

  float d;
  str fn(argv[1]); // input file name
  str hfn(hdr_fn(fn)); // auto-detect header file name
  size_t nrow, ncol, nband, np, i, j;
  hread(hfn, nrow, ncol, nband); // read header

  np = nrow * ncol;
  if(nband != 1) err("this program defines results for 1-band images");

  // read data into float array
  float * dat = bread(fn, nrow, ncol, nband);

  // check consistency of binary class map
  for0(i, np){
    d = dat[i];
    if(isnan(d)) continue;
    if(d != 0. && d != 1.){
      cout << "warning: value detected: " << d << endl;
      err("please check consistency of binary class map.");
    }
  }

  // write output file
  str ofn(fn + str("_invert.bin"));
  str ohfn(fn + str("_invert.hdr"));

  hwrite(ohfn, nrow, ncol, 1); // write output header
  FILE * f = fopen(ofn.c_str(), "wb");

  // write the inverted data
  for0(i, np){
    d = dat[i];
    if(!isnan(d)){
      d = (d == 0.) ? 1. : 0.;
    }
    fwrite(&d, sizeof(float), 1, f);
  }

  fclose(f);
  return 0;
}
