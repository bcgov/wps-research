/* multiply two images value by value, not interleave dependent but assuming data in same dimensions / format */
#include"misc.h"
int main(int argc, char ** argv){

  if(argc < 2){
    err("multiply [image 1] [image 2]");
  }

  size_t nrow, ncol, nband, nf, i, j, k;
  size_t nrow2, ncol2, nband2;
  str fn(argv[1]); // input file name
  str hfn(hdr_fn(fn)); // auto-detect header file name
  hread(hfn, nrow, ncol, nband); // read header

  str fn2(argv[2]); // input file name
  str hfn2(hdr_fn(fn2)); // auto-detect header file name
  hread(hfn2, nrow2, ncol2, nband2); // read header

  if(!(nrow == nrow2 && ncol == ncol2 && nband == nband2)){
    err("image dimensions must match");
  }
  nf = nrow * ncol * nband;

  // read data into float array
  float * dat = bread(fn, nrow, ncol, nband);
  float * da2 = bread(fn2, nrow, ncol, nband);
  float * out = falloc(nf);

  for0(i, nf){
    out[i] = dat[i] * da2[i];
  }

  str ofn(fn + str("_multiply.bin"));
  str ohfn(fn + str("_multiply.hdr"));
  hwrite(ohfn, nrow, ncol, nband); // write header

  FILE * f = fopen(ofn.c_str(), "wb");
  fwrite(&out[0], sizeof(float), nf, f);
  fclose(f);
  free(dat);
  return 0;
}