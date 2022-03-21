/* raster_accumulate.cpp: cumulative sum (by band) of single-band
raster sequence 20220320 */
#include"misc.h"

int main(int argc, char ** argv){
  size_t nrow, ncol, nband, np, i, j, k, ii, ik;
  if(argc < 2){
    err("raster_accumulate.exe [raster cube] [optional arg: max(result, 1.)] \n");
  }
  int max_1 = argc > 2;

  str fn(argv[1]); // input image
  str hfn(hdr_fn(fn)); // input header
  if(!exists(fn)) err("failed to open input file");

  str ofn(fn + str("_accumulate.bin")); // output image
  str ohn(hdr_fn(ofn, true)); // output header, create mode

  hread(hfn, nrow, ncol, nband); // read dimensions from header
  np = nrow * ncol; // number of input pix

  float * dat = bread(fn, nrow, ncol, nband); // read input
  float * out = falloc(nrow * ncol * nband); // output buffer

  for0(k, nband){
    ik = np * k;
    if(k == 0) for0(i, np) out[i] = dat[i]; //first band same
    else{
      for0(i, np){
        ii = i + ik;
        out[ii] = dat[ii] + out[ii - np]; // add this band + last result
        if(max_1 && out[ii] > 1.) out[ii] = 1.; // max result = 1.
      }
    }
  }
  hwrite(ohn, nrow, ncol, nband); // write out dimensions to header
  bwrite(out, ofn, nrow, ncol, nband); // write out image
  return 0;
}