/* raster_accumulate.cpp:
20220320: cumulative sum (by band index) of bands */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 2) err("raster_sum.exe [raster cube]\n");

  str fn(argv[1]); // input image file name
  if(!exists(fn)) err("failed to open input file");
  str hfn(hdr_fn(fn)); // input header file name
  
  str ofn(fn + str("_accumulate.bin")); // output file name
  str ohn(hdr_fn(ofn, true)); // out header file name

  size_t nrow, ncol, nband, np;
  hread(hfn, nrow, ncol, nband); // read header 1

  np = nrow * ncol; // number of input pix
  size_t i, j, k, ix, ij, ik;
  float * out = falloc(nrow * ncol * nband);
  float * dat = bread(fn, nrow, ncol, nband);

  for0(k, nband){
   if(k == 0) for0(i, np) out[i] = dat[i];
   else for0(i, np) out[i] = dat[i] + out[i - np];
  }
  bwrite(out, ofn, nrow, ncol, nband);
  hwrite(ohn, nrow, ncol, nband);
  return 0;
}
