/* raster_increment.cpp: band math, add 1. to hyperspectral cube */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 3) err("raster_sum.exe [raster cube] [output cube]\n");

  str fn(argv[1]); // input image file name
  if(!exists(fn)) err("failed to open input file");

  str hfn(hdr_fn(fn)); // input header file name
  str ofn(argv[2]); // output file name
  str ohn(hdr_fn(ofn, true)); // out header file name

  size_t nrow, ncol, nband, np, nrow2, ncol2, nband2;
  hread(hfn, nrow, ncol, nband); // read header 1

  np = nrow * ncol; // number of input pix
  size_t i, j, k, ix, ij, ik;
  float * out = falloc(nrow * ncol * nband);
  float * dat1 = bread(fn, nrow, ncol, nband);

  for0(i, nrow){
    ix = i * ncol;
    for0(j, ncol){
      ij = ix + j;
      for0(k, nband){
        ik = ij + k * np;
        out[ik] = dat1[ik] + 1.;
      }
    }
  }

  bwrite(out, ofn, nrow, ncol, nband);
  hwrite(ohn, nrow, ncol, nband);
  return 0;
}
