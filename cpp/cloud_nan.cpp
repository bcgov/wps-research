/* 20221026 adapted from raster_sub.cpp

Replace data values with NAN at locations where cloud probability file is GEQ 7. */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 3) err("raster_sub [raster cube 1] [cloud probability band]\n");

  str fn(argv[1]); // input image file name
  str fn2(argv[2]); // input image 2 file name
  if(!(exists(fn) && exists(fn2))) err("failed to open input file");

  str hfn(hdr_fn(fn)); // input header file name
  str hfn2(hdr_fn(fn2)); // input 2 header file name

  size_t nrow, ncol, nband, np, nrow2, ncol2, nband2;
  hread(hfn, nrow, ncol, nband); // read header 1
  hread(hfn2, nrow2, ncol2, nband2); // read header 2
  if(nrow != nrow2 || ncol != ncol2)
    err("input image dimensions should match");
  if(nband2 != 1) err("mask should have one band");

  np = nrow * ncol; // number of input pix
  size_t i, j, k, ix, ij, ik;
  float * dat1 = bread(fn, nrow, ncol, nband);
  float * dat2 = bread(fn2, nrow, ncol, nband2);
  float d;

  for0(i, nrow){
    ix = i * ncol;
    for0(j, ncol){
      ij = ix + j;
      d = dat2[ij];
      if(d > 7. && (!(isnan(d) || isinf(d)))){
        for0(k, nband){
          ik = ij + k * np;
          dat1[ik] = NAN;
        }
      }
    }
  }

  bwrite(dat1, fn, nrow, ncol, nband);
  return 0;
}
