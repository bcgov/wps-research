/* 20220830 raster_total.cpp: band math, add bands together */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 2) err("raster_total [raster cube]\n");
  size_t nrow, ncol, nband, np, nrow2, ncol2, nband2;
  size_t i, j, k, ix, ij, ik, m;
  float * out, * dat;

  str fn(argv[1]);
  str ofn(fn + str("_total.bin"));
  str ohn(hdr_fn(ofn, true)); // out header file name

  if(!exists(fn)) err("failed to open input file");
  str hfn(hdr_fn(fn));
  hread(hfn, nrow, ncol, nband); // read header

  np = nrow * ncol; // number of input pix
  out = falloc(np);
  for0(i, np) out[i] = 0.;
    
  dat = bread(fn, nrow, ncol, nband);
    for0(i, nrow){
      ix = i * ncol;
      for0(j, ncol){
        ij = ix + j;
        for0(k, nband){
          out[ij] += dat[ik];
        }
      }
    }

  printf("+w %s\n", ofn.c_str());
  bwrite(out, ofn, nrow, ncol, nband);
  hwrite(ohn, nrow, ncol, nband);
  return 0;
}
