/* 20220830: from raster_negate.cpp */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 2){
    err("raster_recip [hyperspec cube] # reciprocate the values\n");
  }

  str fn(argv[1]); // input image file name
  if(!(exists(fn))) err("failed to open input file");
  str hfn(hdr_fn(fn)); // input header file name

  str ofn(str(argv[1]) + str("_recip.bin")); // output file name
  str ohn(hdr_fn(ofn, true)); // out header file name

  printf("+w %s\n", ofn.c_str());
  printf("+w %s\n", ohn.c_str());

  size_t nrow, ncol, nband, np;
  hread(hfn, nrow, ncol, nband); // read header 1
  np = nrow * ncol; // number of input pix

  size_t n, i, j, k, ix, ij, ik;
  float * out = falloc(nrow * ncol * nband);
  float * dat = bread(fn, nrow, ncol, nband);

  for0(k, nband){
    ik = np * k;
    for0(i, nrow){
      ix = (i * ncol) + ik;
      for0(j, ncol){
        ij = ix + j;
        out[ij] = 1. / dat[ij];
      }
    }
  }
  bwrite(out, ofn, nrow, ncol, nband);
  hwrite(ohn, nrow, ncol, nband);
  return 0;
}
