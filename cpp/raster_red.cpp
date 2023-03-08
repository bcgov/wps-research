/* raster_sub.cpp: band math, subtract hyperspectral cubes */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 2) err("red [raster cube 1]\n");

  str fn(argv[1]); // input image file name
  if(!(exists(fn))) err("failed to open input file");

  str hfn(hdr_fn(fn)); // input header file name
  str ofn(fn + str("_red.bin"));
  str ohn(fn + str("_red.hdr"));

  size_t nrow, ncol, nband, np;
  hread(hfn, nrow, ncol, nband); // read header 1
  if(nband != 3) err("three band image expected: rgb image");

  np = nrow * ncol; // number of input pix
  size_t i, j, k, ix, ij, ik;
  float * out = falloc(np * 3);
  float * dat = bread(fn, nrow, ncol, nband);

  float H, S, V, r, g, b;
  for0(i, np){
     r = dat[i];
     g = dat[i + np];
     b = dat[i + np + np];

     rgb_to_hsv(r, g, b, &H, &S, &V);

     if(H < 60. || H > 300){
       out[i] = 1.;
       out[i + np] = 0.;
       out[i + np + np] = 0.;
     }
     else{
       out[i] = 0.;
       out[i + np] = 0.;
       out[i + np + np] = 0.;

     }
  }

  bwrite(out, ofn, nrow, ncol, 1);
  hwrite(ohn, nrow, ncol, 1);
  return 0;
}
