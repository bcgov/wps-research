/* raster_smult.cpp: band math, scalar multiply a raster 
adapted from raster_mult.cpp 20220807 */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 3) err("raster_sum.exe [raster cube] [scalar value to multiply] [optional linear offset]\n");
  str fn(argv[1]); // input image file name
  str hfn(hdr_fn(fn));

  float scalar = atof(argv[2]);
  float offset = 0.;
  if(argc > 3) offset = atof(argv[3]);
  if(!(exists(fn))) err("failed to open input file");

  str ofn(fn + str("_smult.bin"));  // output file name
  str ohn(hdr_fn(ofn, true)); // out header file name

  size_t nrow, ncol, nband, np, nf, i, k;
  hread(hfn, nrow, ncol, nband); // read header 1

  np = nrow * ncol; // # input pix
  nf = np * nband; // # input float
  float * out = falloc(nf); // out buffer
  float * dat = bread(fn, nrow, ncol, nband);
  for0(i, nf) out[i] = dat[i] * scalar + offset;
  bwrite(out, ofn, nrow, ncol, nband);
  hwrite(ohn, nrow, ncol, nband);
  return 0;
}
