/* raster_ni: raster normalized index (input 2 band float type) 20220308 */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 2)
    err("raster_ni.exe [raster cube, 2 bands\n");

  str fn(argv[1]); // input image file
  if(!(exists(fn))) err("failed to open input");

  str hfn(hdr_fn(fn)); // input header file 
  str ofn(fn + str("_ni.bin")); // output file
  str ohn(hdr_fn(ofn, true)); // out header file

  size_t nrow, ncol, nband, np, i;
  hread(hfn, nrow, ncol, nband); // read header
  if(nband != 2) err("normalied index: 2-bands req'd");

  np = nrow * ncol;
  float * out = falloc(np); // alloc pixls
  float * dat = bread(fn, nrow, ncol, nband);

  float t, b, d;
  float * b1 = dat;
  float * b2 = &dat[np];
  for0(i, np){
    t = b1[i] - b2[i];
    b = b1[i] + b2[i]; d = t / b;
    out[i] = (isnan(d) || isinf(d))? NAN: d;
  }
  bwrite(out, ofn, nrow, ncol, 1);
  hwrite(ohn, nrow, ncol, 1);
  return 0;
}
