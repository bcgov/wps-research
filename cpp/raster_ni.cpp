/* raster_ni: raster normalized index (input 2 band float type) 20220308 */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 2) err("raster_ni.exe [raster cube, 2 bands\n");

  str fn(argv[1]); // input image file name
  if(!(exists(fn))) err("failed to open input file");

  str hfn(hdr_fn(fn)); // input header file name
  str ofn(fn + str("_ni.bin")); // output file name
  str ohn(hdr_fn(ofn, true)); // out header file name

  size_t nrow, ncol, nband, np, i;
  hread(hfn, nrow, ncol, nband); // read header 1
  if(nband != 2) err("two band input required for normalized index");

  np = nrow * ncol; // number of input pix
  float * out = falloc(np);
  float * dat = bread(fn, nrow, ncol, nband);
  float * b1 = dat;
  float * b2 = &dat[np];

  for0(i, np){
	  float top = b1[i] - b2[i];
	  float bot = b1[i] + b2[i];
	  float d = top / bot;
	  if(isnan(d) || isinf(d)) d = NAN;
	  out[i] = d;
  }
  bwrite(out, ofn, nrow, ncol, 1);
  hwrite(ohn, nrow, ncol, 1);
  return 0;
}
