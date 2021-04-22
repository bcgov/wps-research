#include"misc.h"

int main(int argc, char ** argv){
  
  if(argc < 2) err("hsv2rgb.exe [input raster file name, 3 band]");
  
  str fn(argv[1]); // input image file name
  if(!exists(fn)) err(str("failed to open input file: ") + fn);
  str hfn(hdr_fn(fn)); // input header file name

  size_t nrow, ncol, nband, np;
  hread(hfn, nrow, ncol, nband); // read header
  np = nrow * ncol; // number of input pix
  if(nband != 3) err("3 band input supported"); // need rgb

  size_t i, j, k;
  float * dat = bread(fn, nrow, ncol, nband); // load floats to array
  float * out = falloc(nrow * ncol * nband);

  float h, s, v, r, g, b;
  for0(i, nrow){
    for0(j, ncol){
      k = i * ncol + j;
      h = dat[k];
      s = dat[k + np];
      v = dat[k + np + np];

      hsv_to_rgb(&r, &g, &b, h, s, v);
      out[k] = r;
      out[k + np] = g;
      out[k + np + np] = b;
    }
  }


  str ofn(fn + str("_rgb.bin"));
  str ohn(fn + str("_rgb.hdr"));

  bwrite(out, ofn, nrow, ncol, nband);
  hwrite(ohn, nrow, ncol, nband);
  return 0;
}
