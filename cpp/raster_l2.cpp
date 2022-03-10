/* 20220309: L2 norm for raster. Output: 1-band raster */
#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 2){
    err("raster_l2.exe [hyperspec cube] # scale each band into 0-1\n");
  }
  str fn(argv[1]); // input image file name
  if(!(exists(fn))) err("failed to open input file");
  str hfn(hdr_fn(fn)); // input header file name

  str ofn(str(argv[1]) + str("_norm.bin")); // output file name
  str ohn(hdr_fn(ofn, true)); // out header file name
  printf("+w %s\n", ofn.c_str());
  printf("+w %s\n", ohn.c_str());

  size_t nrow, ncol, nband, np, i, j, k;
  hread(hfn, nrow, ncol, nband); // read header 1
  np = nrow * ncol; // number of input pix

  float * dat = bread(fn, nrow, ncol, nband);
  float * out = falloc(np);

  float d;
  for0(i, np){
    out[i] = 0.;
    for0(k, nband){
      d = dat[i + np * k];
      out[i] += d * d;
    }
    out[i] = (float)sqrt((double)out[i]);
  }
  bwrite(out, ofn, nrow, ncol, 1);
  hwrite(ohn, nrow, ncol, 1);
  return 0;
}