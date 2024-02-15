/* 20230414 raster_isnan.cpp determine where a raster contains nan */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 2) err("raster_isnan [raster cube 1]");
  size_t nrow, ncol, nband, np, i, k, ix;
  float * out, * dat;

  str fn(argv[1]);
  str hfn(hdr_fn(fn));
  hread(hfn, nrow, ncol, nband);
  dat = bread(fn, nrow, ncol, nband);

  str ofn(in_file + "_isnan.bin");
  str ohn(hdr_fn(ofn, true)); // create output header file
  
  np = nrow * ncol; // number of input pix
  out = falloc(np);
  for0(i, np) out[i] = 0.;
	
  for0(k, nband){
    ix = np * k;
    for0(i, np){
      if(isnan(dat[ix + i])){
        out[i] = 1.;
      }
    }
  }

  printf("+w %s\n", ofn.c_str());
  bwrite(out, ofn, nrow, ncol, 1);
  hwrite(ohn, nrow, ncol, 1);
  return 0;
}
