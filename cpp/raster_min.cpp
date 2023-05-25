/* 20230524 raster_min.cpp: band math, minimum of all bands
adapted from raster_total.cpp
 */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 2) err("raster_total [raster cube] # add bands together for a raster\n");
  size_t nrow, ncol, nband, np;
  float * out, * dat;
  size_t i, j, k;

  str fn(argv[1]);
  str ofn(fn + str("_min.bin"));
  str ohn(hdr_fn(ofn, true)); // out header file name

  if(!exists(fn)) err("failed to open input file");
  str hfn(hdr_fn(fn));
  hread(hfn, nrow, ncol, nband); // read header

  np = nrow * ncol; // number of input pix
  out = falloc(np); // one band output
  for0(i, np) out[i] = 0.;  // set output to zero
    
  dat = bread(fn, nrow, ncol, nband);

	float d;
	float min = FLT_MAX;
  for0(i, np){
    for0(k, nband){
	    d = dat[(k * np) + i];
			if(d < min) min = d;
    }
		out[i] = min;
  }
  printf("+w %s\n", ofn.c_str());
  bwrite(out, ofn, nrow, ncol, 1);
  hwrite(ohn, nrow, ncol, 1);
  return 0;
}
