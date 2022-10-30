/* 20221029 raster_grand_total.cpp: add every float together for a raster */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 2) err("raster_grand_total [raster cube] # add every float together for a raster\n");
  size_t nrow, ncol, nband, np, nf;
  float * out, * dat, d;
  size_t i, j, k;

  str fn(argv[1]);
  if(!exists(fn)) err("failed to open input file");
  str hfn(hdr_fn(fn));
  hread(hfn, nrow, ncol, nband); // read header

  np = nrow * ncol; // number of input pix
  nf = np * nband;  

  double total = 0.;
  dat = bread(fn, nrow, ncol, nband);
  for0(i, nf){
	 d = (double)dat[i];
	 if(isnan(d) || isinf(d)){
	 }
	 else{
  	   total += fabs(d);
	 }
  }
  printf("grand total: %e\n", total);
  free(dat); 
  return 0;
}
