/* raster_at.cpp: extract spectra at row, col index */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 4) err("raster_at.exe [raster cube 1] [ x/ row idx] [ y/ col idx]");

  str fn(argv[1]); // input image file name
  if(!exists(fn)) err("failed to open input file");

  int x = atoi(argv[2]);
  int y = atoi(argv[3]);

  str hfn(hdr_fn(fn)); // input header file name

  size_t nrow, ncol, nband, np;
  hread(hfn, nrow, ncol, nband); // read header 1
  np = nrow * ncol; // number of input pix
  
  double d, diff;
  size_t i, j, k, ix, ij, ik;
  float * dat = bread(fn, nrow, ncol, nband);
  
  printf("band_i,value");
  for0(k, nband){
    printf("\n%d,%e", k, dat[np * k + x * ncol + y]);
  }
  printf("\n");
  return 0;
}
