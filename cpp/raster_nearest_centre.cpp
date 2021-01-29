#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 3) err("raster_nearest_centre.cpp [input raster file] [input centres file BIP]");
  str cfn(argv[2]);
  float * centres = float_read(cfn);

  size_t nrow, ncol, nband;
  str bfn(argv[1]); // input "envi type-4" aka IEEE Floating-point 32bit BSQ (band sequential) data stack
  str hfn(hdr_fn(bfn)); // get name of header file
  hread(hfn, nrow, ncol, nband); // get image shape from header
  printf("nrow %d ncol %d nband %d\n", nrow, ncol, nband);
  float * dat = bread(bfn, nrow, ncol, nband); // read image data
 
  
  
  free(centres);
  free(dat);
  return 0;
}

