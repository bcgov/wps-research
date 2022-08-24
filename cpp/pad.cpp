/* 20220824 pad a raster extent (equal on all edges) */
#include"misc.h"

int main(int argc, char ** argv){
  size_t nrow, ncol, nband, np, nrow2, ncol2, nband2, i, j, k, ix, ij, ik, m;
  long int min_i, min_j, max_i, max_j;
  float * out, * dat, d;
  if(argc < 2){
    err("pad.exe [raster cube] # [optional arg: # of pixels] # pad a raster ");
  }
  long int N = 25; // default value, 50 pixels

  if(argc > 2) N = atol(argv[2]);

  str fn(argv[1]);
  str ofn(fn + str("_pad.bin")); // output file name
  
  if(exists(ofn)) err("output file exists");
  str ohn(hdr_fn(ofn, true)); // out header file name

  if(!exists(fn)) err("failed to open input file");
  str hfn(hdr_fn(fn));
  
  hread(hfn, nrow, ncol, nband); // read header

  run(str("gdal_translate -of ENVI -ot Float32 -srcwin ") +
      to_string(0 - N) + str(" ") +
      to_string(0 - N) + str(" ") +
      to_string(ncol + N + N) + str(" ") + 
      to_string(nrow + N + N) + str(" ") +
      fn + str(" ") +
      ofn); 
  run(str("fh ") + ohn);
  return 0;
}
