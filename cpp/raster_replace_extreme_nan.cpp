/* 20230320 replace pixels with extreme magnitude, with NAN */
#include"misc.h"

inline float f_abs(float f){
  return (f < 0) ? -f : f;
}

int main(int argc, char ** argv){
  if(argc < 2) err("raster_replace_extreme_nan.exe [input binary file]\n");
 
  str fn(argv[1]); // input file name
  str hfn(hdr_fn(fn)); // auto-detect header file name
  size_t nrow, ncol, nband, np, i, j, k, nf;
  cout << "+r " << hfn << endl;
  hread(hfn, nrow, ncol, nband); // read header
  np = nrow * ncol; // number of input pix
  nf = np * nband;
  cout << "+r " << fn << endl;
  
  float * dat = bread(fn, nrow, ncol, nband); // load floats to array
  size_t ix;
  float d = 0.;
  for0(i, nf) d = max(d, f_abs(dat[i]));
  for0(i, nf) dat[i] = (dat[i] == d) ? NAN: dat[i]; 

  cout << "+w " << fn << endl;
  FILE * f = fopen(fn.c_str(), "wb");
  if(!f) err("failed to open output file");
  fwrite(dat, sizeof(float) * nrow * ncol * nband, 1, f); // write data
  fclose(f);
  free(dat);
  return 0;
}
