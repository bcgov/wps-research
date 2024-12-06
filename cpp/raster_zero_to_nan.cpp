/* replace all-zero pixels in raster, with NAN*/
#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 2) err("raster_zero_to_nan.exe [input binary file]\n");
 
  str fn(argv[1]); // input file name
  str hfn(hdr_fn(fn)); // auto-detect header file name
  size_t nrow, ncol, nband, np, i, j, k;
  cout << "+r " << hfn << endl;
  hread(hfn, nrow, ncol, nband); // read header
  np = nrow * ncol; // number of input pix
  cout << "+r " << fn << endl;
  
  float * dat = bread(fn, nrow, ncol, nband); // load floats to array
  size_t ix;
  float d;
  for0(i, nrow){
    for0(j, ncol){
      ix = i * ncol + j;
      bool all_zero = true;
      for0(k, nband){
        d = dat[np * k + ix];
        // 20241205 need to consider mixed 0 and nan
        if(!( d == 0. || isnan(d))){
	        all_zero = false;
	      }
      }
      if(all_zero){
    	  for0(k, nband){
          dat[np * k + ix] = NAN;
      	}
      }
    }
  }

  cout << "+w " << fn << endl;
  FILE * f = fopen(fn.c_str(), "wb");
  if(!f) err("failed to open output file");
  fwrite(dat, sizeof(float) * nrow * ncol * nband, 1, f); // write data
  fclose(f);
  free(dat);
  return 0;
}
