/* 20220309 raster_burn_zeroes.cpp: burn the zeros from one image onto another,
in-place: use to remove data on image #2, where image #1 has nodata */
#include"misc.h"
#include<math.h>

int main(int argc, char ** argv){
  if(argc < 3){
    printf("raster_burn.cpp: burn non-data (NAN) onto image #2, where image #1 has nodata (NAN, or (all-zero and nband > 1))\n");
    err("raster_burn_zeros.exe [first raster to burn zeros from] [onto this raster (in place)]\n");
  }
  str fn(argv[1]); // input image file name
  str fn2(argv[2]); // input image 2 file name
  if(!(exists(fn) && exists(fn2))) err("failed to open input file");

  str hfn(hdr_fn(fn)); // input header file name
  str hfn2(hdr_fn(fn2)); // input 2 header file name

  size_t nrow, ncol, nband, np, nrow2, ncol2, nband2;
  hread(hfn, nrow, ncol, nband); // read header 1
  hread(hfn2, nrow2, ncol2, nband2); // read header 2
  if(nrow != nrow2 || ncol != ncol2)
    err("input image dimensions should match");
 
  np = nrow * ncol; // number of input pix
  float d;
  size_t i, j, k;
  bool is_zero, is_nan;
  cout << "+r " << fn << endl;
  float * dat1 = bread(fn, nrow, ncol, nband);
  int * bad = ialloc(nrow * ncol);

  for0(i, np){
    is_zero = true;
    is_nan = false;
    for0(k, nband){
      d = dat1[i + k * np];  // check if raster #1 is zero or NAN
      if(d != 0) is_zero = false;
      if(isnan(d) || isinf(d)) is_nan = true;

      if(is_nan || (is_zero && nband > 1)){
        bad[i] = true;
      }
      else{
        bad[i] = false;
      }
    }
  }
  free(dat1);
  cout << "+r " << fn2 << endl;
  float * dat2 = bread(fn2, nrow, ncol, nband2);

  for0(i, np){
    if(bad[i]){
      for0(k, nband2) dat2[i + k * np] = NAN; // set raster #2 to NAN
    }
  }
  printf("+w %s\n", fn2.c_str());
  bwrite(dat2, fn2, nrow, ncol, nband2);
  return 0;
}
