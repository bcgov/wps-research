/* 20220824 crop raster to content. I.e. keep good pixels
We define a bad pixel as a pixel with:
0. in every channel, or
NAN in every channel
*/
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 2) err("crop.exe [raster cube]");
  size_t nrow, ncol, nband, np, nrow2, ncol2, nband2, i, j, k, ix, ij, ik, m;
  float * out, * dat, d;

  str fn(argv[1]);
  str ofn(fn + str("_crop.bin")); // output file name
  if(exists(ofn)) err("output file exists");
  str ohn(hdr_fn(ofn, true)); // out header file name

  if(!exists(fn)) err("failed to open input file");
  str hfn(hdr_fn(fn));
  hread(hfn, nrow, ncol, nband); // read header

  np = nrow * ncol; // number of input pix
  out = falloc(np * nband);
  for0(i, np * nband) out[i] = 0.;
  dat = bread(fn, nrow, ncol, nband);

  int all_zero, all_nan;
  long int min_i, min_j, max_i, max_j;
  min_i = min_j = max_i = max_j = -1;

  for0(m, np){
    all_zero = all_nan = true;

    for0(k, nband){
      d = dat[m + (k * np)];
      if(d != 0.) all_zero = false;
      if(!(isnan(d) || isinf(d))) all_nan = false;
    }

    if(all_zero || all_nan){
      // bad pixel
    }
    else{
      i = m / ncol;
      j = m % ncol;
      if(min_i < 0){
        (min_i = i), (min_j = j);
      }
      else{
        if(i < min_i) min_i = i;
        if(j < min_j) min_j = j;
      }

      if(max_i < 0){
        (max_i = i), (max_j = j);
      }
      else{
        if(i > max_i) max_i = i;
        if(j > max_j) max_j = j;
      }

    }
  }

  cout << "min_i " << min_i << endl;
  cout << "min_j " << min_j << endl;
  cout << "max_i " << max_i << endl;
  cout << "max_j " << max_j << endl;

  return 0;
}
