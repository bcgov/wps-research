/* 20220824 crop raster to content. I.e. keep good pixels.
   Define a bad pixel as a pixel with:
	0. in every channel, or
	NAN in every channel */
#include"misc.h"

int main(int argc, char ** argv){
  size_t nrow, ncol, nband, np, nrow2, ncol2, nband2, i, j, k, ix, ij, ik, m;
  long int min_i, min_j, max_i, max_j;
  int all_zero, all_nan;
  float * out, * dat, d;
  if(argc < 2){
    err("crop.exe [raster cube] # crop hyperspec cube to content area");
  }

  str fn(argv[1]);
  str ofn(fn + str("_crop.bin")); // output file name
  
  if(exists(ofn)) err("output file exists");
  str ohn(hdr_fn(ofn, true)); // out header file name

  if(!exists(fn)) err("failed to open input file");
  str hfn(hdr_fn(fn));
  
  hread(hfn, nrow, ncol, nband); // read header
  np = nrow * ncol; // number of input pix
  out = falloc(np * nband); // allocate output buf
  for0(i, np * nband) out[i] = 0.;  // set output to 0.
  dat = bread(fn, nrow, ncol, nband);  // read input
  min_i = min_j = max_i = max_j = -1;  // uninit value

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
      i = m / ncol; // row
      j = m % ncol; // col 
      if(min_i < 0){
        (min_i = i), (min_j = j);  // init limits
      }
      else{
        if(i < min_i) min_i = i;  // revise limits
        if(j < min_j) min_j = j;
      }
      if(max_i < 0){
        (max_i = i), (max_j = j); // init limits
      }
      else{
        if(i > max_i) max_i = i;  // revise limits
        if(j > max_j) max_j = j;
      }
    }
  }

  size_t x = max_j - min_j + 2;
  size_t y = max_i - min_i + 2;
  cout << "x_size " << x << " min_i " << min_i << " max_i " << max_i << endl;
  cout << "y_size " << y << " min_j " << min_j << " max_j " << max_j << endl;

  run(str("gdal_translate -of ENVI -ot Float32 -srcwin ") +
      to_string(min_j) + str(" ") +
      to_string(min_i) + str(" ") +
      to_string(x) + str(" ") + 
      to_string(y) + str(" ") +
      fn + str(" ") +
      ofn); 
  run(str("fh ") + ohn);
  return 0;
}
