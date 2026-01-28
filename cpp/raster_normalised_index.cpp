/* raster_normalised_index.cpp: raster normalized index (input >=2 band float type) 20240204 */
#include "misc.h"
#include <vector>
#include <string>

int main(int argc, char ** argv){
  if(argc < 2) err("raster_ni.exe [raster cube, 2 bands]\n");

  str fn(argv[1]); // input image file
  if(!(exists(fn))) err("failed to open input");

  str hfn(hdr_fn(fn));                 // input header file
  str ofn(fn + str("_normalised_index.bin")); // output file
  str ohn(hdr_fn(ofn, true));          // output header file

  size_t nrow, ncol, nband, np, i, j, k;
  hread(hfn, nrow, ncol, nband);       // read header
  if(nband < 2) err("normalized index: 2-bands req'd");

  np = nrow * ncol;                    // number of pixels

  size_t n_out = (nband * nband - nband); // number of output bands

  float * out = falloc(np * n_out);    // output pixels
  float * dat = bread(fn, nrow, ncol, nband);

  std::vector<std::string> bandNames;
  bandNames.reserve(n_out);

  size_t out_i = 0;

  for0(j, nband){
    for0(k, nband){
      if(j == k) continue;

      // Band name reflects the math
      bandNames.push_back(
        "B" + to_string(j+1) + "_minus_B" + to_string(k+1) +
        "_over_" +
        "B" + to_string(j+1) + "_plus_B" + to_string(k+1)
      );

      float * b1 = &dat[np * j];
      float * b2 = &dat[np * k];

      for0(i, np){
        float t = b1[i] - b2[i];
        float b = b1[i] + b2[i];
        float d = t / b;
        out[out_i * np + i] = (isnan(d) || isinf(d)) ? NAN : d;
      }

      out_i++;
    }
  }

  bwrite(out, ofn, nrow, ncol, n_out);

  // Write header with band names
  hwrite(ohn, nrow, ncol, n_out, H_FLOAT32, bandNames);

  return 0;
}

