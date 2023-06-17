/* normalize a multispectral image or radar stack, square window, input
assumed ENVI type-4 32-bit IEEE standard floating-point format, BSQ
interleave

pixel-based normalization

20221027 do a version of this that mimics imv default setting */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 2) err("raster_normalize [input binary file name]");

  str fn(argv[1]); // input file name
  str hfn(hdr_fn(fn)); // auto-detect header file name
  size_t nrow, ncol, nband, np, i, j, k, n, ix, jx; // variables
  hread(hfn, nrow, ncol, nband); // read header
  np = nrow * ncol; // number of input pix
  size_t nf = np * nband;
  float * dat = bread(fn, nrow, ncol, nband); // load floats to array
  float mn, mx, d, r;

  for0(i, nrow){
    for0(j, ncol){
      // max norm? or L2 norm? how about subtract min, divide by max, this pixel??????
      mn = 0; //FLT_MAX;
      ix = i * ncol + j;

      for0(k, nband){
        d = dat[k * np + ix];
        if(!(isnan(d) || isinf(d))){
          mn += d * d;
        }
      }
      mn = sqrt(mn);

      for0(k, nband){
        jx = k * np + ix;
        d = dat[jx];
        if(!isnan(d) && !isinf(d)){
          dat[jx] /= mn;
        }
        else{
          dat[jx] = NAN;
        }
      }
    }
  }

  // write output file
  str ofn(fn + str("_norm_pixel.bin"));
  str ohfn(fn + str("_norm_pixel.hdr"));

  printf("nr %zu nc %zu nband %zu\n", nrow, ncol, nband);
  hwrite(ohfn, nrow, ncol, nband); // write output header

  FILE * f = fopen(ofn.c_str(), "wb");
  if(!f) err("failed to open output file");
  fwrite(dat, sizeof(float), nf, f); // write data
  fclose(f);
  free(dat);
  return 0;
}
