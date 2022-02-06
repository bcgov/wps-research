/* generalizing an idea found in dr. Dey's paper, to more dimensions..

Second level of complexity 20220205 */
#include"misc.h"

// class to sort the values' coordinate indices..

// entropy.. ?

int main(int argc, char ** argv){
  if(argc < 2) err("raster_inequality2 [input binary file name]");

  str fn(argv[1]); // input file name
  str hfn(hdr_fn(fn)); // auto-detect header file name
  size_t nrow, ncol, nband, np, i, j, k, n, ip, jp, ix1, ix2; // variables
  hread(hfn, nrow, ncol, nband); // read header
  np = nrow * ncol; // number of input pix

  if(nband < 2) err("need at least 2 bands..");

  float d, e;
  float * dat = bread(fn, nrow, ncol, nband); // load floats to array

  size_t n_class = nband * nband;
  float * out = falloc(np * n_class); // output channel
  for0(i, np * n_class) out[i] = 0.;

  for0(i, np){
    if(i % 100000 == 0.) printf("%zu of %zu\n", i, np);

    for0(k, nband){
      d = dat[k * np + i];
      if(isnan(d) || isinf(d)){
      }
      else{
        for0(j, nband){
	  size_t class_i = k * nband + j;
	  e = dat[k * np + j];
	  if(isnan(e) || isinf(e)){
	  }
	  else{
	    if(d > e){
		out[np * class_i + i] = 1.;
	    }
	  }
	}
      }
    }
  }

  str ofn(fn + str("_inequality.bin"));
  str ohfn(fn + str("_inequality.hdr"));
  hwrite(ohfn, nrow, ncol, n_class);
  FILE * f = fopen(ofn.c_str(), "wb");
  if(!f) err("failed to open output file");
  fwrite(out, sizeof(float), np * n_class, f); // write data
  fclose(f);
  free(dat);
  return 0;
}
