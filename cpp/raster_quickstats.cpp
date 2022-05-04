/* qs.cpp: quick stats for raster/ image cube. Update: 20220503 
Input: generic binary:
  (*) band-sequential, IEEE standard 32-bit float
  (*) Byte-order 0 (ESA SNAP uses Byte-order 1)
  (*) human-readable ENVI-format header file
Output: basic stats in CSV format
  (*) Min, Max, Mean, Stdv (excluding NaN / infinity)
  (*) NaN / infinity counts */
#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 2)
    err("qs.exe [raster cube 1]"); // usage

  str fn(argv[1]); // input image file name
  if(!exists(fn))
    err("failed to open input file");

  str hfn(hdr_fn(fn)); // input header file name
  size_t nrow, ncol, nband, np;
  hread(hfn, nrow, ncol, nband); // read header 1
  np = nrow * ncol; // number of input pix

  double d, diff;  // data values
  size_t i, j, k, ix, ij, ik;  // indices
  double * n = dalloc(nband);  // counts by band
  double * avg = dalloc(nband);  // mean
  double * fmax = dalloc(nband); // max
  double * fmin = dalloc(nband);  // min
  double * total = dalloc(nband);  // total
  double * stdev = dalloc(nband);  // stdev
  double * n_inf = dalloc(nband); // infinity count
  double * n_nan = dalloc(nband); // NaN count
  double * total_squared = dalloc(nband);  // for stdv
  float * dat = bread(fn, nrow, ncol, nband); // input

  for0(i, nband){
    fmin[i] = DBL_MAX;
    fmax[i] = -DBL_MAX;  // n.b.
    total[i] = n[i] = avg[i] = stdev[i] = 0.;
    total_squared[i] = n_inf[i] = n_nan[i] = 0.;
  }

  for0(i, np){
    for0(k, nband){
      d = dat[k * np + i];
      if(!(isinf(d) || isnan(d))){
        if(d < fmin[k])
	  fmin[k] = d;
        if(d > fmax[k])
	  fmax[k] = d;
        total[k] += d;
        n[k] ++;
      }
      else{
        if(isinf(d))
	  n_inf[k]++;
        if(isnan(d))
	  n_nan[k]++;
      }
    }
  }

  for0(k, nband)
    avg[k] = total[k] / n[k];

  for0(i, np){
    for0(k, nband){
      d = dat[np * k + i];
      if(!(isinf(d) || isnan(d))){
        diff = d - avg[k];
        total_squared[k] += diff * diff;
      }
    }
  }

  for0(k, nband)
    stdev[k] = sqrt(total_squared[k] / n[k]);

  printf("band_i,Min,Max,Mean,Stdv,n_nan,n_inf");
  for0(k, nband)
    printf("\n%zu,%e,%e,%e,%e,%e,%e",
	    k + 1,  // band ix
	    fmin[k], // min value this band
	    fmax[k], // max value this band
	    avg[k], // mean value this band
	    stdev[k], // standard deviation this band
	    n_nan[k], // nan count this band
	    n_inf[k]); // inf count this band
  printf("\n");
  return 0;
}
