/* qs.cpp: quick stats for image cube */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 2) err("qs.exe [raster cube 1]");

  str fn(argv[1]); // input image file name
  if(!exists(fn)) err("failed to open input file");

  str hfn(hdr_fn(fn)); // input header file name

  size_t nrow, ncol, nband, np;
  hread(hfn, nrow, ncol, nband); // read header 1
  np = nrow * ncol; // number of input pix
  
  double d, diff;
  size_t i, j, k, ix, ij, ik;
  float * dat = bread(fn, nrow, ncol, nband);

  double * fmax = dalloc(nband);
  double * fmin = dalloc(nband);
  double * total = dalloc(nband);
  double * avg = dalloc(nband);
  double * n = dalloc(nband); 
  double * total_squared = dalloc(nband);
  double * stdev = dalloc(nband);
 
  double * n_inf = dalloc(nband); // count infinity
  double * n_nan = dalloc(nband); // count nan

  for0(i, nband){
    fmax[i] = DBL_MIN;
    fmin[i] = DBL_MAX;
    total[i] = n[i] = avg[i] = total_squared[i] = stdev[i] = 0.;
    n_inf[i] = n_nan[i] = 0.;
  }

  for0(i, nrow){
    ix = i * ncol;
    
    for0(j, ncol){
      ij = ix + j;
      for0(k, nband){
        ik = ij + k * np;
        d = dat[ik];

        if(!(isinf(d) || isnan(d))){
          if(d < fmin[k]) fmin[k] = d;
          if(d > fmax[k]) fmax[k] = d;
          total[k] += d;
          n[k] ++;
        }
	else{
	  if(isinf(d)) n_inf++;
	  if(isnan(d)) n_nan++;
	}
      }
    }
  }
  
  for0(k, nband) avg[k] = total[k] / n[k];

  for0(i, nrow){
    ix = i * ncol;
    
    for0(j, ncol){
      ij = ix + j;
      
      for0(k, nband){
        ik = ij + k * np;
        d = dat[ik];

        if(!(isinf(d) || isnan(d))){
          diff = d - avg[k];
          total_squared[k] += diff * diff;
        }
      }
    }
  }

  for0(k, nband) stdev[k] = sqrt(total_squared[k] / n[k]);
  
  printf("band_i,Min,Max,Mean,Stdv,n_nan,n_inf");
  for0(k, nband) printf("\n%d,%e,%e,%e,%e", k, fmin[k], fmax[k], avg[k], stdev[k], n_nan[k], n_inf[k]);
  printf("\n");
  return 0;
}
