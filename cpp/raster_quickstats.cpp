/* qs.cpp: quick stats for raster/ image cube. Update: 20220504 
Input: generic binary:
  (*) band-sequential, IEEE standard 32-bit float
  (*) Byte-order 0 (ESA SNAP uses Byte-order 1)
  (*) human-readable ENVI-format header file
Output: basic stats in CSV format
  (*) Min, Max, Mean, Stdv (excluding NaN / infinity)
  (*) NaN / infinity counts (and percentages)
Note: uses lossless scientific notation */
#include"misc.h"
void printd(double d){
  char s[DBL_DIG + 8];
  sprintf(s, "%.*E", DBL_DIG, d);
  vector<str> w(split(str(s), 'E'));
  rtrim(w[0], str("0")); // trim  0's
  if(w[0].back() == '.') w[0] += str("0");
  printf(",%sE%s", w[0].c_str(), w[1].c_str());
}

int main(int argc, char ** argv){
  if(argc < 2)
    err("qs.exe [raster cube 1]"); // usage

  str fn(argv[1]); // input image file name
  if(!exists(fn))
    err("failed to open input file");

  str hfn(hdr_fn(fn)); // input header filename
  size_t nrow, ncol, nband, np;

  vector<str> s;
  hread(hfn, nrow, ncol, nband, s); // read header
  np = nrow * ncol; // input pixl count

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
      d = dat[k * np + i];  // band-sequential
      if(!(isinf(d) || isnan(d))){
        if(d < fmin[k])  // find min
	  fmin[k] = d;
        if(d > fmax[k])  // find max
	  fmax[k] = d;
        total[k] += d;  // sum
        n[k] ++;
      }
      else{
        if(isinf(d))  // infinity count
	  n_inf[k]++;
        if(isnan(d))  // nan count
	  n_nan[k]++;
      }
    }
  }

  for0(k, nband)  // mean
    avg[k] = total[k] / n[k];

  for0(i, np){
    for0(k, nband){
      d = dat[np * k + i];
      if(!(isinf(d) || isnan(d))){
        diff = d - avg[k];  // for stdv
        total_squared[k] += diff * diff;
      }
    }
  }

  for0(k, nband)  // stdv
    stdev[k] = sqrt(total_squared[k] / n[k]);

  printf("band_i,Min,Max,Mean,Stdv,n_nan,n_inf,%%nan,%%inf,band_name");  // csv header
  for0(k, nband){
    printf("\n%zu", k + 1);
    printd(fmin[k] !=  DBL_MAX? fmin[k]: NAN); // min value this band
    printd(fmax[k] != -DBL_MAX? fmax[k]: NAN); // max value this band
    printd(avg[k]); // mean value this band
    printd(stdev[k]); // standard deviation this band
    printd(n_nan[k]); // nan count this band
    printd(n_inf[k]); // inf count this band
    printd(100. * n_nan[k] / (double)np); // nan percent of total this band
    printd(100. * n_inf[k] / (double)np); // inf percent of total this band
    printf(",%s", s[k].c_str());
  }
  printf("\n");
  return 0;
}
