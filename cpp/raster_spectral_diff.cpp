/* 20221128: raster_spectral_diff: non-centered first-order derivative approximation
 To be improved:
- weight by differences (between wavelengths)
- make centered */
#include"misc.h"
#include<math.h>

int main(int argc, char ** argv){
  if(argc < 2){
    err("raster_spectral_diff.cpp: apply centred difference formula\n");
  }

  str fn(argv[1]); // input image file name
  vector<string> s;
  if(!(exists(fn))) err("failed to open input file");
  str hfn(hdr_fn(fn)); // input header file name
  str fn2(fn + str("_spectral_diff.bin"));
  str hf2(fn + str("_spectral_diff.hdr"));

  size_t nrow, ncol, nband, np, i, j, k;
  hread(hfn, nrow, ncol, nband, s); // read header 1
  np = nrow * ncol; // number of input pix
  double* lambda = dalloc(nband);

  str datestr("");
  for0(i, s.size()){
    vector<string> w;
    w = split(s[i], ' ');
    str date_str(w[0]); // get date string if we can
    long long dsn = atoll(date_str.c_str()); // could check for number
    datestr = date_str;

    str last(w[w.size() -1]);
    cout << "last: " << last << endl;
    char m = last[strlen(last.c_str()) - 1];
    char n = last[strlen(last.c_str()) - 2];
    if(m != 'm' || n != 'n'){
      cout << "band name ending: " << m << n << endl;
      err("expected band name to end in nm");
    }
    str wl(last.substr(0, strlen(last.c_str()) - 2));
    lambda[i] = (double)atof(wl.c_str());
    cout << "lambda:" << lambda[i] << endl;
  }

  float d;
  float * dat = bread(fn, nrow, ncol, nband);
  float * out = falloc(nrow * ncol * (nband-1));
  bool is_zero, is_nan;

  for0(i, np){
    is_nan = false;
    is_zero = true; 
    for0(k, nband){
      d = dat[i + k * np]; // check if raster #1 is zero or NAN
      if(d != 0.) is_zero = false;
      if(isnan(d) || isinf(d)) is_nan = true;
    }
    if(is_nan || is_zero) for0(k, nband - 1) out[i + k * np] = NAN;
    else for0(k, nband -1) out[i + k * np] = dat[i + (k + 1) * np] - dat[i + k * np];
  }

  printf("+w %s\n", fn2.c_str());
  bwrite(out, fn2, nrow, ncol, nband - 1);
  hwrite(hf2, nrow, ncol, nband - 1);
  return 0;
}
