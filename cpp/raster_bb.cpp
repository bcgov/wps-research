/* 20221128: raster_bb.cpp: apply inverse planck formula to each band
Assumed units of Spectral Radiance (W/m2-sr-um)
https://ncc.nesdis.noaa.gov/data/planck.html */
#include"misc.h"
#include<math.h>

int main(int argc, char ** argv){
  if(argc < 2){
    err("raster_bb.cpp: apply inverse planck bb formula to each band\n");
  }

  double c1 = 1.191042e08; // W/m^2-sr-um
  double c2 = 1.4387752e04; // K um

  str fn(argv[1]); // input image file name
  vector<string> s;
  if(!(exists(fn))) err("failed to open input file");
  str hfn(hdr_fn(fn)); // input header file name
  str fn2(fn + str("_bb.bin"));
  str hf2(fn + str("_bb.hdr"));

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
    lambda[i] = (double)atof(wl.c_str()) * (double).001; // convert nm to um
    cout << "lambda:" << lambda[i] << endl;
  }

  float d;
  float * dat = bread(fn, nrow, ncol, nband);
  bool is_zero, is_nan;

  for0(i, np){
    for0(k, nband){
      d = dat[i + k * np]; // check if raster #1 is zero or NAN
      if(d == 0 || isnan(d) || isinf(d)) d = NAN;
      else{
        double h = 6.6260755*pow(10.,-34.);
        double c = 2.9979246*pow(10.,8.);
        double K = 1.380658*pow(10.,-23.);
        double RADIANCE = ((double)d); // * (1. / 10000.);
        double CWL = lambda[k] *0.000001;
        double T = h * c / (K * CWL) / log((2. * h * c * c) / (RADIANCE * 1000000. * pow(CWL, 5.)) + 1.);
	d = (float) T;
        //d = (float)( c2 / ((double)lambda[k] * log(1. + c1 / ( pow((double)lambda[k], 5.) * d))));
      }
      dat[i + k * np] = d;
    }
  }

  printf("+w %s\n", fn2.c_str());
  bwrite(dat, fn2, nrow, ncol, nband);
  run(str("cp -v " ) + hfn + str(" ") + hf2);
  return 0;
}
