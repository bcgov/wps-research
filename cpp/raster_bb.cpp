/* 20221128: raster_bb.cpp: apply inverse planck formula to each band
Assumed units of Spectral Radiance (W/m2-sr-um)
https://ncc.nesdis.noaa.gov/data/planck.html

Note:
L1C data is in TOA reflectance. Need to apply (sentinel2 tbx) reflectance to radiance operator:
https://github.com/senbox-org/s2tbx/blob/master/s2tbx-reflectance-to-radiance-ui/src/main/resources/org/esa/s2tbx/reflectance2radiance/docs/ReflectanceToRadianceAlgorithmSpecification.html


Reflectance to Radiance Algorithm Specification
Reflectance to Radiance

Reflectance is the proportion of the radiation striking a surface to the radiation reflected off of it.
Radiance is the amount of radiation coming from an area.
Top-of-atmosphere reflectance (or TOA reflectance) is the reflectance measured by a space-based sensor flying higher than the earth's atmosphere.
These reflectance values will include contributions from clouds and atmospheric aerosols and gases.

The Reflectance to Radiance results from the following formula:

radiance = pixelValue * cosinus(radians(incidenceAngle)) * solarIrradiance * scale / (pi * d2)

, where:
d2 = 1.0 / U (the sun-earth distance)
solarIrradiance = the mean solar exoatmospheric irradiances for each band
scale = 1 / (0.001 * 1000) = 1 (default)

For Sentinel-2 the incidence angle is replaced with the values from the sun_zenith band.*/
#include"misc.h"
#include<math.h>

int main(int argc, char ** argv){
  printf("raster_bb\n");
  if(argc < 2){
    err("raster_bb.cpp: apply inverse planck bb formula to each band\n");
  }

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
    lambda[i] = (double)atof(wl.c_str()); 
    cout << "lambda:" << lambda[i] << endl;
  }

  float d;
  float * dat = bread(fn, nrow, ncol, nband);
  bool is_zero, is_nan;

  /*
  function IPLANCKWL(){
    h= 6.6260755*Math.pow(10,-34)
    c= 2.9979246*Math.pow(10,8)
    k=1.380658*Math.pow(10,-23)
    form = document.formIWL
    RADIANCE= form.RADIANCE.value
    CWL= form.CWL.value*0.000001

    T=h*c/(k*CWL)/Math.log((2*h*c*c)/(RADIANCE*1000000*Math.pow(CWL,5)) + 1)
    form.TEMPERATURE.value=T
  }
  */
  double T;
  double RADIANCE;
  double h = 6.6260755 * pow(10.,-34.);
  double c = 2.9979246 * pow(10.,8.);
  double K = 1.380658 * pow(10.,-23.);
  for0(k, nband){
    double wavelength_nm = (double) lambda[k];  // our wavelength is in nanometres
    double wavelength_um = 0.001 * wavelength_nm; // the form expected wavelength in micrometre(um)
    double Lambda = wavelength_um * 0.000001;
    for0(i, np){
      d = dat[i + k * np]; // get the value of pixl "i" for band "k"
      if(d == 0. || isnan(d) || isinf(d)) T = NAN; // check if value is zero or NAN (no-data)
      else{
        RADIANCE = ((double)d) / 10000.;
        T = ((h * c) / (K * Lambda))  / log((2. * h * pow(c, 2.) * pow(Lambda, -5.) / (RADIANCE * 1000000.)) + 1.);
        //T -= 273.15;
      }
      if(i %1000 == 0) printf("cwl %f d %f RAD %e T %f\n", lambda[k],d, RADIANCE, (float)T);
      dat[i + k * np] = (float)T;
    }
  }

  printf("+w %s\n", fn2.c_str());
  bwrite(dat, fn2, nrow, ncol, nband);
  run(str("cp -v " ) + hfn + str(" ") + hf2);
  return 0;
}
