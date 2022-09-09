#include"misc.h"
/* Simple Spectral interpolation with wavelengths (nm) in header file 20220308
 
20220324: need to add extrapolation case for data outside? */
int main(int argc, char ** argv){
  if(argc < 3) err("raster_spectral_interp [input file name] [wavelength value to interpolate] .. [more values]");
  size_t nrow, ncol, nband, np, i, j, k, n, ij;
  set<float> interp;
  map<float, str> interps;
  for(i = 2; i < argc; i++){
    interp.insert(atof(argv[i]));
    interps[atof(argv[i])] = str(argv[i]);
  }
  cout << "wavelengths (nm) to interpolate at:" << interp << endl;
  str fn(argv[1]); /* binary files */
  str ofn(fn + "_spectral_interp.bin");
  str hfn(hdr_fn(fn)); /* headers */
  str hf2(hdr_fn(ofn, true));

  vector<string> s;
  hread(hfn, nrow, ncol, nband, s);
  np = nrow * ncol;
  n = s.size();

  map<float, int> lower_i;
  map<float, int> upper_i;
  set<float>::iterator it;
  map<float, float> lower_last;
  map<float, float> upper_last;

  str datestr("");
  for0(i, n){
    vector<string> w;
    w = split(s[i], ' ');
    str date_str(w[0]); // get date string if we can
    long long dsn = atoll(date_str.c_str()); // could check for number
    datestr = date_str;

    str last(w[w.size() -1]);
    char m = last[strlen(last.c_str()) - 1];
    char n = last[strlen(last.c_str()) - 2];
    if(m != 'm' || n != 'n'){
      cout << "band name ending: " << m << n << endl;
      err("expected band name to end in nm");
    }
    str wl(last.substr(0, strlen(last.c_str()) - 2));
    float wavelength = atof(wl.c_str());
    for(it = interp.begin(); it != interp.end(); it++){
      if(lower_i.count(*it) < 1){
        // haven't compared anything with this interpolation value
        if(wavelength <= *it){
          lower_i[*it] = i;
          lower_last[*it] = wavelength;
        }
      }
      else{
        if(wavelength >= lower_last[*it] && wavelength <= *it){
          lower_last[*it] = wavelength;
          lower_i[*it] = i;
        }
      }
      if(upper_i.count(*it) < 1){
        if(wavelength >= *it){
          upper_i[*it] = i;
          upper_last[*it] = wavelength;
        }
      }
      else{
        if(wavelength <= upper_last[*it] && wavelength >= *it){
          upper_last[*it] = wavelength;
          upper_i[*it] = i;
        }
      }
    }
  }
  /* cout << i << endl;
  cout << lower_i << endl;
  cout << lower_last << endl;
  cout << upper_i << endl;
  cout << upper_last << endl;
  */

  float * out = falloc(np);
  float * dat = bread(fn, nrow, ncol, nband);
  FILE * f = wopen(ofn);
  vector<str> band_names;
  for(it = interp.begin(); it != interp.end(); it++){
    float y = *it;
    cout << y << ": f(" << lower_last[y] << ") + (" << y << " - " << lower_last[y] << ") * (f(" << upper_last[y] << ") - f(" << lower_last[y] << ") / (" << upper_last[y] << " - " << lower_last[y] << "))" << endl;

    cout << "\t=" << " f(" << lower_last[y] << ") + " << (y - lower_last[y]) << " * (f(" << upper_last[y] << ") - f(" << lower_last[y] << ") / " << (upper_last[y] - lower_last[y]) << ")" << endl;
    float * dx = &dat[np * lower_i[y]];
    float * dz = &dat[np * upper_i[y]];
    for0(i, np){
      out[i] = dx[i] + (y - lower_last[y]) * (dz[i] - dx[i]) / (upper_last[y] - lower_last[y]);
    }
    size_t nw = fwrite(out, np, sizeof(float), f);
    band_names.push_back(datestr + str(" Interpolated value at: ") + std::to_string((int)y) + str("nm"));
  }
  hwrite(hf2, nrow, ncol, interp.size(), 4, band_names);
  free(dat);
  free(out);
  return 0;
}
