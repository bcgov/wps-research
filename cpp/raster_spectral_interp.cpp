#include"misc.h"
/* Simple Spectral interpolation with wavelengths (nm) in header file 20220308 */
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
  str ofn(fn + "_active.bin");
  str hfn(hdr_fn(fn)); /* headers */
  str hf2(hdr_fn(ofn, true));

  vector<string> s;
  hread(hfn, nrow, ncol, nband, s);
  np = nrow * ncol;
  n = s.size();

  map<float, int> lower_i;
  map<float, int> upper_i;
  map<float, float> lower_last;
  map<float, float> upper_last;
  set<float>::iterator it;
  for0(i, n){
    vector<string> w;
    w = split(s[i], ' ');
    str last(w[w.size() -1]);
    char m = last[strlen(last.c_str()) - 1];
    char n = last[strlen(last.c_str()) - 2];
    if(m != 'm' || n != 'n') err("expected band name to end in nm");
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
  cout << i << endl;
  cout << lower_i << endl;
  cout << lower_last << endl;
  cout << upper_i << endl;
  cout << upper_last << endl;

  for(it = interp.begin(); it != interp.end(); it++){
  	cout << *it << ": " << endl;
  }
  /*
  float * out = falloc(np);
  float * dat = bread(fn, nrow, ncol, nband);

  float * b1, *b2, *b3;
  b1 = &dat[bi[0]]; b2 = &dat[bi[1]]; b3 = &dat[bi[2]];
  for0(i, np) out[i] = (float)(b2[i] - b1[i]) > 175.;
  cout << "second" << endl;
  for0(i, np){
    out[i] *= (float)(b3[i] > b2[i]);
  }
  bwrite(out, ofn, nrow, ncol, 1);
  hwrite(hf2, nrow, ncol, 1);
  free(dat);
  free(out);
  */
  return 0;
}
