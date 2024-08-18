/* csv_spectra_raster_distance.cpp:
calculate the mean and stdv spectra for the (already labelled) class indicated
regular, derivative or integral mode..

we are assuming that the
*/
#include"misc.h"
#include<vector>
int debug = false;

void deriv(float * x, int x_len, float * y){
  int i;
  for0(i, (x_len - 1))
  y[i] = x[i + 1] - x[i];
}

void integ(float * x, int x_len, float * y){
  int i;
  y[0] = x[0];
  for0(i, (x_len - 1))
  y[i + 1] = y[i] + x[i + 1];
}

int main(int argc, char ** argv){
  if(argc < 5) err(str("csv_spectra_raster_distance ") +
  str("[csv file] ") + // csv file to get categorized spectra from
  str("[col-label of field of interest] ") + // label of column containing categorical field
  str("[observation of field of interest to match]") + // value of the categorical field, over which to average spectra
  str("[raster image to project onto]"));

  vector<str> cases;
  cases.push_back(str("regular"));
  /* cases.push_back(str("derivative"));
  cases.push_back(str("integral"));
  */

  str csv_fn(argv[1]); // input image file name
  str fn(argv[4]); // input image 2 file name
  if(!(exists(csv_fn) && exists(fn))) err("failed to open all input files");
  str hfn(hdr_fn(fn)); // input raster header file name

  size_t nrow, ncol, nband, np, i, j, k, ix, ij, ik;
  hread(hfn, nrow, ncol, nband); // read header 1
  np = nrow * ncol; // number of input pix

  vector<str> fields;
  vector<int> spec_fi;
  vector<vector<str>> lines(read_csv(csv_fn, fields));
  int fi = -1; // locate the relevant categorical col!
  for0(i, fields.size()){
    if(fields[i] == str(argv[2])) fi = i;
    const char * s = fields[i].c_str();
    int s_len = strlen(s);
    if(s[s_len-2] == 'n' && s[s_len-1] == 'm')
    spec_fi.push_back(i); // gather spectral cols too.. ending nm!
  }

  printf("fi %d: %s\n", fi, fields[fi].c_str());
  cout << "spec_fi" << spec_fi << endl;
  int M = spec_fi.size();

  if(nband != M){
    cout << "nband " << nband << endl;
    cout << "M " << M << endl;
	  err("unexpected number of bands in raster file");
  }
  float * spec = falloc(M); // buffer for a spectrum
  float * trans = falloc(M); // transformed spec
  float * ts; // pointer to transformed spec (or not)
  float * mean = falloc(M);
  for0(i, M) mean[i] = 0.;
  float count = 0;

  float * out = falloc(nrow * ncol); // output distance
  vector<str>::iterator ii;
  vector<vector<str>>::iterator it;
  float * dat = bread(fn, nrow, ncol, nband);

  for(ii=cases.begin(); ii != cases.end(); ii++){
    for0(i, M) mean[i] = 0.;
    count = 0;

    for(it = lines.begin(); it != lines.end(); it++){
      if((*it)[fi] == str(argv[3])){
        for0(i, M){
          spec[i] = (float)strtod((*it)[spec_fi[i]].c_str(), NULL);
        }

        int tM = M;
        ts = trans;
        if((*ii) == str("derivative")){
          tM = M - 1;
          deriv(spec, M, trans);
        }
        else if((*ii) == str("integral")){
          integ(spec, M, trans);
        }
        else{
          ts = spec; // use the untransformed spectra
        }

        for0(i, M){
          mean[i] += ts[i];
        }
        count += 1.;
      }
    }
    for0(i, M) mean[i] /= count;

    printf("mean (%s):\n[", ii->c_str());
    for0(i, M)
    printf("%s%f", (i > 0 ? ",": ""), mean[i]);
    printf("]\n");
    str ofn(fn + str("_") + str(fields[fi]) + str("_eq_") + str(argv[3]) +  str("_") + (*ii) + str("_class_dist.bin"));

    cout << "+w " << ofn << endl;

    for0(i, nrow){
      ix = i * ncol;
      for0(j, ncol){
        ij = ix + j;
        for0(k, nband){
          spec[k] = dat[(np * k) + ij];
        }

        int tM = M;
        ts = trans;
        if((*ii) == str("derivative")){
          tM = M - 1;
          deriv(spec, M, trans);
        }
        else if((*ii) == str("integral")){
          integ(spec, M, trans);
        }
        else{
          ts = spec; // use the untransformed spectra
        }

        // calcuate distance here
        float d = 0.;
        for0(k, tM){
          float x = (spec[k] - mean[k]); // CHANGE BACK TO TS[K]
	  if(debug) printf("x %f\n", x);
          d += sqrt(x * x);
        }
        out[ij] = d;
	if(debug){
	  printf("d %f\n", d); 
	  exit(1);
	}
      }
    }

    str ohn(hdr_fn(ofn, true)); // out header file name
    hwrite(ohn, nrow, ncol, 1); // expand this to per-class right away!
    bwrite(out, ofn, nrow, ncol, 1); // nband : this produces a very trippy result);
  }
  return 0;
}
