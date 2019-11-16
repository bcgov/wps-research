#include"misc.h"
int main(int argc, char** argv){
  //if(argc < 3) err("vri_polygonize [vri polygon raster file (ENVI type 4)] [vri poly attr. csv file]");

  size_t nrow, ncol, nband, i;
  str prf("vri/vri_poly.bin"); // polygon raster file
  str prf_hf(hdr_fn(prf));
  hread(prf_hf, nrow, ncol, nband);
  float * pr = bread(prf, nrow, ncol, nband);
  if(nband != 1) err("expected nband == 1");

  map<float, size_t> cf;
  size_t np = nrow * ncol;
  
  for0(i, np){
    float d = pr[i];
    if(cf.count(d) < 1) cf[d] = 0;
    cf[d] += 1;
  }

  map<size_t, size_t> cs;
  map<float, size_t>::iterator it;
  size_t min_i, max_i;
  min_i = max_i = cf.begin()->first;

  for(it = cf.begin(); it != cf.end(); it++){
    size_t first = (size_t) it->first;
    cs[first] = it->second; 
    if(first < min_i) min_i = first;
    if(first > max_i) max_i = first;
  }

  if(cf.size() != cs.size()) err("conversion from float failed");
  printf("max_i %ld min_i %ld n_i %ld\n", (long int)min_i, (long int)max_i, cf.size());

  str dbf("vri/VRI_KLoops.csv");

  str s, hdr;
  ifstream f(dbf);
  if(!f.is_open()) err("failed to open file");

  getline(f, hdr);
  vector<str> fields(split(hdr));
  cout << fields << endl;
  while(getline(f, s)){
	 vector<str> w(split_special(s));
         cout << fields.size() << "," << w.size() << endl;
	 if(w.size() != fields.size()){
	   cout << w << endl;
	   err("");
	 }
  }

  f.close();
  return 0;
}
