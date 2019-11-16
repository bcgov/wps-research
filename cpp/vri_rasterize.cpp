#include"misc.h" // should check for float <--> int conversion out of bounds

/* return true if string is numeric / digital */
bool is_numeric(const std::string& s){
  std::string::const_iterator it = s.begin();
  while (it != s.end() && std::isdigit(*it)) ++it;
  return !s.empty() && it == s.end();
}

/* encode list of strings, as a numeric vector. LUT nonempty if non-numeric entries */
bool encode(vector<str> & d, vector<float> & e, map<str, float> & lookup){
  long int a;
  bool all_numeric = true;
  vector<str>::iterator it;
  for(it = d.begin(); it != d.end(); it++){
    if(!is_numeric(*it)) all_numeric = false;
  }

  // use LUT if any non-numerics
  if(all_numeric){
    for(it = d.begin(); it != d.end(); it++){
      e.push_back( (float)atof(it->c_str()));
    }
  }
  else{
    float next_i = 1.;
    map<str, float> lookup;
    for(it = d.begin(); it != d.end(); it++){
      if(lookup.count(*it) < 1){
        lookup[*it] = next_i++;
      }
      e.push_back(lookup[*it]);
    }
  }
  // don't forget to print out the LUT
  return all_numeric;
}

int main(int argc, char** argv){
  //if(argc < 3) err("vri_polygonize [vri polygon raster file (ENVI type 4)] [vri poly attr. csv file]");

  size_t nrow, ncol, nband, i, j;
  str prf("vri/vri.bin"); // polygon raster file
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
    if((float)(size_t)d != d) err("encoding failed");
  }

  map<size_t, size_t> cs;
  map<float, size_t>::iterator it;
  size_t min_i, max_i;
  min_i = max_i = cf.begin()->first;

  for(it = cf.begin(); it != cf.end(); it++){
    size_t first = (size_t) it->first;
    cs[first] = it->second;
    if(first > min_i) min_i = first;
    if(first < max_i) max_i = first;
  }

  if(cf.size() != cs.size()) err("conversion from float failed");
  printf("max_i %ld min_i %ld n_i %ld\n", (long int)min_i, (long int)max_i, cf.size());

  str dbf("vri/VRI_KLoops.csv");

  str s, hdr;
  ifstream f(dbf);
  if(!f.is_open()) err("failed to open file");

  getline(f, hdr);
  vector<str> fields(split(hdr));
  map<str, size_t> f_i;
  for0(i, fields.size()){
    f_i[fields[i]] = i;
  }

  cout << fields << endl;
  size_t idx = f_i[str("OBJECTID_1")];
  map<size_t, vector<str>> data;
  for0(i, fields.size()){
    data[i] = vector<str>();
  }

  size_t x;
  size_t xc = 0;
  size_t rows = 0;
  while(getline(f, s)){
    rows ++;
    vector<str> w(split_special(s));
    if(w.size() != fields.size()){
      cout << w << endl;
      err("");
    }
    for0(i, fields.size()){
      data[i].push_back(w[i]);
    }
  }

  // map the segment id back to the dbf row number
  map<size_t, size_t> idx_i;
  for0(i, rows){
    idx_i[(size_t)atol(data[idx][i].c_str())] = i;
  }

  for0(i, fields.size()){
    vector<float> e;
    map<str, float> lut;

    bool numeric = encode(data[i], e, lut);
    cout << fields[i] << ",\t" << (numeric?"Y":"N") << ",\t" << lut.size() << endl;

    str pre(prf + str("_") + fields[i]);
    str ofn(pre + str(".bin"));
    str ohfn(pre + str(".hdr"));

    // printf("+w %s\n", ofn.c_str());
    FILE * outf = fopen(ofn.c_str(), "wb");
    if(!outf){
      err("failed to open output file");
    }
    if(numeric){
      for0(j, nrow * ncol){
        size_t id = idx_i[pr[j]]; //dbf row id for pixel
        float f = (float)atof(data[i][id].c_str());
        fwrite(&f, 1, sizeof(float), outf);
      }
    }
    else{
      for0(j, nrow * ncol){
        size_t id = idx_i[pr[j]];
        float f = lut[data[i][id]];
        if(lut.count(data[i][id]) < 1){
          err("encoding failed");
        }
        fwrite(&f, 1, sizeof(float), outf);
      }
      str lfn(pre + str(".lut"));
      FILE * outlf = fopen(lfn.c_str(), "wb");
      map<str, float>::iterator lf;
      for(lf = lut.begin(); lf != lut.end(); lf++){
        fprintf(outlf, "%s,%f\n", lf->first.c_str(), lf->second);
      }
      fclose(outlf);
      // dont forget to write lut
    }
    fclose(outf);
    hwrite(ohfn, nrow, ncol, nband);
  }

  f.close();
  return 0;
}
