/* Band unstacking. Should revise this with band names now
20220308 revised to accept specific bands only */
#include"misc.h"

int main(int argc, char *argv[]){
  if(argc < 2) err("unstack [input data file] [optional arg: band to extract (from 1)] ... [optional arg: band to extract (from 1)]");
  size_t i, j, nr, nc, nb, np;
  
  set<int> selected;
  for(i = 2; i < argc; i++) selected.insert(atoi(argv[i]));
  
  FILE * f;
  str ifn(argv[1]);
  str hfn(hdr_fn(ifn));
  hread(hfn, nr, nc, nb);
  vector<str> band_names(parse_band_names(hfn));

  np = nr * nc;
  for(set<int>::iterator it = selected.begin(); it != selected.end(); it ++){
    if(*it < 1 || *it > nb){
      err("selected band out of bounds");
    }
  }

  float * d = bread(ifn, nr, nc, nb); // read input data
  
  for0(i, nb){
    if(selected.size() < 1 || selected.count((int)(i + 1)) > 0){
      str pre(ifn + str("_") + zero_pad(to_string(i + 1), 3));
      pre += band_names[i];
      str ofn(pre + str(".bin"));
      str ohn(pre + str(".hdr"));
      f = wopen(ofn.c_str());
      fwrite(&d[np * i], sizeof(float), np, f);
      fclose(f);
      hwrite(ohn, nr, nc, 1, 4); // always type 4, one band
    }
  }
  return 0;
}
