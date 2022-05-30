/* Band unstacking. Should revise this with band names now
20220308 revised to accept specific bands only */
#include"misc.h"

int main(int argc, char *argv[]){
  if(argc < 2){
    err("unstack [input data file] [optional arg: band 1-ix] ... [optional: band 1-ix]");
  }
  size_t i, j, nr, nc, nb, np;
  set<int> selected;
  for(i = 2; i < argc; i++) selected.insert(atoi(argv[i]));
  
  FILE * f;
  str ifn(argv[1]);
  str hfn(hdr_fn(ifn));
  hread(hfn, nr, nc, nb);
  vector<str> band_names(parse_band_names(hfn));

  /* don't include band names in output filenames
      if band names look like folders*/
  int use_bn = true;
  for(vector<str>::iterator it = band_names.begin();
      it != band_names.end();
      it++){
    str x(*it);
    vector<str> y(split(x, ' '));
    vector<str> z(split(x, '/'));
    if(y.size() > 1 || z.size() > 1) use_bn = false;
  }

  np = nr * nc;
  for(set<int>::iterator it = selected.begin();
      it != selected.end();
      it ++){
    if(*it < 1 || *it > nb){
      err("selected band out of bounds");
    }
  }

  float * d = bread(ifn, nr, nc, nb); // read input data
  for0(i, nb){
    if(selected.size() < 1 || selected.count((int)(i + 1)) > 0){
      str pre(ifn + str("_") + zero_pad(to_string(i + 1), 3));
      if(use_bn){
        pre += (str("_") + band_names[i]);
      }
      str ofn(pre + str(".bin"));
      str ohn(pre + str(".hdr"));
      f = wopen(ofn.c_str());
      fwrite(&d[np * i], sizeof(float), np, f);
      fclose(f);
      hwrite(ohn, nr, nc, 1, 4); // always type 4, one band

      str cmd(str("python3 ~/GitHub/wps-research/py/envi_header_copy_mapinfo.py ") + 
	      hfn + str(" ") +
	      ohn);
      cout << cmd << endl;
      system(cmd.c_str());
    }
  }
  return 0;
}
