/* 20240220 unstack bands, no prefix, to folder with name of stack

revised from unstack.cpp */
#include"misc.h"

int main(int argc, char *argv[]){
  if(argc < 2) err("unstack [input data file]");
  size_t i, j, nr, nc, nb, np;
  FILE * f;
  str ifn(argv[1]);
  str hfn(hdr_fn(ifn));
  hread(hfn, nr, nc, nb);
  vector<str> band_names(parse_band_names(hfn));

  // filename for stack
  vector<str> frags(split(ifn, '/'));
  str stem(frags[frags.size() -1]);
  frags = split(stem, '.');
  stem = frags[0];
  cout << "mkdir -p " << stem << endl;

  int retcode = system((str("mkdir -p ") + stem).c_str());

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

  float * d = bread(ifn, nr, nc, nb); // read input data
  for0(i, nb){
    //str pre(ifn + str("_") + zero_pad(to_string(i + 1), 3));
    str pre(stem + str("/"));
    if(use_bn) pre += band_names[i];
    str ofn(pre + str(".bin"));
    str ohn(pre + str(".hdr"));
    f = wopen(ofn.c_str());
    fwrite(&d[np * i], sizeof(float), np, f);
    fclose(f);
    hwrite(ohn, nr, nc, 1, 4); // always type 4, one band

    str cmd(str("python3 ~/GitHub/wps-research/py/envi_header_copy_mapinfo.py ") + hfn + str(" ") + ohn);
    cout << cmd << endl;
    system(cmd.c_str());
  }
  return 0;
}
