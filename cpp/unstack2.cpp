/* 20230620: Band unstacking. 

This version extracts a single 0-indexed band, using random file access to seek the band, 
and writes it to a specified location */
#include"misc.h"

int main(int argc, char *argv[]){
  if(argc < 4){
    err("unstack [input data file] [band 1-ix] [output binary file locaion]");
  }
  size_t i, j, nr, nc, nb, np;
  int selected = atoi(argv[2]);
  str ofn(argv[3]);

  str fn(argv[1]);
  str hfn(hdr_fn(fn));
  hread(hfn, nr, nc, nb);
  np = nr * nc;

  float * out = falloc(np);  // hold one band
  vector<str> band_names(parse_band_names(hfn));

  FILE * f = ropen(fn.c_str());
  size_t target = np * (size_t)selected *sizeof(float);
  fseek(f, target, SEEK_SET);
  if(ftell(f) != target) err("seek failed");
  size_t rs = fread(out, sizeof(float), np, f);
  if(rs != np) err("did not read expected");
  fclose(f);

  FILE * g = wopen(ofn.c_str());
  rs = fwrite(out, sizeof(float), np, g);
  if(rs != np) err("did not write expected");
  fclose(g);

  vector<str> bn2;
  bn2.push_back(band_names[selected]);
  str ohn(hdr_fn(ofn, true));
  hwrite(ohn, nr, nc, 1, 4, bn2); // always type 4, one band
  str cmd(str("envi_header_copy_mapinfo.py ") + 
	      hfn + str(" ") +
	      ohn);
  cout << cmd << endl;
  run(cmd);
  return 0;
}
