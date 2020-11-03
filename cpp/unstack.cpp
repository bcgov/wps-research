#include"misc.h"

int main(int argc, char *argv[]){
  if(argc < 2) err("unstack [input data file]");

  FILE * f;
  str ifn(argv[1]);
  str hfn(hdr_fn(ifn));
  size_t i, j, nr, nc, nb, np;
  hread(hfn, nr, nc, nb);
  np = nr * nc;

  float * d = bread(ifn, nr, nc, nb); // read input data

  for0(i, nb){
    str obfn(ifn + str("_") + to_string(i + 1) + str(".bin"));
    str obhn(ifn + str("_") + to_string(i + 1) + str(".hdr"));
    f = wopen(obfn.c_str());
    fwrite(&d[np * i], sizeof(float), np, f);
    fclose(f);
    hwrite(obhn, nr, nc, nb, 4); // always type 4!
  }
  return 0;
}
