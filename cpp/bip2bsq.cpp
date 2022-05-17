/* 20220516 convert bip to bsq */
#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 2) err("bip2bsq [input raster]");
  str fn(argv[1]);
  str hfn(hdr_fn(fn));
  size_t np, nr, nc, nb, i, k, ci;
  hread(hfn, nr, nc, nb);
  np = nr * nc;

  float * dat = bread(fn, nr, nc, nb); // read the input data
  float * out = falloc(np * nb);
  str ofn(fn + str("_bsq.bin"));
  str ohn(fn + str("_bsq.hdr"));

  ci = 0;
  for0(i, np)
    for0(k, nb)
      out[k * np + i] = dat[ci++];

  bwrite(out, ofn, nr, nc, nb);
  hwrite(ohn, nr, nc, nb);
  return 0;
}
