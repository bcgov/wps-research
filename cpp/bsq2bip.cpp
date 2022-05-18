/* 20220516 convert bsq to bip */
#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 2) err("bsq2bip [input raster]");
  str fn(argv[1]);
  str hfn(hdr_fn(fn));
  size_t np, nr, nc, nb, i, k, ci;
  hread(hfn, nr, nc, nb);
  np = nr * nc;

  float * dat = bread(fn, nr, nc, nb); // read the input data
  float * out = falloc(np * nb);
  str ofn(fn + str("_bip.bin"));
  str ohn(fn + str("_bip.hdr"));

  ci = 0;
  for0(i, np)
    for0(k, nb)
      out[ci++] = dat[k * np + i];
  
  bwrite(out, ofn, nr, nc, nb);
  hwrite(ohn, nr, nc, nb);
  return 0;
}
