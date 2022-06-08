/* 20201031 convert bip data to bsq */
#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 2) err("bip2bsq [input binary file name (assumed of interleave BIP)]\n");

  str ifn(argv[1]); // input filename
  str hfn(hdr_fn(ifn)); // input header
  size_t nr, nc, nb, np; // image dims
  hread(hfn, nr, nc, nb); // read image shape. In future, assert BIP format
  float * d = bread(ifn, nr, nc, nb); // read input image data
  np = nr * nc; // number of pix

  str ofn(ifn + str("_bsq.bin")); // output image
  str ohn(ifn + str("_bsq.hdr")); // output header
  hwrite(ohn, nr, nc, nb); // write output header. Should really be BSQ

  size_t i, j, k, ki, ii;
  float * d2 = falloc(np * nb);
  for0(k, nb){
    ki = k * np;
    for0(i, nr){
      ii = i * nc;
      for0(j, nc) d2[ki + ii + j] = d[((ii + j) * nb) + k];
    }
  }
  bwrite(d2, ofn, nr, nc, nb); // write bip data
  return 0;
}
