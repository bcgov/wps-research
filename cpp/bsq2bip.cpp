/* 20201031 convert bsq data to bip data */
#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 2) err("bsq2bip [input binary file name]\n");

  str ifn(argv[1]); // input filename
  str hfn(hdr_fn(ifn)); // input header
  size_t nr, nc, nb, np; // image dims
  hread(hfn, nr, nc, nb); // read image dims
  float * d = bread(ifn, nr, nc, nb); // read input image data
  np = nr * nc; // number of pix

  str ofn(ifn + str("_bip.bin")); // output image
  str ohn(ifn + str("_bip.hdr")); // output header
  hwrite(ohn, nr, nc, nb, 4); // write BIP format output header

  size_t i, j, k, ki, ii; 
  float * d2 = falloc(np * nb);
  for0(k, nb){
    ki = k * np;
    for0(i, nr){
      ii = i * nc;	
      for0(j, nc) d2[((ii + j) * nb) + k] = d[ki + ii + j];
    }
  }
  bwrite(d2, ofn, nr, nc, nb); // write bip data
  return 0;
}
