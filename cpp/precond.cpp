/* 20201101 precond.cpp: precondition data to set records with "bad" element to
0-vector. A somewhat arbitrary practice. More recently switched to converting
0-vector data to NAN! */
#include"misc.h"

float * dat; // input data bip format
size_t nr, nc, nb;

int main(int argc, char ** argv){
  if(argc < 2){
    printf("output is data where any record that includes nan or infinity, is replaced with 0.");
    err("precond [input file bip format] # precondition bip format data.");
  }
  str inf(argv[1]); // input file
  str hfn(hdr_fn(inf)); // input header
  size_t np, i, j, ix, k;
  hread(hfn, nr, nc, nb); // read input hdr
  np = nr * nc; // number of pixels
  dat = bread(inf, nr, nc, nb); // read bip data

  str ofn(inf + str("_precond.bin"));
  str ohn(inf + str("_precond.hdr"));
  printf("ofn %s\n", ofn.c_str());
  printf("ohn %s\n", ohn.c_str());
  hwrite(ohn, nr, nc, nb, 4);

  float x;
  size_t n = 0;
  size_t nz = 0;
  for0(i, nr){
    for0(j, nc){
      ix = i * nc + j;
      float * d = &dat[ix * nb];
      bool bad = false;
      for0(k, nb){
        x = d[k];
        if(isinf(-x) || isnan(-x) || isinf(x) || isnan(x)) bad = true;
      }
      if(bad){
        n ++;
        for0(k, nb) d[k] = 0.; // printf("ix %zu\n", ix);
      }
      bool z = true;
      for0(k, nb) if(d[k] != 0) z = false;
      if(z) nz ++;  // count zero vectors:
    }
  }
  bwrite(dat, ofn, nr, nc, nb); // write "preconditioned" data
  printf("number of bad px: %zu\n", n);
  printf("number of zoutpx: %zu\n", nz);
  return 0;
}
