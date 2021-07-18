#include<stdlib.h>
#include<string.h>
#include<stdio.h>
#include<math.h>
#include<time.h>
#include"misc.h"
using namespace std;

int main(int argc, char ** argv){
  if(argc < 2) err("convert_cplx_to_iq.cpp: convert a complex channel (ENVI type 6) into real(i) and imag(q) components, implemented 20210717. \n\tuse: convert_cplx_to_iq [ch] \n\tNote: config.txt file must be present in input directory\n");
  
  size_t sf = sizeof(float);
  size_t nrow, ncol, row, col, i, j, k, ind, nb;
  char * ifn = argv[1];
  str ihfn(hdr_fn(ifn));
  hread(ihfn, nrow, ncol, nb); // read header file for ENVI type 6 input file
  printf("nrow %d ncol %d infile %s\n", nrow, ncol, ifn);

  str ofn1(str(ifn) + str("_i.bin"));
  str ofn2(str(ifn) + str("_q.bin"));
  float a, b, c, d;

  FILE * if1 = ropen(ifn);
  FILE * of1 = wopen(ofn1);
  FILE * of2 = wopen(ofn2);
  
  for0(row, nrow){
    if(nrow % 100 ==0) printf("\rProcessing row %d of %d ", row + 1, nrow);
    for0(col, ncol){
      fread( &a, sf, 1, if1);
      fread( &b, sf, 1, if1);
      fwrite(&a, sf, 1, of1);
      fwrite(&b, sf, 1, of2);
    }
  }
  printf("\r");
  hwrite(str(ifn) + str("_i.hdr"), nrow, ncol, 1);
  hwrite(str(ifn) + str("_q.hdr"), nrow, ncol, 1);
  return 0;
}
