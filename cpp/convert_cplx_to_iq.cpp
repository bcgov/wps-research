#include<stdlib.h>
#include<string.h>
#include<stdio.h>
#include<math.h>
#include<time.h>
#include"misc.h"
using namespace std;

int main(int argc, char ** argv){
  if(argc < 4) error("convert_cplx_to_iq.cpp: convert a complex channel (ENVI type 6) into real(i) and imag(q) components, implemented 20210717. \n\tuse: convert_cplx_to_iq [nrow] [ncol] [ch] \n\tNote: config.txt file must be present in input directory\n");
  
  size_t sf = sizeof(float);
  int nrow, ncol, row, col, i, j, k, ind;
  nrow = atoi(argv[1]);
  ncol = atoi(argv[2]);
  char * ifn = argv[3];
  printf("nrow %d ncol %d infile %s\n", nrow, ncol, ifn);

  str ofn1(str(ifn) + str("_i.bin"));
  str ofn2(str(ifn) + str("_q.bin"));
  float a, b, c, d;

  FILE * if1 = ropen(infn);
  FILE * of1 = wopen(ofn1);
  FILE * of2 = wopen(ofn2);
  
  for0(row, nrow){
    if(nrow % 100 ==0) printf("\rProcessing row %d of %d ", row + 1, nrow);
    for0(col, ncol){
      fread( &a, sf, 1, infile); fread( &b, sf, 1, infile);
      fwrite(&a, sf, 1,    of1); fwrite(&b, sf, 1,    of2);
    }
  }
  printf("\r");
  write_envi_hdr(str(ifn) + str("_i.hdr"), nrow, ncol);
  write_envi_hdr(str(ifn) + str("_q.hdr"), nrow, ncol);
  return 0;
}
