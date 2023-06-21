#include"misc.h"

#define sq(x) x*x
int main(int argc, char ** argv){
  if(argc < 2){
    err("abs.cpp: magnitude of complex channel (ENVI type 6); usage\n\t:abs [file: ENVI type 6]\n\tNote: config.txt file must be present in input directory\n");
  }
  size_t nrow, ncol, row, col, nband;
  long int i, j, k, ind;
  char * infn = argv[1];
  str hfn(hdr_fn(str(infn)));
  hread(hfn, nrow, ncol, nband);
  printf("nrow %zu ncol %zu infile %s\n", nrow, ncol, infn);

  str ofn(str(infn) + str("_abs.bin"));
  str ohn(str(infn) + str("_abs.hdr"));
  float real, imag;
  FILE * infile = fopen(infn, "rb");
  if(!infile) err("failed to open input file");
  
  float * out = falloc(nrow * ncol);
  size_t ci = 0;

  printf("here\n");
  for0(row, nrow){
    if(nrow % 100 ==0){
      printf("\rProcessing row %zu of %zu ", row + 1, nrow);
    }
    for0(col, ncol){
      fread(&real, sizeof(float), 1, infile);
      fread(&imag, sizeof(float), 1, infile);
      out[ci++] = (float)(sqrt(sq(real) + sq(imag)));
    }
  }
  fclose(infile);
  // bwrite(float * d, str bfn, size_t nrow, size_t ncol, size_t nband
  bwrite(out, ofn, nrow, ncol, 1);
  printf("\r");

  vector<str> bn;
  bn.push_back(str("band 1"));
  hwrite(ohn, nrow, ncol, 1, 4, bn);
  return 0;
}
