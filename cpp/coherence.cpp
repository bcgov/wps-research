/* 20230620 coherence of two complex number (ENVI type-6 format)
Check formula for denominator? */
#include"misc.h"

#define sq(x) x*x
int main(int argc, char ** argv){
  if(argc < 3){
    err("coherence; usage\n\t:coherence [file1: ENVI type 6] [file 2: ENVI type 6]");
  }
  size_t nrow, ncol, row, col, nband;
  long int i, j, k, ind;
  char * fn1 = argv[1];
  char * fn2 = argv[2];
  
  str hfn1(hdr_fn(str(fn1)));
  str hfn2(hdr_fn(str(fn2)));
  hread(hfn1, nrow, ncol, nband);
  hread(hfn2, nrow, ncol, nband);

  str ofn(str(fn1) + str("_") + str(fn2) + str("_coh.bin"));
  str ohn(str(fn1) + str("_") + str(fn2) + str("_coh.hdr"));
  float re1, im1, re2, im2, re3, im3, d1, d2;
  FILE * inf1 = fopen(fn1, "rb");
  if(!inf1) err("failed to open input file1");
  FILE * inf2 = fopen(fn2, "rb");
  if(!inf2) err("failed to open input file2");

  float * out = falloc(nrow * ncol * 2);
  size_t ci = 0;

  size_t nr;
  printf("here\n");
  for0(row, nrow){
    if(nrow % 100 ==0){
      printf("\rProcessing row %zu of %zu ", row + 1, nrow);
    }
    for0(col, ncol){
      nr += fread(&re1, sizeof(float), 1, inf1);
      nr += fread(&im1, sizeof(float), 1, inf1);
      nr += fread(&re2, sizeof(float), 1, inf2);
      nr += fread(&im2, sizeof(float), 1, inf2);

      re3 = re1 * re2 + im1 * im2;
      im3 = im1 * re2 - re1 * im2;
      d1 = float(sqrt(sq(re1) + sq(im1))) * float(sqrt(sq(re2) + sq(im2)));
      (out[ci++] = re3 / d1), (out[ci++] = im3 / d1);
    }
  }
  fclose(inf1);
  fclose(inf2);
  bwrite(out, ofn, nrow, ncol, 2);
  printf("\r");

  vector<str> bn;
  bn.push_back(str("band 1"));
  hwrite(ohn, nrow, ncol, 1, 4, bn);
  return 0;
}
