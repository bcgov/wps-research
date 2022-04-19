#include<stdlib.h>
#include<string.h>
#include<stdio.h>
#include<math.h>
#include<time.h>
#include"envi.h"
using namespace std;

int main(int argc, char ** argv){
  if(argc < 5) error("cp_2_t2.cpp: convert two complex channel (ENVI type 6), implemented 20210527. \n\tuse: cp_2_t2.exe [nrow] [ncol] [ch] [cv] \n\tNote: config.txt file must be present in input directory\n");
  
  size_t sf = sizeof(float);
  int nrow, ncol, row, col, i, j, k, ind;
  nrow = atoi(argv[1]);
  ncol = atoi(argv[2]);
  char * infn = argv[3];
  char * inf2 = argv[4];
  printf("nrow %d ncol %d infile %s\n", nrow, ncol, infn);

  string outfn(string(infn) + string(".abs"));
  float a, b, c, d;


  FILE * infile = open(infn);
  if(!infile){
    printf("Error: could not open file %s\n", infn);
    exit(1);
  }

  FILE * infil2 = open(inf2);
  if(!infil2){
    printf("Error: could not open file %s\n", inf2);
    exit(1);
  }

  FILE * t11 = wopen("T11.bin");
  if(!t11){
    printf("Error: could not open file T11.bin");
    exit(1);
  }
  FILE * t22 = wopen("T22.bin");
  FILE * t12_r = wopen("T12_real.bin");
  FILE * t12_i = wopen("T12_imag.bin");


  for0(row, nrow){
    if(nrow % 100 ==0) printf("\rProcessing row %d of %d ", row + 1, nrow);
    for0(col, ncol){
      fread(&a, sf, 1, infile);
      fread(&b, sf, 1, infile);
      fread(&c, sf, 1, infil2);
      fread(&d, sf, 1, infil2);
      
      float T11 = a*a + b*b;
      float T22 = c*c + d*d;
      float T12_r = a* c + d* b;
      float T12_i = b* c - d* a;

      fwrite(&T11, sf, 1, t11);
      fwrite(&T22, sf, 1, t22);
      fwrite(&T12_r, sf, 1, t12_r);
      fwrite(&T12_i, sf, 1, t12_i);
    }
  }
  printf("\r");
  fclose(t11);
  fclose(t22);
  fclose(t12_r);
  fclose(t12_i);
  write_envi_hdr(outfn + string(".hdr"), nrow, ncol);
  return 0;
}
