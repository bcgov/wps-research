#include"misc.h"
using namespace std;

int main(int argc, char ** argv){
  if(argc < 2){
    error("abs.cpp: magnitude of complex channel (ENVI type 6); usage\n\t:abs [file: ENVI type 6]\n\tNote: config.txt file must be present in input directory\n");
  }
  long int nrow, ncol, row, col, i, j, k, ind;
  nrow = atol(argv[1]);
  ncol = atol(argv[2]);
  char * infn = argv[3];
  str hfn(hdr_fn(str(infn));
  hread(hfn, nrow, ncol, nband);
  printf("nrow %d ncol %d infile %s\n", nrow, ncol, infn);

  str ofn(str(infn) + str("_abs.bin"));
  str ohn(str(infn) + str("_abs.hdr"));
  float real, imag;
  FILE * infile = ropen(ofn);
  FILE * outfile = wopen(outfn);
  float * out = falloc(nrow * ncol);
  size_t ci = 0;

  for0(row, nrow){
    if(nrow % 100 ==0){
      printf("\rProcessing row %d of %d ", row + 1, nrow);
    }
    for0(col, ncol){
      fread(&real, sf, 1, infile);
      fread(&imag, sf, 1, infile);
      out[ci++] = (float)(sqrt(sq(real) + sq(imag)));
    }
  }

  // bwrite(float * d, str bfn, size_t nrow, size_t ncol, size_t nband
  size_t nw = fwrite(out, outfile, nrow, ncol, 1);
  printf("\r");
  fclose(outfile);

  vector<str> bn;
  bn.push_back(str("band 1"));
  hwrite(outfn + string(".hdr"), nrow, ncol, 1, bn);
  return 0;
}
