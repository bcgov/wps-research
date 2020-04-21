// drop specified bands from a hyperspectral image (don't write header)
#include"misc.h"
#include<set>

int main(int argc, char ** argv){
  if(argc < 4) err("rm_band [input binary file name] [output binary file name] [band index from 1] .. [more band index from 1, if desired]");

  str fn(argv[1]); // input file name
  str of(argv[2]); // output file name

  set<int> drop;
  for(int i = 3; i < argc; i++){
    printf("drop band %d\n", atoi(argv[i]));
    drop.insert(atoi(argv[i]));
  }

  str hfn(hdr_fn(fn)); // auto-detect header file name
  size_t nrow, ncol, nband, np, i, j, k;
  hread(hfn, nrow, ncol, nband); // read header
  np = nrow * ncol; // number of input pix

  float * dat = bread(fn, nrow, ncol, nband); // load floats to array

  FILE * g = fopen(of.c_str(), "wb");
  if(!g) err("failed to open output file");
  printf("+w %s\n", of.c_str());

  for0(i, nband){
    int i1 = i + 1;
    if(drop.find(i1) != drop.end()){
      // drop this band
    }
    else{
      // keep this band!
      printf(" +w band %zu\n", i);
      size_t w = fwrite(&dat[i * np], sizeof(float), np, g);
    }
  }
  fclose(g);
  free(dat);
  return 0;
}