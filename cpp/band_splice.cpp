/* Assuming image is floating point 32bit IEEE standard type, BSQ interleave,
 replace a band with the single-band supplied */

#include"misc.h"
int main(int argc, char ** argv){

  if(argc < 4){
    err("band_splice [input BSQ image stack] [replacement band] [0-index of band to replace]");
  }
  str fn(argv[1]); // input file name, stack which will get one band replaced
  str hfn(hdr_fn(fn)); // auto-detect header-file name

  str fn2(argv[2]); // input file name, single band that will overwrite one band in stack 
  str hfn2(hdr_fn(fn2)); // auto-detect header-file name
  
  int bi = atoi(argv[3]); // band index
  if(bi < 0 || bi > nband) err("please verify index of band to splice");

  size_t nrow, ncol, nband, np, nrow2, ncol2, nband2;
  long int i, j, di, dj;
  hread(hfn, nrow, ncol, nband); // read header
  hread(hfn2, nrow2, ncol2, nband2); // read header

  np = nrow * ncol;
  if(nband2 != 1) err("this program expects the data to be spliced in, to be 1-band");
  if(nrow != nrow2 || ncol != ncol2) err("please verify input image dimensions");

  float * dat = bread(fn2, nrow2, ncol2, nband2); // read in band to splice
  printf("done splice\n");
  return 0;
}
