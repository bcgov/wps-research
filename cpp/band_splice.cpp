/* Assuming image is floating point 32bit IEEE standard type, BSQ interleave,
replace a band with the single-band supplied */

#include"misc.h"
int main(int argc, char ** argv){

  if(argc < 4){
    err("band_splice [input BSQ image stack] [replacement band] [1-index of band to replace]");
  }
  str fn(argv[1]); // input file name, stack which will get one band replaced
  str hfn(hdr_fn(fn)); // auto-detect header-file name

  str fn2(argv[2]); // input file name, single band that will overwrite one band in stack
  str hfn2(hdr_fn(fn2, true)); // auto-detect header-file name, don't assume exists

  size_t nrow, ncol, nband, np, nrow2, ncol2, nband2;
  long int i, j, di, dj;
  hread(hfn, nrow, ncol, nband); // read header
  if(exists(hfn2)){
    hread(hfn2, nrow2, ncol2, nband2); // read header, if exists
  }
  else{
    nrow2 = nrow; // else, assume replacement band has same dimensions
    ncol2 = ncol;
    nband2 = 1;
  }

  int bi = atoi(argv[3]) - 1; // band index
  if(bi < 0 || bi > nband) err("please verify 1-index of band to splice");
  printf("band 0-index %d\n", bi);
  
  np = nrow * ncol;
  printf("pixels %zu\n", np);
  if(nband2 != 1) err("this program expects the data to be spliced in, to be 1-band");
  if(nrow != nrow2 || ncol != ncol2) err("please verify input image dimensions");

  // check sizes of both files, match headers
  size_t fs1 = fsize(fn);
  size_t fs2 = fsize(fn2);
  if(fs1 != np * nband * sizeof(float)) err("please verify file size for input stack");
  if(fs2 != np * sizeof(float)) err("please verify file size for replacement band to splice");
  printf("file_size %zu\n", fs1);

  // do the splice
  float * dat = bread(fn2, nrow2, ncol2, nband2); // read in band to splice
  FILE * f = fopen(argv[1], "r+b"); // open stack with read/write access, then splice in
  size_t f_p = np * sizeof(float) * (size_t) bi; // splice location: should implement a shuffle version too
  printf("fp %zu\n", f_p);
  fseek(f, f_p, SEEK_SET); // goto splice location
  size_t nr = fwrite(dat, sizeof(float), np, f);  // random-access write
  printf("nr %zu\n", nr);
  fclose(f);

  // qa: assert file size unchanged
  size_t fs1p = fsize(fn);
  if(fs1p != fs1){
    printf("File size: %zu expected %zu\n", fs1p, fs1);
    err("splice failed");
  }

  // could also read the spliced band back in, to verify

  printf("done splice\n");
  return 0;
}
