/* Assuming image is floating point 32bit IEEE standard type,
with values in {0, 1} only:

perform dilate operation, i.e. grow "positive" areas */

#include"misc.h"
int main(int argc, char ** argv){

  if(argc < 3){
    err("binary_dilate [input binary class file] [window size] # [optional parameter to select erode mode]");
  }
  int dilate = argc < 4; // optional arg which, when present, selects erosion instead of dilate
  printf("%s\n", dilate ? "dilate" : "erode");

  size_t nwin = (size_t) atoi(argv[2]); // window size, assert odd number
  if(nwin % 2 != 1) err("window size must be odd");
  long int dx = (nwin - 1) / 2;
  printf("nwin %ld\ndx %ld\n", nwin, dx);
  printf("input %s\n", argv[1]);

  str fn(argv[1]); // input file name
  str hfn(hdr_fn(fn)); // auto-detect header file name
  size_t nrow, ncol, nband, np;
  long int i, j, di, dj;
  hread(hfn, nrow, ncol, nband); // read header

  np = nrow * ncol;
  if(nband != 1) err("this program defines results for 1-band images");
  float * dat = bread(fn, nrow, ncol, nband); // read data into float array
  float * out = falloc(np); // result buffer area

  for0(i, np){
    if(!(dat[i] == 0. || dat[i] == 1.)){
      err("assert data in {0, 1}, failed");
    }
  }

  for0(i, np) out[i] = dat[i]; // initialize output with input values, then modify

  size_t n_dilate, n_erode;
  n_dilate = n_erode = 0;

  if(dilate){
    for0(i, nrow){
      if(i % 1000 == 0) printf("i=%ld of %zu\n", i, nrow);
      for0(j, ncol){
        if(dat[i * ncol + j] == 1.){
          n_dilate ++;
          for(di = i - dx; di <= i + dx; di++){
	    if(di < 0 || di > ncol) continue;
            for(dj = j - dx; dj <= j + dx; dj++){
	      if(dj < 0 || dj > nrow) continue;
              out[di * ncol + dj] = 1.;
            }
          }
        }
      }
    }
  }
  else{
    for0(i, nrow){
      if(i % 1000 == 0) printf("i=%ld of %zu\n", i, nrow);
      for0(j, ncol){
        if(dat[i * ncol + j] == 0.){
          n_erode ++;
          for(di = i - dx; di <= i + dx; di++){
	    if(di < 0 || di > ncol) continue;
            for(dj = j - dx; dj <= j + dx; dj++){
	      if(dj < 0 || dj > nrow) continue;
              out[di * ncol + dj] = 0.;
            }
          }
        }
      }
    }

  }
  printf("n_dilate %zu n_erode %zu\n", n_dilate, n_erode);

  str ofn(fn + str("_dilate.bin")); // output product file
  str ohfn(fn + str("_dilate.hdr")); // output product header
  printf("output %s\n", ofn.c_str());

  hwrite(ohfn, nrow, ncol, nband); // write header
  bwrite(out, ofn, nrow, ncol, nband); // write output product
  return 0;
}
