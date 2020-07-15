/* Assuming image is floating point 32bit IEEE standard type,
with values in {0, 1} only:

perform dilate operation, i.e. grow "positive" areas */

#include"misc.h"
int main(int argc, char ** argv){

  if(argc < 3){
    err("binary_dilate [input binary class file] [window size] # [optional parameter to select erode mode]");
  }
  int dilate = argc < 4; // optional arg which, when present, selects erosion instead of dilate

  size_t nwin = (size_t) atoi(argv[2]); // window size, assert odd number
  if(nwin % 2 != 1) err("window size must be odd");
  size_t dx = (nwin - 1) / 2;

  str fn(argv[1]); // input file name
  str hfn(hdr_fn(fn)); // auto-detect header file name
  size_t nrow, ncol, nband, np;
  long int i, j, di, dj;
  hread(hfn, nrow, ncol, nband); // read header

  np = nrow * ncol;
  if(nband != 1) err("this program defines results for 1-band images");
  float * dat = bread(fn, nrow, ncol, nband); // read data into float array
  float * out = falloc(np); // result buffer area
  for0(i, np) out[i] = dat[i]; // initialize output with input values, then modify

  if(dilate){
    for0(i, nrow){
      for0(j, ncol){
        if(dat[i * ncol + j] == 1.){
          for(di = i - dx; di <= i + dx; di++){
            for(dj = j - dx; dj <= j + dx; dj++){
              out[di * ncol + dj] = 1.;
            }
          }
        }
      }
    }
  }
  else{
    for0(i, nrow){
      for0(j, ncol){
        if(dat[i * ncol + j] == 0.){
          for(di = i - dx; di <= i + dx; di++){
            for(dj = j - dx; dj <= j + dx; dj++){
              out[di * ncol + dj] = 0.;
            }
          }
        }
      }
    }

  }

  str ofn(fn + str("_dilate.bin")); // output product file
  str ohfn(fn + str("_dilate.hdr")); // output product header

  hwrite(ohfn, nrow, ncol, nband); // write header
  bwrite(out, ofn, nrow, ncol, nband); // write output product
  return 0;
}