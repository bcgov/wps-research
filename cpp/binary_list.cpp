/* list points where a float mask is equal to 1. */
#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 2) err("binary_list [input binary class file name]");

  float d;
  str fn(argv[1]); // input file name
  str hfn(hdr_fn(fn)); // auto-detect header file name
  size_t nrow, ncol, nband, np, i, j;
  hread(hfn, nrow, ncol, nband); // read header

  np = nrow * ncol;
  if(nband != 1) err("this program defines results for 1-band images");

  // read data into float array
  float * dat = bread(fn, nrow, ncol, nband);

  for0(i, nrow){
    for0(j, ncol){
      if(dat[i * ncol + j] == 1.){
        cout << i << " " << j << " ";
      }
    }
  }
  cout << endl;
  return 0;
}
