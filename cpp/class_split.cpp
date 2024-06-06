/* 20240605 split a class map into one file per class
*/
#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 2) err("class_split [input binary (classification) file name]");

  str fn(argv[1]); // input file name
  str hfn(hdr_fn(fn)); // auto-detect header file name
  size_t nrow, ncol, nband, np, i, j;
  hread(hfn, nrow, ncol, nband); // read header

  np = nrow * ncol;
  if(nband != 1) err("this program defines results for 1-band images");

  float * dat = bread(fn, nrow, ncol, nband); // read data into array
  map<float, size_t> count; // accumulate the data
  for0(i, np){
    float d = dat[i];
    if(count.count(d) < 1) count[d] = 0;
    count[d] += 1;
  }

  float * out = falloc(nrow * ncol);

  for(map<float, size_t>::iterator it = count.begin(); it != count.end(); it++){
    float w = it->first;
    for0(i, np){
      float d = dat[i];
      if(w == d) out[i] = 1.;
      else out[i] = 0.;
    } 
    bwrite(out, fn + std::to_string(w) + str(".bin"), nrow, ncol, 1);
    hwrite(fn + std::to_string(w) + str(".hdr"), nrow, ncol, 1);
  }
  free(dat);
  return 0;
}
