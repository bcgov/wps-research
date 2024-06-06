/* recode a class map by counting up from one, or 0
use this to make class labels contiguous again, after merging ops */
#include"misc.h"
int main(int argc, char ** argv){

  if(argc < 2) err("class_recode [input binary (classification) file name] # [optional extra arg that specifies start from zero, instead of 1]");
  int start_0 = argc > 2;
  int skip_0 = argc > 2;

  str fn(argv[1]); // input file name
  str ofn(fn + str("_recode.bin"));
  str ohfn(fn +str("_recode.hdr"));

  str hfn(hdr_fn(fn)); // auto-detect header file name
  size_t nrow, ncol, nband, np, i, j;
  hread(hfn, nrow, ncol, nband); // read header

  np = nrow * ncol;
  if(nband != 1) err("this program defines results for 1-band images");
  float * dat = bread(fn, nrow, ncol, nband); // read data into array  
	float * out = falloc(nrow * ncol) ;
  map<float, size_t> count; // accumulate the data
  for0(i, np){
    float d = dat[i];
    if(isnan(d)) continue;
    if(count.count(d) < 1) count[d] = 0;
    count[d] += 1;
  }

  float d;
  float ci = start_0 ? 0. : 1.; // default: start on 1
  map<float, float> lookup;
  map<float, size_t>::iterator it;
  for(it = count.begin(); it != count.end(); it++) lookup[it->first] = ci ++;
  hwrite(ohfn, nrow, ncol, 1); // write header

  FILE * f = fopen(ofn.c_str(), "wb");
  for0(i, np){
    d = dat[i];
    if(!(isnan(d))){
      d = lookup[dat[i]];
    }
  }
  fwrite(&d, sizeof(float), nrow * ncol, f);
  fclose(f);
  free(dat);
  return 0;
}
