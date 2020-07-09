/* recode a class map by counting up from one */
#include"misc.h"
int main(int argc, char ** argv){

  if(argc < 2) err("class_wheel [input binary (classification) file name]");

  str fn(argv[1]); // input file name
  str ofn(fn + str("_recode.bin"));
  str ohfn(fn +str("_recode.hdr"));

  str hfn(hdr_fn(fn)); // auto-detect header file name
  size_t nrow, ncol, nband, np, i, j;
  hread(hfn, nrow, ncol, nband); // read header

  np = nrow * ncol;
  if(nband != 1) err("this program defines results for 1-band images");

  // read data into float array
  float * dat = bread(fn, nrow, ncol, nband);

  // accumulate the data
  map<float, size_t> count;
  for0(i, np){
    if(count.count(dat[i]) < 1){
      count[dat[i]] = 0;
    }
    count[dat[i]] += 1;
  }

  float d;
  float ci = 1.;
  map<float, float> lookup;
  map<float, size_t>::iterator ii;

  for(it = count.begin(); it != count.end(); it++){
    lookup[it->first] = ci ++;
  }

  hwrite(ohfn, nrow, ncol, 1); // write header

  FILE * f = fopen(ofn.c_str(), "wb");
  for0(i, np){
    d = lookup[dat[i]];
    fwrite(&d, sizeof(float), 1, f);
  }
  fclose(f);
  
  free(dat);
  return 0;
}
